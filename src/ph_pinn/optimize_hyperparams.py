"""
optimize_hyperparams.py
-----------------------
This script uses the Optuna framework to perform an automated hyperparameter search for the Combined_PH_PINN model defined in pH_PINN.py.
It defines an "objective" function that Optuna repeatedly calls with different hyperparameter combinations.
Each call (a "trial") trains the model for a fixed number of epochs and returns the best validation loss.
Optuna uses these results to intelligently search for the optimal set of hyperparameters.
Results of the study are saved to a SQLite database file (optimize_hyperparams.db) in the 'results/PINN Data/' directory, allowing the optimization to be paused and resumed.

To run this script:
1. Make sure you have Optuna and its storage dependencies installed:
   pip install optuna "optuna[storages]"

2. Place this file in the `src/ph_pinn/` directory.

3. Run from the root directory of your project:
   python -m src.ph_pinn.optimize_hyperparams --objective val_loss
   OR
   python -m src.ph_pinn.optimize_hyperparams --objective h_loss

To view the results dashboard after running (run from project root):
   optuna-dashboard "sqlite:///results/PINN Data/optimize_hyperparams.db"
"""
import os
import sys
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import optuna
import argparse

# Ensure the script can find other modules in the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Import necessary components from your existing files ---
from src.hr_model.model import DEFAULT_PARAMS
from src.ph_pinn.pH_PINN import (
    Combined_PH_PINN,
    generate_data,
    normalize,
    train_step,
    loss_fn as evaluate_model # Use loss_fn for evaluation as it returns components
)

# JAX configuration
jax.config.update("jax_enable_x64", True)


def objective(trial, epochs_per_trial, static_data, objective_metric):
    """
    The main objective function that Optuna will minimize.
    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameters.
        epochs_per_trial (int): The number of epochs to train for during each trial.
        static_data (dict): A dictionary containing all pre-processed data and stats.
        objective_metric (str): The metric to optimize ('val_loss' or 'h_loss').
    Returns:
        float: The best value of the chosen objective metric achieved during the trial.
    """
    # --- 1. Suggest Hyperparameters from the Search Space ---
    # Retrieve the fixed master key from static_data
    master_key = static_data['master_key']

    # Use the fixed key for model initialization and create a subkey for epoch shuffling
    model_key, epoch_key = jax.random.split(master_key)


    # StateNN Fourier Features
    mapping_size = trial.suggest_categorical("mapping_size", [32, 64, 128, 256, 512])
    scale = trial.suggest_float("scale", 10, 1000, log=True)

    # Network Architectures (width and depth for each component)
    state_width = trial.suggest_categorical("state_width", [128, 256, 512, 1024])
    state_depth = trial.suggest_int("state_depth", 2, 6)
    h_width = trial.suggest_categorical("h_width", [32, 64, 128, 256, 512])
    h_depth = trial.suggest_int("h_depth", 1, 4)
    d_width = trial.suggest_categorical("d_width", [2, 4, 8, 16, 32, 64, 128])
    d_depth = trial.suggest_int("d_depth", 1, 4)
    j_width = trial.suggest_categorical("j_width", [2, 4, 8, 16, 32, 64, 128])
    j_depth = trial.suggest_int("j_depth", 1, 4)
    epsilon = trial.suggest_float("epsilon", 0.01, 5)

    # Optimizer
    initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-2, log=True)
    decay_steps = trial.suggest_int("decay_steps", 500, 3000)

    # Training and Loss
    batch_size = trial.suggest_categorical("batch_size", [2000, 4000, 8000, 16000, 32000, 64000])
    lambda_conservative_max = trial.suggest_float("lambda_conservative_max", 0.1, 30)
    lambda_dissipative_max = trial.suggest_float("lambda_dissipative_max", 0.1, 30)
    lambda_physics_max = trial.suggest_float("lambda_physics_max", 0.1, 30)
    lambda_j_structure_max = trial.suggest_float("lambda_j_structure_max", 0.1, 30)
    lambda_r_structure_max = trial.suggest_float("lambda_r_structure_max", 0.1, 30)
    lambda_phys_res_max = trial.suggest_float("lambda_phys_res_max", 0.1, 30)
    lambda_warmup_epochs = trial.suggest_int("lambda_warmup_epochs", 500, 3000)

    # --- 2. Build Model and Optimizer with Suggested Values ---
    nn_config = {
        "state_net": {
            "out_size": static_data['x_dim'] + static_data['e_dim'],
            "hidden_sizes": [state_width] * state_depth,
            "fourier_features": {"in_size": 1, "mapping_size": mapping_size, "scale": scale}
        },
        "hamiltonian_net": {
            "hidden_sizes": [h_width] * h_depth, "epsilon": epsilon
        },
        "dissipation_net": {
            "hidden_sizes": [d_width] * d_depth
        },
        "j_net": {
            "hidden_sizes": [j_width] * j_depth
        },
        "activation": jax.nn.softplus,
    }

    model = Combined_PH_PINN(key=model_key, config=nn_config, state_dim=static_data['e_dim'])

    lr_schedule = optax.linear_schedule(
        init_value=initial_learning_rate, end_value=1e-5, transition_steps=decay_steps
    )
    optimizer = optax.adamw(learning_rate=lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- 3. Run the Training Loop ---
    best_val_loss = jnp.inf
    best_train_h_loss = jnp.inf
    num_batches = static_data['t_train_norm'].shape[0] // batch_size
    if num_batches == 0: num_batches = 1

    for epoch in range(epochs_per_trial):
        warmup = jnp.minimum(1.0, (epoch + 1) / lambda_warmup_epochs)
        lambdas = {
            "lambda_conservative": lambda_conservative_max * warmup,
            "lambda_dissipative": lambda_dissipative_max * warmup,
            "lambda_physics": lambda_physics_max * warmup,
            "lambda_j_structure": lambda_j_structure_max * warmup,
            "lambda_r_structure": lambda_r_structure_max * warmup,
            "lambda_phys_res": lambda_phys_res_max * warmup,
        }

        # Split the key for shuffling, ensuring a different shuffle per epoch
        # but the sequence is the same for every trial
        epoch_key, shuffle_key = jax.random.split(epoch_key)
        perm = jax.random.permutation(shuffle_key, static_data['t_train_norm'].shape[0])
        t_s, e_s, x_s, edot_s, xdot_s, H_s = (
            static_data[k][perm] for k in ['t_train_norm', 'e_train_norm', 'x_train_norm',
                                           'e_dot_train_norm', 'x_dot_train_norm', 'H_train_norm']
        )
        epoch_train_h_loss = 0.0
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            t_b, e_b, x_b, edot_b, xdot_b, H_b = (
                arr[start:end] for arr in [t_s, e_s, x_s, edot_s, xdot_s, H_s]
            )

            model, opt_state, _, loss_comps = train_step(
                model, opt_state, optimizer, t_b, e_b, x_b, edot_b, xdot_b, H_b,
                **lambdas, **static_data['static_params']
            )
            epoch_train_h_loss += loss_comps['hamiltonian']

        avg_epoch_train_h_loss = epoch_train_h_loss / num_batches

        if avg_epoch_train_h_loss < best_train_h_loss:
            best_train_h_loss = avg_epoch_train_h_loss

        # --- Evaluate for validation loss (optional, but good for monitoring) ---
        val_loss, _ = evaluate_model(
            model, static_data['t_val_norm'], static_data['e_val_norm'],
            static_data['x_val_norm'], static_data['e_dot_val_norm'],
            static_data['x_dot_val_norm'], static_data['H_val_norm'],
            **lambdas, **static_data['static_params']
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # --- 4. Return the Final Metric to Optuna ---
    if objective_metric == 'h_loss':
        return float(best_train_h_loss)
    return float(best_val_loss)


def main():
    """Main execution block to set up and run the Optuna study."""
    # --- 0. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization for the pH-PINN model.")
    parser.add_argument(
        '--objective',
        type=str,
        default='h_loss',
        choices=['val_loss', 'h_loss'],
        help="The objective metric to minimize ('val_loss' or 'h_loss')."
    )
    args = parser.parse_args()
    print(f"Starting optimization with objective: {args.objective}")

    # --- 1. Load and Prepare Data (Done Once) ---
    print("Loading and preparing data for optimization...")
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'PINN Data', 'error_system_data.pkl')
    t, e, x, e_dot, x_dot, H, _ = generate_data(data_path)
    if t is None:
        sys.exit("Exiting: Data loading failed.")

    # Define a single master seed for all operations
    master_seed = 42
    master_key = jax.random.PRNGKey(master_seed)

    validation_split = 0.2
    num_samples = e.shape[0]

    # Use the master key for the initial train/val split
    split_key, _ = jax.random.split(master_key)
    perm = jax.random.permutation(split_key, num_samples)

    data_arrays = [t[perm].reshape(-1, 1), e[perm], x[perm], e_dot[perm], x_dot[perm], H[perm]]

    split_idx = int(num_samples * (1 - validation_split))
    train_data, val_data = zip(*(jnp.split(arr, [split_idx]) for arr in data_arrays))

    t_train, e_train, x_train, e_dot_train, x_dot_train, H_train = train_data
    t_val, e_val, x_val, e_dot_val, x_dot_val, H_val = val_data

    # --- Calculate and package normalization statistics ---
    stats = { #
        't': (jnp.mean(t_train), jnp.std(t_train)),
        'e': (jnp.mean(e_train, axis=0), jnp.std(e_train, axis=0)),
        'x': (jnp.mean(x_train, axis=0), jnp.std(x_train, axis=0)),
        'e_dot': (jnp.mean(e_dot_train, axis=0), jnp.std(e_dot_train, axis=0)),
        'x_dot': (jnp.mean(x_dot_train, axis=0), jnp.std(x_dot_train, axis=0)),
        'H': (jnp.mean(H_train), jnp.std(H_train)),
    }

    # --- Package all data and parameters for the objective function ---
    static_data = {
        'master_key': master_key, # Pass the master key to the objective
        'e_dim': e_train.shape[1], 'x_dim': x_train.shape[1],
        't_train_norm': normalize(t_train, *stats['t']),
        'e_train_norm': normalize(e_train, *stats['e']),
        'x_train_norm': normalize(x_train, *stats['x']),
        'e_dot_train_norm': normalize(e_dot_train, *stats['e_dot']),
        'x_dot_train_norm': normalize(x_dot_train, *stats['x_dot']),
        'H_train_norm': normalize(H_train, *stats['H']),
        't_val_norm': normalize(t_val, *stats['t']),
        'e_val_norm': normalize(e_val, *stats['e']),
        'x_val_norm': normalize(x_val, *stats['x']),
        'e_dot_val_norm': normalize(e_dot_val, *stats['e_dot']),
        'x_dot_val_norm': normalize(x_dot_val, *stats['x_dot']),
        'H_val_norm': normalize(H_val, *stats['H']),
        'static_params': {
            'hr_params': {**DEFAULT_PARAMS, 'ge': 0.62},
            'I_ext': jnp.array([0.8, 0.8]),
            'xi': jnp.array([[0, 1], [1, 0]]),
            't_mean': stats['t'][0], 't_std': stats['t'][1],
            'e_mean': stats['e'][0], 'e_std': stats['e'][1],
            'x_mean': stats['x'][0], 'x_std': stats['x'][1],
            'e_dot_mean': stats['e_dot'][0], 'e_dot_std': stats['e_dot'][1],
            'x_dot_mean': stats['x_dot'][0], 'x_dot_std': stats['x_dot'][1],
            'H_mean': stats['H'][0], 'H_std': stats['H'][1],
        }
    }

    # --- 2. Create and Run the Optuna Study ---
    print("\nStarting Optuna hyperparameter search...")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data')
    os.makedirs(results_dir, exist_ok=True)
    db_path = os.path.join(results_dir, "optimize_hyperparams.db")
    storage_name = f"sqlite:///{db_path}"
    # Use different study names to avoid conflicts
    study_name = f"sphnn_pinn_optimization_{args.objective}"

    study = optuna.create_study(
        study_name=study_name, storage=storage_name,
        direction="minimize", load_if_exists=True
    )

    objective_with_args = lambda trial: objective(trial, epochs_per_trial=500, static_data=static_data, objective_metric=args.objective)
    study.optimize(objective_with_args, n_trials=100)

    # --- 3. Print and Save the Results ---
    print("\nOptimization finished.")
    print(f"Study results are saved in: {storage_name}")
    print(f"To view dashboard, run: optuna-dashboard {storage_name}")

    best_trial = study.best_trial
    print("\n" + "="*40)
    print("         Best Trial Found")
    print("="*40)
    print(f"  Value (Best {args.objective}): {best_trial.value:.6f}")
    print("  Best Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    '{key}': {value},")
    print("="*40)


if __name__ == "__main__":
    main()