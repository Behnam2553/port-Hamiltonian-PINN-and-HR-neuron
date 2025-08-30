"""
optimize_hyperparams.py
-----------------------
This script uses the Optuna framework to perform an automated hyperparameter
search for the Combined_sPHNN_PINN model defined in pH_PINN.py.

It defines an "objective" function that Optuna repeatedly calls with different
hyperparameter combinations. Each call (a "trial") trains the model for a
fixed number of epochs and returns the best validation loss. Optuna uses
these results to intelligently search for the optimal set of hyperparameters.

Results of the study are saved to a SQLite database file (optimize_hyperparams.db)
in the 'results/PINN Data/' directory, allowing the optimization to be paused
and resumed.

To run this script:
1. Make sure you have Optuna and its storage dependencies installed:
   pip install optuna
   pip install "optuna[storages]"
2. Place this file in the `src/sph_pinn/` directory.
3. Run from the root directory of your project: python -m src.sph_pinn.optimize_hyperparams

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

# Ensure the script can find other modules in the project
# This adds the project's root directory to the Python path.
# Assumes the script is run from the root directory as recommended.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Import necessary components from your existing files ---
from src.hr_model.model import DEFAULT_PARAMS
from src.sph_pinn.pH_PINN import (
    StateNN, HamiltonianNN, DissipationNN, DynamicJ_NN, # Import component networks
    generate_data,
    normalize,
    denormalize,
    train_step,
    evaluate_model,
)

# JAX configuration
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. LOCAL MODEL DEFINITION FOR HYPERPARAMETER FLEXIBILITY
# ==============================================================================

# We redefine the combined model here to easily pass architectural hyperparameters
# from the Optuna trial into the sub-networks.
class Combined_sPHNN_PINN(eqx.Module):
    """Main model combining a unified state predictor and sPHNN structure."""
    state_net: StateNN
    hamiltonian_net: HamiltonianNN
    dissipation_net: DissipationNN
    j_net: DynamicJ_NN

    def __init__(self, key, config):
        state_key, h_key, d_key, j_key = jax.random.split(key, 4)
        state_dim = config['state_dim']
        x0_norm = jnp.zeros(state_dim)

        # Use hyperparameters from config for all networks
        self.state_net = StateNN(
            key=state_key,
            out_size=config['q_dim'] + state_dim,
            width=config['state_width'],
            depth=config['state_depth'],
            mapping_size=config['mapping_size'],
            scale=config['scale']
        )
        self.hamiltonian_net = HamiltonianNN(
            h_key, in_size=state_dim, width=config['h_width'], depth=config['h_depth'],
            x0=x0_norm, epsilon=config['h_epsilon']
        )
        self.dissipation_net = DissipationNN(
            d_key, state_dim=state_dim, width=config['d_width'],
            depth=config['d_depth'], activation=config['activation']
        )
        self.j_net = DynamicJ_NN(
            j_key, state_dim=state_dim, width=config['j_width'],
            depth=config['j_depth'], activation=config['activation']
        )

# ==============================================================================
# 2. OBJECTIVE FUNCTION FOR OPTUNA
# ==============================================================================

def objective(trial, epochs_per_trial, static_data):
    """
    The main objective function that Optuna will minimize.
    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameters.
        epochs_per_trial (int): The number of epochs to train for during each trial.
        static_data (dict): A dictionary containing all pre-processed data and stats.
    Returns:
        float: The best training Hamiltonian loss achieved during the trial.
    """
    # --- 1. Suggest Hyperparameters from the Search Space ---
    key = jax.random.PRNGKey(42)  # Use a fixed key for reproducibility across trials
    model_key, _ = jax.random.split(key)

    # StateNN Fourier Features
    mapping_size = trial.suggest_int("mapping_size", 64, 512, step=2)  # Enforce even numbers
    scale = trial.suggest_float("scale", 10, 1000, log=True)

    # Network Architectures
    state_width = trial.suggest_int("state_width", 64, 1024)
    state_depth = trial.suggest_int("state_depth", 2, 6)
    h_width = trial.suggest_int("h_width", 32, 512)
    h_depth = trial.suggest_int("h_depth", 2, 6)
    d_width = trial.suggest_int("d_width", 4, 64)
    d_depth = trial.suggest_int("d_depth", 2, 6)
    j_width = trial.suggest_int("j_width", 4, 64)
    j_depth = trial.suggest_int("j_depth", 2, 6)
    h_epsilon = trial.suggest_float("h_epsilon", 0.01, 5)

    # Optimizer
    lr_initial = trial.suggest_float("lr_initial", 1e-4, 1e-2, log=True)
    decay_steps = trial.suggest_int("decay_steps", 100, 2000)

    # Training and Loss
    batch_size = trial.suggest_int("batch_size", 100, 4000)
    lambda_conservative_max = trial.suggest_float("lambda_conservative_max", 0.1, 10, log=True)
    lambda_dissipative_max = trial.suggest_float("lambda_dissipative_max", 0.1, 10, log=True)
    lambda_physics_max = trial.suggest_float("lambda_physics_max", 0.1, 10, log=True)
    lambda_warmup_epochs = trial.suggest_int("lambda_warmup_epochs", 500, 2000)

    # --- 2. Build Model and Optimizer with Suggested Values ---
    nn_config = {
        "state_dim": static_data['s_dim'],
        "q_dim": static_data['q_dim'],
        "state_width": state_width, "state_depth": state_depth,
        "mapping_size": mapping_size, "scale": scale,
        "h_width": h_width, "h_depth": h_depth, "h_epsilon": h_epsilon,
        "d_width": d_width, "d_depth": d_depth,
        "j_width": j_width, "j_depth": j_depth,
        "activation": jax.nn.softplus,
    }
    model = Combined_sPHNN_PINN(key=model_key, config=nn_config)

    lr_schedule = optax.linear_schedule(
        init_value=lr_initial, end_value=1e-5, transition_steps=decay_steps
    )
    optimizer = optax.adamw(learning_rate=lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- 3. Run the Training Loop and Track Training Hamiltonian Loss ---
    best_training_hamiltonian_loss = jnp.inf
    num_batches = static_data['t_train_norm'].shape[0] // batch_size
    if num_batches == 0:
        num_batches = 1

    for epoch in range(epochs_per_trial):
        warmup_factor = jnp.minimum(1.0, (epoch + 1) / lambda_warmup_epochs)
        current_lambda_conservative = lambda_conservative_max * warmup_factor
        current_lambda_dissipative = lambda_dissipative_max * warmup_factor
        current_lambda_physics = lambda_physics_max * warmup_factor

        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, static_data['t_train_norm'].shape[0])
        t_shuffled, s_shuffled, q_shuffled, s_dot_shuffled, H_shuffled = (
            static_data['t_train_norm'][perm], static_data['s_train_norm'][perm],
            static_data['q_train_norm'][perm], static_data['s_dot_train_norm'][perm],
            static_data['H_train_norm'][perm]
        )

        epoch_hamiltonian_loss = 0.0
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            t_b, s_b, q_b, s_dot_b, H_b = (
                t_shuffled[start:end], s_shuffled[start:end], q_shuffled[start:end],
                s_dot_shuffled[start:end], H_shuffled[start:end]
            )
            # The train_step function returns the model, optimizer state, total loss, and a dictionary of loss components
            model, opt_state, _, loss_components = train_step(
                model, opt_state, optimizer, t_b, s_b, q_b, s_dot_b, H_b,
                current_lambda_conservative, current_lambda_dissipative, current_lambda_physics,
                static_data['hr_params'], static_data['t_mean'], static_data['t_std'],
                static_data['s_mean'], static_data['s_std'], static_data['q_mean'],
                static_data['q_std'], static_data['s_dot_mean'], static_data['s_dot_std'],
                static_data['H_mean'], static_data['H_std']
            )
            # Accumulate the Hamiltonian loss from the training batch
            epoch_hamiltonian_loss += loss_components['hamiltonian']

        # Calculate the average training Hamiltonian loss for the epoch
        avg_epoch_hamiltonian_loss = epoch_hamiltonian_loss / num_batches

        # Update the best training Hamiltonian loss if the current one is better
        if avg_epoch_hamiltonian_loss < best_training_hamiltonian_loss:
            best_training_hamiltonian_loss = avg_epoch_hamiltonian_loss

    # --- 4. Return the Final Metric to Optuna ---
    return best_training_hamiltonian_loss


# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- 1. Load and Prepare Data (Done Once) ---
    print("Loading and preparing data for optimization...")
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'PINN Data', 'error_system_data.pkl')
    t, s, q, s_dot_true, H_analytical = generate_data(data_path)
    if t is None:
        sys.exit("Exiting: Data loading failed. Make sure 'error_system_data.pkl' exists.")

    validation_split = 0.2
    num_samples = s.shape[0]
    key = jax.random.PRNGKey(123)
    perm = jax.random.permutation(key, num_samples)
    t_shuffled, s_shuffled, q_shuffled, s_dot_shuffled, H_shuffled = \
        t[perm], s[perm], q[perm], s_dot_true[perm], H_analytical[perm]
    t_shuffled = t_shuffled.reshape(-1, 1)

    split_idx = int(num_samples * (1 - validation_split))
    t_train, t_val = jnp.split(t_shuffled, [split_idx])
    s_train, s_val = jnp.split(s_shuffled, [split_idx])
    q_train, q_val = jnp.split(q_shuffled, [split_idx])
    s_dot_train, s_dot_val = jnp.split(s_dot_shuffled, [split_idx])
    H_train, H_val = jnp.split(H_shuffled, [split_idx])

    # --- Normalize Data ---
    t_mean, t_std = jnp.mean(t_train), jnp.std(t_train)
    s_mean, s_std = jnp.mean(s_train, axis=0), jnp.std(s_train, axis=0)
    q_mean, q_std = jnp.mean(q_train, axis=0), jnp.std(q_train, axis=0)
    s_dot_mean, s_dot_std = jnp.mean(s_dot_train, axis=0), jnp.std(s_dot_train, axis=0)
    H_mean, H_std = jnp.mean(H_train), jnp.std(H_train)

    static_data = {
        's_dim': s_train.shape[1], 'q_dim': q_train.shape[1],
        'hr_params': DEFAULT_PARAMS.copy(),
        't_train_norm': normalize(t_train, t_mean, t_std),
        's_train_norm': normalize(s_train, s_mean, s_std),
        'q_train_norm': normalize(q_train, q_mean, q_std),
        's_dot_train_norm': normalize(s_dot_train, s_dot_mean, s_dot_std),
        'H_train_norm': normalize(H_train, H_mean, H_std),
        't_val_norm': normalize(t_val, t_mean, t_std),
        's_val_norm': normalize(s_val, s_mean, s_std),
        'q_val_norm': normalize(q_val, q_mean, q_std),
        's_dot_val_norm': normalize(s_dot_val, s_dot_mean, s_dot_std),
        'H_val_norm': normalize(H_val, H_mean, H_std),
        't_mean': t_mean, 't_std': t_std, 's_mean': s_mean, 's_std': s_std,
        'q_mean': q_mean, 'q_std': q_std, 's_dot_mean': s_dot_mean, 's_dot_std': s_dot_std,
        'H_mean': H_mean, 'H_std': H_std,
    }

    # --- 2. Create and Run the Optuna Study ---
    print("\nStarting Optuna hyperparameter search...")

    # Define the storage directory and ensure it exists
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data')
    os.makedirs(results_dir, exist_ok=True)

    # Define the full path for the database file
    db_name = os.path.basename(__file__).replace('.py', '.db')
    db_path = os.path.join(results_dir, db_name)

    storage_name = f"sqlite:///{db_path}"
    study_name = "sphnn_pinn_optimization_study"

    objective_with_args = lambda trial: objective(trial, epochs_per_trial=500, static_data=static_data)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True
        )

    study.optimize(objective_with_args, n_trials=200)

    # --- 3. Print and Save the Results ---
    print("\nOptimization finished.")
    print(f"Study results are saved in: {storage_name}")
    print("Number of finished trials: ", len(study.trials))

    best_trial = study.best_trial

    print("Best trial:")
    print(f"  Value (Best Validation Loss): {best_trial.value:.6f}")
    print("  Best Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the best hyperparameters to a text file
    output_txt_path = os.path.join(results_dir, "best_hyperparams.txt")
    with open(output_txt_path, 'w') as f:
        f.write("Best Hyperparameter Optimization Results\n")
        f.write("========================================\n\n")
        f.write(f"Best Value (Validation Loss): {best_trial.value}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"    {key}: {value}\n")

    print(f"\n✅ Best hyperparameters saved to: {output_txt_path}")