"""
This script implements a Port-Hamiltonian Physics-Informed Neural Network.
The model learns the underlying physical structure of an error-feedback system
by training a composite model that simultaneously predicts the system's state
and enforces the port-Hamiltonian structure as a physics constraint.

The model is composed of two main parts:
1.  State Prediction Network: An MLP with Fourier features (StateNN)
    learns the combined state trajectories x(t) and e(t).
2.  Port-Hamiltonian Networks: Three networks that define the error system's
    dynamics based on the predicted state e_pred:
      e_dot = (J(e,x1,u1,u2,phi1) - R(e,x1,u1,u2,phi1)) * grad_e(H(e,x1,u1,u2,phi1))
    - HamiltonianNN (H): A convex network learning the system's energy; takes e AND (x1,u1,u2,phi1)
      but autodiff is done only w.r.t. e (the four extra inputs are constants in grad).
    - DynamicJ_NN (J): An MLP learning the conservative dynamics; depends on e and (x1,u1,u2,phi1).
    - DissipationNN (R): An MLP learning the dissipative dynamics; depends on e and (x1,u1,u2,phi1).

The training loss is a combination of:
- Data Fidelity Loss: MSE between predicted states (e_pred, x_pred) and true data.
- Physics Residual Loss: Enforces that the time derivatives of the full state
  (x, e) from the StateNN (via autodiff) match the analytical vector fields.
- Conservative Loss: Enforces that the learned Hamiltonian is invariant
  under the conservative flow (Lie derivative is zero).
- Dissipative Loss: Enforces that the time derivative of the Hamiltonian
  is correctly described by the dissipative flow.
- Physics Structure Loss: Enforces that the output of the learned PHPINN
  structure matches the analytical vector field for the error dynamics.
"""

import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import sys
from src.hr_model.model import DEFAULT_PARAMS
import os
from pathlib import Path
import pickle
import optuna
import copy


# JAX configuration to use 64-bit precision.
jax.config.update("jax_enable_x64", True)


# ==============================================================================
# 1. NEURAL NETWORK DEFINITIONS
# ==============================================================================

class FourierFeatures(eqx.Module):
    """Encodes a 1D input into a higher-dimensional space using Fourier features."""
    b_matrix: jax.Array
    output_size: int = eqx.field(static=True)

    def __init__(self, key, config: dict):
        in_size = config['in_size']
        mapping_size = config['mapping_size']
        scale = config['scale']
        n_pairs = mapping_size // 2
        self.b_matrix = jax.random.normal(key, (n_pairs, in_size)) * scale
        self.output_size = n_pairs * 2

    def __call__(self, t):
        if t.ndim == 1:
            t = t[None, :]
        t_proj = t @ self.b_matrix.T
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1).squeeze()


class StateNN(eqx.Module):
    """An MLP with Fourier Features to approximate the combined state [x(t), e(t)]."""
    layers: list

    def __init__(self, key, config: dict):
        fourier_config = config['fourier_features']
        out_size = config['out_size']
        hidden_sizes = config['hidden_sizes']

        # keys: 1 for Fourier + (#hidden + 1) for linears
        keys = jax.random.split(key, len(hidden_sizes) + 2)
        fourier_key, layer_keys = keys[0], keys[1:]

        fourier_layer = FourierFeatures(fourier_key, config=fourier_config)

        layers = [fourier_layer]
        in_dim = fourier_layer.output_size

        # hidden layers
        for i, h in enumerate(hidden_sizes):
            layers.append(eqx.nn.Linear(in_dim, h, key=layer_keys[i]))
            in_dim = h

        # output head
        layers.append(eqx.nn.Linear(in_dim, out_size, key=layer_keys[-1]))
        self.layers = layers

    def __call__(self, t):
        x_out = self.layers[0](t)
        for layer in self.layers[1:-1]:
            x_out = jax.nn.tanh(layer(x_out))
        return self.layers[-1](x_out)


# --- PHPINN Component Networks (from PHPINN implementation) ---

class _FICNN(eqx.Module):
    """Fully Input Convex Neural Network with variable hidden sizes."""
    w_layers: list    # list of Linear(in_size, h_i)
    u_layers: list    # list of Linear(h_{i-1}, h_i) with nonnegative weights
    final_layer: eqx.nn.Linear
    activation: callable = eqx.field(static=True)

    def __init__(self, key, in_size: int, out_size: int, hidden_sizes: list):
        assert len(hidden_sizes) >= 1, "FICNN needs at least one hidden layer."
        self.activation = jax.nn.softplus

        L = len(hidden_sizes)
        # keys: L for W, (L-1) for U, 1 for final
        keys = jax.random.split(key, 2 * L)
        w_keys = keys[:L]
        u_keys = keys[L:2 * L - 1]
        final_key = keys[-1]

        # Input skip connections into each hidden layer
        self.w_layers = [eqx.nn.Linear(in_size, hidden_sizes[i], key=w_keys[i]) for i in range(L)]

        # Layer-to-layer connections (constrained to be nonnegative)
        self.u_layers = []
        for i in range(1, L):
            self.u_layers.append(eqx.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], use_bias=False, key=u_keys[i - 1]))

        # Final convex head (no bias)
        self.final_layer = eqx.nn.Linear(hidden_sizes[-1], out_size, use_bias=False, key=final_key)

    def __call__(self, e_and_ctx):
        z = self.activation(self.w_layers[0](e_and_ctx))
        for i, u in enumerate(self.u_layers, start=1):
            u_nonneg = eqx.tree_at(lambda l: l.weight, u, jnp.abs(u.weight))
            z = self.activation(u_nonneg(z) + self.w_layers[i](e_and_ctx))
        return self.final_layer(z)[0]



class HamiltonianNN(eqx.Module):
    """
    Learns a convex Hamiltonian function H([e, ctx]) with a guaranteed minimum at e0 (ctx acts as a parameter).
    ctx = [x1, u1, u2, phi1] (all normalized).
    """
    ficnn: _FICNN
    x0: jax.Array
    epsilon: float = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)  # e_dim + 4

    def __init__(self, key, in_size_with_ctx, e_dim, hidden_sizes, x0, epsilon):
        self.ficnn = _FICNN(key, in_size_with_ctx, out_size=1, hidden_sizes=hidden_sizes)
        self.x0 = x0  # shape (e_dim,)
        self.epsilon = epsilon
        self.input_dim = in_size_with_ctx

    def __call__(self, e_in, ctx_in):
        # Concatenate e and context for the FICNN input
        x_in = jnp.concatenate([e_in, ctx_in])  # shape (e_dim+4,)
        x0_ext = jnp.concatenate([self.x0, ctx_in])  # anchor uses the same ctx so H(e=0,ctx)=0

        f_x = self.ficnn(x_in)
        f_x0 = self.ficnn(x0_ext)
        grad_f_x0 = jax.grad(self.ficnn)(x0_ext)  # grad wrt the full input

        # Normalization term to set H(e0,ctx)=0 and grad_e H(e0,ctx)=0 (ctx part cancels since ctx_in - ctx_in = 0)
        f_norm = f_x0 + jnp.dot(grad_f_x0, x_in - x0_ext)

        # Regularization term to ensure a strict minimum in e-space
        f_reg = self.epsilon * jnp.sum((e_in - self.x0) ** 2)
        return f_x - f_norm + f_reg

class DissipationNN(eqx.Module):
    """
    Sparse (non-symmetric) dissipation matrix R(e, ctx) with only:
      diagonals:  R11, R22, R33, R44, R55
      off-diags:  R43, R45, R53, R54   (NOTE: non-symmetric; R34,R35 are forced zero)

    Output head predicts 9 scalars in this order:
      [R11, R22, R33, R44, R55, R43, R45, R53, R54]
    """
    layers: list
    activation: callable
    state_dim: int = eqx.field(static=True)
    input_dim: int  # e_dim + 4

    def __init__(self, key, state_dim, input_dim, hidden_sizes, activation):
        assert state_dim == 5, "Sparse DissipationNN assumes state_dim == 5."
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.activation = activation

        if len(hidden_sizes) == 0:
            # no hidden layers
            self.layers = [eqx.nn.Linear(self.input_dim, 9, key=key)]
        else:
            keys = jax.random.split(key, len(hidden_sizes) + 1)
            self.layers = [eqx.nn.Linear(self.input_dim, hidden_sizes[0], key=keys[0])]
            for i in range(1, len(hidden_sizes)):
                self.layers.append(eqx.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], key=keys[i]))
            self.layers.append(eqx.nn.Linear(hidden_sizes[-1], 9, key=keys[-1]))

    def __call__(self, e, ctx):
        z = jnp.concatenate([e, ctx])

        # hidden layers
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))

        # head: 9 parameters
        params = self.layers[-1](z)  # shape (9,)

        # unpack
        diag_vals = params[:5]            # R11,R22,R33,R44,R55
        off_vals  = params[5:]            # R43,R45,R53,R54 (in this exact order)

        # build R with two scatter writes
        R = jnp.zeros((5, 5), dtype=params.dtype)

        # set diagonal
        idx = jnp.arange(5)
        R = R.at[idx, idx].set(diag_vals)

        # set allowed off-diagonals: (3,2),(3,4),(4,2),(4,3)
        rows = jnp.array([3, 3, 4, 4])
        cols = jnp.array([2, 4, 2, 3])
        R = R.at[rows, cols].set(off_vals)

        return R


class DynamicJ_NN(eqx.Module):
    """
    Sparse/skew-symmetric J with only the first row learned:
      J[0, 1:] = [j12, j13, j14, j15] (predicted)
      J[i, 0]  = -J[0, i]             (skew)
      diag(J)  = 0
      all other entries = 0
    """
    layers: list
    state_dim: int = eqx.field(static=True)
    input_dim: int  # e_dim + 4
    activation: callable

    def __init__(self, key, state_dim, input_dim, hidden_sizes, activation):
        assert state_dim == 5, "DynamicJ_NN (sparse) assumes state_dim == 5."
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.activation = activation

        out_dim = 4  # j12, j13, j14, j15
        if len(hidden_sizes) == 0:
            self.layers = [eqx.nn.Linear(self.input_dim, out_dim, key=key)]
        else:
            keys = jax.random.split(key, len(hidden_sizes) + 1)
            self.layers = [eqx.nn.Linear(self.input_dim, hidden_sizes[0], key=keys[0])]
            for i in range(1, len(hidden_sizes)):
                self.layers.append(eqx.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], key=keys[i]))
            self.layers.append(eqx.nn.Linear(hidden_sizes[-1], out_dim, key=keys[-1]))

    def __call__(self, e, ctx):
        # Concatenate normalized inputs, same convention as before
        z = jnp.concatenate([e, ctx])

        # Forward pass
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        row_tail = self.layers[-1](z)  # shape (4,) -> [j12, j13, j14, j15]

        # Build J with two scatter writes (fast, JAX-friendly)
        J = jnp.zeros((5, 5), dtype=row_tail.dtype)
        J = J.at[0, 1:].set(row_tail)   # first row, columns 2..5
        J = J.at[1:, 0].set(-row_tail)  # first column, rows 2..5 (skew)

        # Diagonal already zero; everything else zero by construction
        return J


# --- The Combined Model ---

class Combined_PH_PINN(eqx.Module):
    """Main model combining a unified state predictor and PHPINN structure."""
    state_net: StateNN
    hamiltonian_net: HamiltonianNN
    dissipation_net: DissipationNN
    j_net: DynamicJ_NN

    def __init__(self, key, config: dict, state_dim: int):
        state_key, h_key, d_key, j_key = jax.random.split(key, 4)

        # Extract sub-configs
        state_net_config = config['state_net']
        h_net_config = config['hamiltonian_net']
        d_net_config = config['dissipation_net']
        j_net_config = config['j_net']
        activation_fn = config['activation']

        # The equilibrium point for the normalized error system is the origin.
        x0_norm = jnp.zeros(state_dim)

        # Input dimensions (normalized inputs): e_dim + 4 context features (x1,u1,u2,phi1)
        input_dim_with_ctx = state_dim + 4

        self.state_net = StateNN(key=state_key, config=state_net_config)
        self.hamiltonian_net = HamiltonianNN(
            h_key,
            in_size_with_ctx=input_dim_with_ctx,
            e_dim=state_dim,
            hidden_sizes=h_net_config['hidden_sizes'],
            x0=x0_norm,
            epsilon=h_net_config['epsilon'],
        )
        self.dissipation_net = DissipationNN(
            d_key,
            state_dim=state_dim,
            input_dim=input_dim_with_ctx,
            hidden_sizes=d_net_config['hidden_sizes'],
            activation=activation_fn,
        )
        self.j_net = DynamicJ_NN(
            j_key,
            state_dim=state_dim,
            input_dim=input_dim_with_ctx,
            hidden_sizes=j_net_config['hidden_sizes'],
            activation=activation_fn,
        )


# ==============================================================================
# 2. DATA HANDLING
# ==============================================================================

def generate_data(file_path: str):
    """
    Loads and prepares training data from a pre-generated pickle file containing
    multiple simulation runs.
    """
    print(f"Loading simulation data from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            # The file contains a list of result dictionaries
            all_runs_results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        print("Please run 'generate_data_for_PINN.py' to create the data file.")
        return None, None, None, None, None, None

    # Initialize lists to hold data from all runs
    # TEST MODIFICATION: Added all_x_dot list
    all_t, all_e, all_x, all_e_dot, all_x_dot, all_H, all_dHdt  = [], [], [], [], [], [], []

    # Process each simulation run
    for i, results in enumerate(all_runs_results):
        print(f"  ... processing run {i + 1}/{len(all_runs_results)}")

        # Extract data for the current run
        t = jnp.asarray(results['t'])
        e = jnp.vstack([
            results['e_x'], results['e_y'], results['e_z'],
            results['e_u'], results['e_phi']
        ]).T
        x = jnp.vstack([
            results['x1'], results['y1'], results['z1'], results['u1'], results['phi1'],
            results['x2'], results['y2'], results['z2'], results['u2'], results['phi2']
        ]).T
        e_dot_true = jnp.vstack([
            results['d_e_x'], results['d_e_y'], results['d_e_z'],
            results['d_e_u'], results['d_e_phi']
        ]).T
        H_analytical = jnp.asarray(results['Hamiltonian'])
        dHdt = jnp.asarray(results['dHdt'])

        # Stack the true derivatives for the x states
        x_dot_true = jnp.vstack([
            results['d_x1'], results['d_y1'], results['d_z1'], results['d_u1'], results['d_phi1'],
            results['d_x2'], results['d_y2'], results['d_z2'], results['d_u2'], results['d_phi2']
        ]).T

        # Append to the main lists
        all_t.append(t)
        all_e.append(e)
        all_x.append(x)
        all_e_dot.append(e_dot_true)
        all_H.append(H_analytical)
        all_dHdt.append(dHdt)
        all_x_dot.append(x_dot_true)

    # Concatenate all runs into single arrays
    final_t = jnp.concatenate(all_t)
    final_e = jnp.concatenate(all_e)
    final_x = jnp.concatenate(all_x)
    final_e_dot = jnp.concatenate(all_e_dot)
    final_H = jnp.concatenate(all_H)
    final_dHdt = jnp.concatenate(all_dHdt)
    final_x_dot = jnp.concatenate(all_x_dot)

    print("Data loading and aggregation complete.")
    return final_t, final_e, final_x, final_e_dot, final_x_dot, final_H, final_dHdt


def normalize(data, mean, std):
    """Normalizes data using pre-computed statistics."""
    return (data - mean) / (std + 1e-8)


def denormalize(data, mean, std):
    """Denormalizes data using pre-computed statistics."""
    return data * std + mean


def load_best_config_from_study(default_config, objective='h_loss'):
    """
    Loads the best hyperparameters from a completed Optuna study and updates
    the default training configuration.

    Args:
        default_config (dict): The default TRAIN_CONFIG dictionary.
        objective (str): The objective of the study to load ('val_loss' or 'h_loss').

    Returns:
        dict: The updated configuration dictionary with the best hyperparameters.
    """
    # Define the path to the Optuna database
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data')  #
    db_path = os.path.join(results_dir, "optimize_hyperparams.db")  #
    storage_name = f"sqlite:///{db_path}"  #
    study_name = f"sphnn_pinn_optimization_{objective}"  #

    print("=" * 60)
    print(f"Attempting to load best hyperparameters for study '{study_name}'...")
    print(f"Database: {db_path}")
    print("=" * 60)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)  #
        best_params = study.best_trial.params
        print("✅ Successfully loaded best hyperparameters from study.")
    except Exception as e:
        print(f"❌ WARNING: Could not load Optuna study. Reason: {e}")
        print("         Falling back to the default TRAIN_CONFIG.")
        return default_config

    # Create a deep copy to avoid modifying the original default config
    new_config = copy.deepcopy(default_config)

    # --- Update TRAIN_CONFIG with loaded hyperparameters ---

    # Update top-level keys that map directly
    direct_mapping_keys = [
        'batch_size', 'initial_learning_rate', 'decay_steps', 'lambda_warmup_epochs',
        'lambda_conservative_max', 'lambda_dissipative_max', 'lambda_physics_max',
        'lambda_j_structure_max', 'lambda_r_structure_max', 'lambda_phys_res_max'
    ]
    for key in direct_mapping_keys:
        if key in best_params:
            new_config[key] = best_params[key]

    # Update nested network architecture keys explicitly
    if 'mapping_size' in best_params:
        new_config['nn']['state_net']['fourier_features']['mapping_size'] = best_params['mapping_size']  #
    if 'scale' in best_params:
        new_config['nn']['state_net']['fourier_features']['scale'] = best_params['scale']  #
    if 'epsilon' in best_params:
        new_config['nn']['hamiltonian_net']['epsilon'] = best_params['epsilon']  #

    # Handle network hidden layer structures (built from width and depth)
    if 'state_width' in best_params and 'state_depth' in best_params:
        new_config['nn']['state_net']['hidden_sizes'] = [best_params['state_width']] * best_params['state_depth']  #
    if 'h_width' in best_params and 'h_depth' in best_params:
        new_config['nn']['hamiltonian_net']['hidden_sizes'] = [best_params['h_width']] * best_params['h_depth']  #
    if 'd_width' in best_params and 'd_depth' in best_params:
        new_config['nn']['dissipation_net']['hidden_sizes'] = [best_params['d_width']] * best_params['d_depth']  #
    if 'j_width' in best_params and 'j_depth' in best_params:
        new_config['nn']['j_net']['hidden_sizes'] = [best_params['j_width']] * best_params['j_depth']  #

    print("✅ TRAIN_CONFIG has been updated with optimized hyperparameters.")
    return new_config

# ==============================================================================
# 3. TRAINING LOGIC
# ==============================================================================

# --- Helper functions for the physics-based loss terms ---

def _alpha(u1, u2, m):
    """Helper function for the dissipative field f_d."""
    conds = [
        jnp.logical_and(u1 >= 1, jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(u1 >= 1, u2 <= -1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 >= 1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 <= -1),
        jnp.logical_and(u1 <= -1, u2 >= 1),
        jnp.logical_and(u1 <= -1, jnp.logical_and(u2 > -1, u2 < 1)),
    ]
    choices = [2 * m - 1., -1., -1., 2 * m - 1., -1., -1., 2 * m - 1.]
    return jnp.select(conds, choices, default=-1.)


def _beta(u1, u2, m):
    """Helper function for the dissipative field f_d."""
    conds = [
        jnp.logical_and(u1 >= 1, jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(u1 >= 1, u2 <= -1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 >= 1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 <= -1),
        jnp.logical_and(u1 <= -1, u2 >= 1),
        jnp.logical_and(u1 <= -1, jnp.logical_and(u2 > -1, u2 < 1)),
    ]
    choices = [
        2 * m * (u1 - 1), -4 * m, -2 * m * (u1 - 1), 0.,
        -2 * m * (u1 + 1), 4 * m, 2 * m * (u1 + 1),
    ]
    return jnp.select(conds, choices, default=0.)


def f_c_fn(e, x, hr_params):
    """Calculates the conservative vector field f_c(e)."""
    e_x, e_y, e_u, e_phi = e[0], e[1], e[3], e[4]
    x1, u1 = x[0], x[3]

    k, f, rho, d, r, s = \
        hr_params['k'], hr_params['f'], hr_params['rho'], hr_params['d'], hr_params['r'], hr_params['s']

    return jnp.array([
        e_y + 2 * k * f * u1 * x1 * e_u + rho * x1 * e_phi,
        -2 * d * x1 * e_x,
        r * s * e_x,
        e_x,
        e_x
    ])


def f_d_fn(e, x, hr_params):
    """Calculates the dissipative vector field f_d(e)."""
    e_x, e_y, e_z, e_u, e_phi = e[0], e[1], e[2], e[3], e[4]
    x1, u1, phi1, u2 = x[0], x[3], x[4], x[8]

    a, b, k, h, f, rho, g_e, r, q_param, m = \
        hr_params['a'], hr_params['b'], hr_params['k'], hr_params['h'], \
            hr_params['f'], hr_params['rho'], hr_params['ge'], hr_params['r'], \
            hr_params['q'], hr_params['m']

    N_val = -3 * a * x1 ** 2 + 2 * b * x1 + k * h + k * f * u1 ** 2 + rho * phi1 - 2 * g_e
    alpha_val = _alpha(u1, u2, m)
    beta_val = _beta(u1, u2, m)

    return jnp.array([
        N_val * e_x,
        -e_y,
        -r * e_z,
        alpha_val * e_u + beta_val,
        -q_param * e_phi
    ])


def hr_vector_field(t, state, N, params, I_ext, xi):
    """
    Calculates the time derivatives for a network of N coupled Hindmarsh-Rose neurons.
    """
    # Reshape the flat state vector into a 2D array (N neurons x 5 variables)
    state_matrix = state.reshape((N, 5))
    x, y, z, u, phi = state_matrix.T

    # Electrical Coupling
    x_diff = x[jnp.newaxis, :] - x[:, jnp.newaxis]
    electrical_coupling = params['ge'] * jnp.sum(jnp.asarray(xi) * x_diff, axis=1)

    # --- Calculate derivatives ---
    dxdt = (y - (params['a'] * x ** 3) + (params['b'] * x ** 2)
            + (params['k'] * (params['h'] + (params['f'] * (u ** 2))) * x)
            + (params['rho'] * phi * x) + I_ext
            + electrical_coupling
            )
    dydt = params['c'] - (params['d'] * x ** 2) - y
    dzdt = params['r'] * (params['s'] * (x + params['x0']) - z)
    dudt = -u + (params['m'] * (jnp.abs(u + 1.0) - jnp.abs(u - 1.0))) + x
    dphidt = x - (params['q'] * phi)

    # Assign calculated derivative vectors to the output matrix
    d_state_dt_matrix = jnp.zeros_like(state_matrix).at[:, 0].set(dxdt).at[:, 1].set(dydt) \
        .at[:, 2].set(dzdt).at[:, 3].set(dudt) \
        .at[:, 4].set(dphidt)

    return d_state_dt_matrix.flatten()

def align_to_reference(ref: jax.Array, pred: jax.Array):
    """Flip pred by ±1 (and recenter) to best match ref. JIT-safe, no divisions."""
    ref_c  = ref  - jnp.mean(ref)
    pred_c = pred - jnp.mean(pred)
    # sign of covariance
    s = jnp.sign(jnp.vdot(ref_c, pred_c))
    s = jnp.where(s == 0.0, 1.0, s)  # benign fallback if covariance is 0
    pred_aligned = s * pred_c + jnp.mean(ref)
    return pred_aligned, s


@eqx.filter_jit
def loss_fn(model: Combined_PH_PINN, t_batch_norm, e_true_batch_norm, x_true_batch_norm, e_dot_true_batch_norm,
            x_dot_true_batch_norm,
            H_true_batch_norm,
            lambda_conservative: float, lambda_dissipative: float, lambda_physics: float,
            lambda_j_structure: float,
            lambda_r_structure: float,
            lambda_phys_res: float,
            hr_params: dict,
            I_ext: jax.Array, xi: jax.Array,
            t_mean, t_std, e_mean, e_std, x_mean, x_std, e_dot_mean, e_dot_std, x_dot_mean, x_dot_std, H_mean, H_std):
    """Calculates the composite data and new physics-based losses."""

    # --- Part 1: State Prediction and Unified Data Fidelity Loss ---
    all_states_pred_norm = jax.vmap(model.state_net)(t_batch_norm)
    all_states_true_norm = jnp.concatenate([x_true_batch_norm, e_true_batch_norm], axis=1)
    data_loss = jnp.mean((all_states_pred_norm - all_states_true_norm) ** 2)

    x_pred_batch_norm = all_states_pred_norm[:, :10]
    e_pred_batch_norm = all_states_pred_norm[:, 10:]

    e_pred = denormalize(e_pred_batch_norm, e_mean, e_std)
    x_pred = denormalize(x_pred_batch_norm, x_mean, x_std)

    # Build normalized context inputs (x1,u1,u2,phi1) from normalized x_pred
    x1n = x_pred_batch_norm[:, 0]
    u1n = x_pred_batch_norm[:, 3]
    u2n = x_pred_batch_norm[:, 8]
    phi1n = x_pred_batch_norm[:, 4]
    ctx_batch_norm = jnp.stack([x1n, u1n, u2n, phi1n], axis=1)  # shape (B,4)

    # --- Part 2: Physics Calculations ---
    # --- e-derivatives from Autodiff ---
    get_autodiff_grad_e_slice = lambda net, t: jax.jvp(lambda t_scalar: net(t_scalar)[10:], (t,), (jnp.ones_like(t),))[1]
    e_dot_autodiff_norm = jax.vmap(get_autodiff_grad_e_slice, in_axes=(None, 0))(model.state_net, t_batch_norm)
    e_dot_autodiff = e_dot_autodiff_norm * (e_std / (t_std + 1e-8))

    # --- x-derivatives from Autodiff ---
    get_autodiff_grad_x_slice = lambda net, t: jax.jvp(lambda t_scalar: net(t_scalar)[:10],
                                                       (t,), (jnp.ones_like(t),))[1]
    x_dot_autodiff_norm = jax.vmap(get_autodiff_grad_x_slice, in_axes=(None, 0))(model.state_net, t_batch_norm)
    x_dot_autodiff = x_dot_autodiff_norm * (x_std / (t_std + 1e-8))

    # --- Calculate derivatives from Analytical Equations ---
    f_c_batch = jax.vmap(f_c_fn, in_axes=(0, 0, None))(e_pred, x_pred, hr_params)
    f_d_batch = jax.vmap(f_d_fn, in_axes=(0, 0, None))(e_pred, x_pred, hr_params)
    e_dot_diss_cons = f_c_batch + f_d_batch

    # --- Analytical derivatives for x from the HR vector field ---
    vmapped_hr_vector_field = jax.vmap(hr_vector_field, in_axes=(None, 0, None, None, None, None))
    x_dot_vectorfield = vmapped_hr_vector_field(None, x_pred, 2, hr_params, I_ext, xi)

    # --- Part 3: Loss Components ---
    # Physics Residual Losses (error and HR states)
    e_dot_true = denormalize(e_dot_true_batch_norm, e_dot_mean, e_dot_std)
    x_dot_true = denormalize(x_dot_true_batch_norm, x_dot_mean, x_dot_std)
    physics_residual_loss1 = jnp.mean((e_dot_true - e_dot_diss_cons) ** 2)
    physics_residual_loss2 = jnp.mean((x_dot_true - x_dot_vectorfield) ** 2)
    physics_residual_loss = physics_residual_loss1 + physics_residual_loss2

    # PH Structure Losses
    # grad only w.r.t. e (ctx is treated as constant via closure)
    def grad_H_single(e_norm, ctx_norm):
        return jax.grad(lambda ee: model.hamiltonian_net(ee, ctx_norm))(e_norm)

    grad_H_norm = jax.vmap(grad_H_single)(e_pred_batch_norm, ctx_batch_norm)
    grad_H = grad_H_norm * (H_std / (e_std + 1e-8))

    J = jax.vmap(model.j_net)(e_pred_batch_norm, ctx_batch_norm)  # treat as physical operator
    R = jax.vmap(model.dissipation_net)(e_pred_batch_norm, ctx_batch_norm)  # treat as physical operator
    e_dot_from_structure = jax.vmap(lambda j, r, g: (j - r) @ g)(J, R, grad_H)  # physical ė
    loss_phys = jnp.mean((e_dot_true - e_dot_from_structure) ** 2)

    # --- PH structure fidelity (PHYSICAL vs PHYSICAL) ---
    j_grad_h = jax.vmap(lambda j, g: j @ g)(J, grad_H)
    r_grad_h = jax.vmap(lambda r, g: -r @ g)(R, grad_H)
    loss_j_structure = jnp.mean((f_c_batch - j_grad_h) ** 2)
    loss_r_structure = jnp.mean((f_d_batch - r_grad_h) ** 2)

    # --- Conservative loss (Lie derivative; PHYSICAL) ---
    lie_derivative = jax.vmap(jnp.dot)(grad_H, f_c_batch)
    loss_conservative = jnp.mean(lie_derivative ** 2)

    # --- Dissipative loss (PHYSICAL) ---
    dHdt_from_true = jax.vmap(jnp.dot)(grad_H, e_dot_true)
    dHdt_from_equations = jax.vmap(jnp.dot)(grad_H, f_d_batch)
    loss_dissipative = jnp.mean((dHdt_from_true - dHdt_from_equations) ** 2)

    # Hamiltonian Loss (for monitoring only)
    H_pred_norm = jax.vmap(lambda e, ctx: model.hamiltonian_net(e, ctx))(e_pred_batch_norm, ctx_batch_norm)
    H_pred = denormalize(H_pred_norm, H_mean, H_std)
    H_true = denormalize(H_true_batch_norm, H_mean, H_std)
    H_pred_aligned, _ = align_to_reference(H_true, H_pred)
    loss_hamiltonian = jnp.mean((H_pred_aligned - H_true) ** 2)

    # --- Part 4: Total Loss ---
    total_loss = (data_loss
                  + (lambda_conservative * loss_conservative)
                  + (lambda_dissipative * loss_dissipative)
                  + (lambda_physics * loss_phys)
                  + (lambda_j_structure * loss_j_structure)
                  + (lambda_r_structure * loss_r_structure)
                  + (lambda_phys_res * physics_residual_loss))

    loss_components = {
        "total": total_loss,
        "data_unified": data_loss,
        "physics_residual": physics_residual_loss,
        "phys": loss_phys,
        "conservative": loss_conservative,
        "dissipative": loss_dissipative,
        "j_structure": loss_j_structure,
        "r_structure": loss_r_structure,
        "hamiltonian": loss_hamiltonian,
    }
    return total_loss, loss_components

@eqx.filter_jit
def train_step(model, opt_state, optimizer, t_batch_norm, e_batch_norm, x_batch_norm, e_dot_batch_norm,
               x_dot_batch_norm,
               H_batch_norm,
               lambda_conservative, lambda_dissipative, lambda_physics,
               lambda_j_structure,
               lambda_r_structure,
               lambda_phys_res,
               hr_params, I_ext, xi,
               t_mean, t_std, e_mean, e_std, x_mean, x_std, e_dot_mean, e_dot_std, x_dot_mean, x_dot_std, H_mean,
               H_std):
    """Performs a single training step."""
    (loss_val, loss_components), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, t_batch_norm, e_batch_norm, x_batch_norm, e_dot_batch_norm,
        x_dot_batch_norm,
        H_batch_norm,
        lambda_conservative, lambda_dissipative, lambda_physics,
        lambda_j_structure,
        lambda_r_structure,
        lambda_phys_res,
        hr_params, I_ext, xi,
        t_mean, t_std, e_mean, e_std, x_mean, x_std, e_dot_mean, e_dot_std, x_dot_mean, x_dot_std, H_mean, H_std
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val, loss_components

@eqx.filter_jit
def evaluate_model(model, t_batch_norm, e_batch_norm, x_batch_norm, e_dot_batch_norm,
                   x_dot_batch_norm,
                   H_batch_norm,
                   lambda_conservative, lambda_dissipative, lambda_physics,
                   lambda_j_structure,
                   lambda_r_structure,
                   lambda_phys_res,
                   hr_params, I_ext, xi,
                   t_mean, t_std, e_mean, e_std, x_mean, x_std, e_dot_mean, e_dot_std, x_dot_mean, x_dot_std, H_mean,
                   H_std):
    """Calculates the loss for the validation set."""
    # This function now returns both the total loss and the components dictionary
    loss_val, loss_components = loss_fn(
        model, t_batch_norm, e_batch_norm, x_batch_norm, e_dot_batch_norm,
        x_dot_batch_norm,
        H_batch_norm,
        lambda_conservative, lambda_dissipative, lambda_physics,
        lambda_j_structure,
        lambda_r_structure,
        lambda_phys_res,
        hr_params, I_ext, xi,
        t_mean, t_std, e_mean, e_std, x_mean, x_std, e_dot_mean, e_dot_std, x_dot_mean, x_dot_std, H_mean, H_std
    )
    return loss_val, loss_components

# ==============================================================================
# 4. MAIN EXECUTION LOGIC
# ==============================================================================
def main():
    """Main function to run the training and evaluation."""

    # ==========================================================================
    # --- Centralized Training and Model Configuration ---
    # ==========================================================================

    TRAIN_CONFIG = {
        # --- General Setup ---
        "seed": 42,
        "data_file_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data/',
                                       'error_system_data.pkl'),
        "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'temp/'),

        # --- Training Hyperparameters ---
        "batch_size": 32000,
        "validation_split": 0.2,
        "epochs": 1000,

        # --- Optimizer and Learning Rate Schedule ---
        "initial_learning_rate": 1e-3,
        "end_learning_rate": 1e-5,
        "decay_steps": 2000,

        # --- Physics Loss Hyperparameters ---
        "lambda_conservative_max": 1,
        "lambda_dissipative_max": 1,
        "lambda_physics_max": 1.5,
        "lambda_j_structure_max": 1,
        "lambda_r_structure_max": 1,
        "lambda_phys_res_max": 0.01,
        "lambda_warmup_epochs": 5000,

        # --- System Parameters ---
        "hr_params": {
            **DEFAULT_PARAMS,
            'ge': 0.62,
        },
        "I_ext": jnp.array([0.8, 0.8]),
        "xi": jnp.array([[0, 1], [1, 0]]),

        # --- Visualization ---
        "run_to_visualize_idx": 0,
        "vis_start_ratio": 0,
        "vis_end_ratio": 0.5,

        # --- Neural Network Architectures ---
        "nn": {
            "state_net": {
                "out_size": 15,  # Will be updated dynamically based on data shape (x_dim + e_dim)
                "hidden_sizes": [1024, 1024],
                "fourier_features": {
                    "in_size": 1,
                    "mapping_size": 32,
                    "scale": 300
                }
            },
            "hamiltonian_net": {
                "hidden_sizes": [256, 256],
                "epsilon": 0.525
            },
            "dissipation_net": {
                "hidden_sizes": [16, 9],
            },
            "j_net": {
                "hidden_sizes": [4],
            },
            "activation": jax.nn.tanh,
        }
    }

    # --- Overwrite config with best params from Optuna study if enabled ---
    read_config_from_study = True
    if read_config_from_study:
        # Load the study that optimized for Hamiltonian loss ('h_loss')
        TRAIN_CONFIG = load_best_config_from_study(TRAIN_CONFIG, objective='h_loss')
    TRAIN_CONFIG["epochs"] = 500
    print(TRAIN_CONFIG)
    # ==========================================================================

    # --- Setup ---
    key = jax.random.PRNGKey(TRAIN_CONFIG["seed"])
    model_key, data_key = jax.random.split(key)
    hr_params = TRAIN_CONFIG["hr_params"]
    I_ext = TRAIN_CONFIG["I_ext"]
    xi = TRAIN_CONFIG["xi"]

    # --- Generate and Prepare Data ---
    t, e, x, e_dot_true, x_dot_true, H_analytical, dHdt_analytical = generate_data(TRAIN_CONFIG["data_file_path"])
    if t is None:
        sys.exit("Exiting: Data loading failed.")

    num_samples = e.shape[0]
    perm = jax.random.permutation(data_key, num_samples)
    t_shuffled, e_shuffled, x_shuffled, e_dot_shuffled, x_dot_shuffled, H_shuffled = \
        t[perm], e[perm], x[perm], e_dot_true[perm], x_dot_true[perm], H_analytical[perm]
    t_shuffled = t_shuffled.reshape(-1, 1)

    split_idx = int(num_samples * (1 - TRAIN_CONFIG["validation_split"]))
    t_train, t_val = jnp.split(t_shuffled, [split_idx])
    e_train, e_val = jnp.split(e_shuffled, [split_idx])
    x_train, x_val = jnp.split(x_shuffled, [split_idx])
    e_dot_train, e_dot_val = jnp.split(e_dot_shuffled, [split_idx])
    x_dot_train, x_dot_val = jnp.split(x_dot_shuffled, [split_idx])
    H_train, H_val = jnp.split(H_shuffled, [split_idx])

    # --- Normalize Data (using ONLY training set statistics) ---
    t_mean, t_std = jnp.mean(t_train), jnp.std(t_train)
    e_mean, e_std = jnp.mean(e_train, axis=0), jnp.std(e_train, axis=0)
    x_mean, x_std = jnp.mean(x_train, axis=0), jnp.std(x_train, axis=0)
    e_dot_mean, e_dot_std = jnp.mean(e_dot_train, axis=0), jnp.std(e_dot_train, axis=0)
    # TEST MODIFICATION: Calculate norm stats for x_dot
    x_dot_mean, x_dot_std = jnp.mean(x_dot_train, axis=0), jnp.std(x_dot_train, axis=0)
    H_mean, H_std = jnp.mean(H_train), jnp.std(H_train)

    t_train_norm = normalize(t_train, t_mean, t_std)
    e_train_norm = normalize(e_train, e_mean, e_std)
    x_train_norm = normalize(x_train, x_mean, x_std)
    e_dot_train_norm = normalize(e_dot_train, e_dot_mean, e_dot_std)
    # TEST MODIFICATION: Normalize x_dot_train
    x_dot_train_norm = normalize(x_dot_train, x_dot_mean, x_dot_std)
    H_train_norm = normalize(H_train, H_mean, H_std)

    t_val_norm = normalize(t_val, t_mean, t_std)
    e_val_norm = normalize(e_val, e_mean, e_std)
    x_val_norm = normalize(x_val, x_mean, x_std)
    e_dot_val_norm = normalize(e_dot_val, e_dot_mean, e_dot_std)
    # TEST MODIFICATION: Normalize x_dot_val
    x_dot_val_norm = normalize(x_dot_val, x_dot_mean, x_dot_std)
    H_val_norm = normalize(H_val, H_mean, H_std)

    # --- Initialize Model ---
    e_dim = e_train.shape[1]
    x_dim = x_train.shape[1]

    # Update network config with data-dependent shapes
    nn_config = TRAIN_CONFIG["nn"]
    nn_config["state_net"]["out_size"] = e_dim + x_dim

    model = Combined_PH_PINN(key=model_key, config=nn_config, state_dim=e_dim)

    # --- Training Loop ---
    batch_size = TRAIN_CONFIG["batch_size"]
    epochs = TRAIN_CONFIG["epochs"]
    num_batches = t_train_norm.shape[0] // batch_size
    if num_batches == 0 and t_train_norm.shape[0] > 0:
        print(f"Warning: batch_size ({batch_size}) > num samples. Setting num_batches to 1.")
        num_batches = 1

    lr_schedule = optax.linear_schedule(
        init_value=TRAIN_CONFIG["initial_learning_rate"],
        end_value=TRAIN_CONFIG["end_learning_rate"],
        transition_steps=TRAIN_CONFIG["decay_steps"]
    )
    optimizer = optax.adamw(learning_rate=lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_losses, val_losses = [], []
    phys_losses, conservative_losses, dissipative_losses, hamiltonian_losses = [], [], [], []
    best_model, best_val_loss, best_h_loss = model, jnp.inf, jnp.inf
    data_unified_losses, physics_residual_losses = [], []
    j_structure_losses, r_structure_losses = [], []

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Loss weight warmup schedule
        warmup_factor = jnp.minimum(1.0, (epoch + 1) / TRAIN_CONFIG["lambda_warmup_epochs"])
        current_lambda_conservative = TRAIN_CONFIG["lambda_conservative_max"] * warmup_factor
        current_lambda_dissipative = TRAIN_CONFIG["lambda_dissipative_max"] * warmup_factor
        current_lambda_physics = TRAIN_CONFIG["lambda_physics_max"] * warmup_factor
        current_lambda_j_structure = TRAIN_CONFIG["lambda_j_structure_max"] * warmup_factor
        current_lambda_r_structure = TRAIN_CONFIG["lambda_r_structure_max"] * warmup_factor
        current_lambda_phys_res = TRAIN_CONFIG["lambda_phys_res_max"] * warmup_factor

        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, t_train_norm.shape[0])
        # TEST MODIFICATION: Add x_dot_shuffled
        t_shuffled, e_shuffled, x_shuffled, e_dot_shuffled, x_dot_shuffled, H_shuffled = \
            t_train_norm[perm], e_train_norm[perm], x_train_norm[perm], \
                e_dot_train_norm[perm], x_dot_train_norm[perm], H_train_norm[perm]

        # Initialize epoch loss accumulators
        epoch_losses = {k: 0.0 for k in
                        ["total", "data_unified", "physics_residual", "phys", "conservative", "dissipative",
                         "j_structure", "r_structure", "hamiltonian"]}

        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            # TEST MODIFICATION: Add x_dot_b
            t_b, e_b, x_b = t_shuffled[start:end], e_shuffled[start:end], x_shuffled[start:end]
            e_dot_b, x_dot_b, H_b = e_dot_shuffled[start:end], x_dot_shuffled[start:end], H_shuffled[start:end]

            # TEST MODIFICATION: Pass new data to train_step
            model, opt_state, train_loss_val, loss_comps = train_step(
                model, opt_state, optimizer, t_b, e_b, x_b, e_dot_b, x_dot_b, H_b,
                current_lambda_conservative, current_lambda_dissipative, current_lambda_physics,
                current_lambda_j_structure,
                current_lambda_r_structure,
                current_lambda_phys_res,
                hr_params, I_ext, xi,
                t_mean, t_std, e_mean, e_std, x_mean, x_std, e_dot_mean, e_dot_std, x_dot_mean, x_dot_std, H_mean, H_std
            )
            for k in epoch_losses:
                if k in loss_comps:
                    epoch_losses[k] += loss_comps[k]

        # Calculate average losses for the epoch
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        # TEST MODIFICATION: Pass new data to evaluate_model
        val_loss, val_loss_comps = evaluate_model(
            model, t_val_norm, e_val_norm, x_val_norm, e_dot_val_norm, x_dot_val_norm, H_val_norm,
            current_lambda_conservative, current_lambda_dissipative, current_lambda_physics,
            current_lambda_j_structure,
            current_lambda_r_structure,
            current_lambda_phys_res,
            hr_params, I_ext, xi,
            t_mean, t_std, e_mean, e_std, x_mean, x_std, e_dot_mean, e_dot_std, x_dot_mean, x_dot_std, H_mean, H_std
        )

        # Append all losses for plotting
        train_losses.append(avg_losses["total"])
        val_losses.append(val_loss)
        phys_losses.append(avg_losses["phys"])
        conservative_losses.append(avg_losses["conservative"])
        dissipative_losses.append(avg_losses["dissipative"])
        hamiltonian_losses.append(avg_losses["hamiltonian"])
        data_unified_losses.append(avg_losses["data_unified"])
        physics_residual_losses.append(avg_losses["physics_residual"])
        j_structure_losses.append(avg_losses["j_structure"])
        r_structure_losses.append(avg_losses["r_structure"])

        # --- UPDATED MODEL SAVING LOGIC ---
        # Get the Hamiltonian loss from the training set results
        train_h_loss = avg_losses['hamiltonian']

        # Save the model if the training Hamiltonian loss is the best we've seen
        if train_h_loss < best_h_loss:
            best_h_loss = train_h_loss
            best_val_loss = val_loss  # Keep saving the corresponding total val loss for reference
            best_model = model

        # if val_loss < best_val_loss:
        #     best_h_loss = train_h_loss
        #     best_val_loss = val_loss  # Keep saving the corresponding total val loss for reference
        #     best_model = model

        if (epoch + 1) % 1 == 0 or epoch == 0:
            log_str = (
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_losses['total']:.4f} | Val Loss: {val_loss:.4f} | "
                f"H_Loss: {avg_losses['hamiltonian']:.4f} | "
                f"Data: {avg_losses['data_unified']:.4f} | PhysRes: {avg_losses['physics_residual']:.4f} | "
                f"PH: {avg_losses['phys']:.4f} | J_Struct: {avg_losses['j_structure']:.4f} | "
                f"R_Struct: {avg_losses['r_structure']:.4f} | "
                f"Cons: {avg_losses['conservative']:.4f} | Diss: {avg_losses['dissipative']:.4f}"
            )
            print(log_str)

    print("Training finished.")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")
    print(f"Best hamiltonian loss achieved: {best_h_loss:.6f}")

    # ==============================================================================
    # 5. VISUALIZATION AND ANALYSIS
    # ==============================================================================
    output_dir = TRAIN_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    run_to_visualize_idx = TRAIN_CONFIG["run_to_visualize_idx"]
    print(f"\nGenerating visualization plots for simulation run #{run_to_visualize_idx + 1}...")

    # Load the data again to isolate a single run for clean plotting
    with open(TRAIN_CONFIG["data_file_path"], 'rb') as f:
        all_runs = pickle.load(f)

    # Ensure the chosen index is valid
    if run_to_visualize_idx >= len(all_runs):
        print(
            f"Error: 'run_to_visualize_idx' ({run_to_visualize_idx}) is out of bounds. Max is {len(all_runs) - 1}. Setting to 0.")
        run_to_visualize_idx = 0

    vis_results = all_runs[run_to_visualize_idx]

    # Use the selected run's data for all subsequent plotting
    t_test = jnp.asarray(vis_results['t']).reshape(-1, 1)
    e_test = jnp.vstack([
        vis_results['e_x'], vis_results['e_y'], vis_results['e_z'],
        vis_results['e_u'], vis_results['e_phi']
    ]).T
    x_test = jnp.vstack([
        vis_results['x1'], vis_results['y1'], vis_results['z1'], vis_results['u1'], vis_results['phi1'],
        vis_results['x2'], vis_results['y2'], vis_results['z2'], vis_results['u2'], vis_results['phi2']
    ]).T
    e_dot_test = jnp.vstack([
        vis_results['d_e_x'], vis_results['d_e_y'], vis_results['d_e_z'],
        vis_results['d_e_u'], vis_results['d_e_phi']
    ]).T
    H_analytical_vis = jnp.asarray(vis_results['Hamiltonian'])

    # Get x_dot_true from the data
    x_dot_test = jnp.vstack([
        vis_results['d_x1'], vis_results['d_y1'], vis_results['d_z1'], vis_results['d_u1'], vis_results['d_phi1'],
        vis_results['d_x2'], vis_results['d_y2'], vis_results['d_z2'], vis_results['d_u2'], vis_results['d_phi2']
    ]).T

    # Normalize the visualization data using the previously computed training statistics
    t_test_norm = normalize(t_test, t_mean, t_std)

    # --- Get all model predictions for the full dataset ---
    all_states_pred_norm = jax.vmap(best_model.state_net)(t_test_norm)
    x_pred_norm = all_states_pred_norm[:, :10]
    e_pred_norm = all_states_pred_norm[:, 10:]

    e_pred = denormalize(e_pred_norm, e_mean, e_std)
    x_pred = denormalize(x_pred_norm, x_mean, x_std)

    # Build normalized context for viz
    x1n_v = x_pred_norm[:, 0]
    u1n_v = x_pred_norm[:, 3]
    u2n_v = x_pred_norm[:, 8]
    phi1n_v = x_pred_norm[:, 4]
    ctx_norm_v = jnp.stack([x1n_v, u1n_v, u2n_v, phi1n_v], axis=1)

    # --- Calculate all derivatives for comparison ---
    # Autodiff derivatives
    get_e_slice_autodiff_grad = lambda net, t: jax.jvp(lambda t_scalar: net(t_scalar)[10:], (t,), (jnp.ones_like(t),))[1]
    e_dot_autodiff_norm = jax.vmap(get_e_slice_autodiff_grad, in_axes=(None, 0))(best_model.state_net, t_test_norm)
    e_dot_autodiff = e_dot_autodiff_norm * (e_std / (t_std + 1e-8))

    get_x_slice_autodiff_grad = lambda net, t: jax.jvp(lambda t_scalar: net(t_scalar)[:10], (t,), (jnp.ones_like(t),))[1]
    x_dot_autodiff_norm = jax.vmap(get_x_slice_autodiff_grad, in_axes=(None, 0))(best_model.state_net, t_test_norm)
    x_dot_autodiff = x_dot_autodiff_norm * (x_std / (t_std + 1e-8))

    # Analytical derivatives from predicted states
    vmapped_hr_vector_field = jax.vmap(hr_vector_field, in_axes=(None, 0, None, None, None, None))
    x_dot_from_vectorfield_vis = vmapped_hr_vector_field(None, x_pred, 2, hr_params, I_ext, xi)
    f_c_batch_vis = jax.vmap(f_c_fn, in_axes=(0, 0, None))(e_pred, x_pred, hr_params)
    f_d_batch_vis = jax.vmap(f_d_fn, in_axes=(0, 0, None))(e_pred, x_pred, hr_params)
    e_dot_from_equations = f_c_batch_vis + f_d_batch_vis

    # PH structure derivative
    def grad_H_single_v(e_norm, ctx_norm):
        return jax.grad(lambda ee: best_model.hamiltonian_net(ee, ctx_norm))(e_norm)

    # Gradient of H in PHYSICAL units (same as training)
    grad_H_norm = jax.vmap(grad_H_single_v)(e_pred_norm, ctx_norm_v)  # dH/de_norm
    grad_H = grad_H_norm * (H_std / (e_std + 1e-8))

    # J and R as used in training (inputs are normalized [e_norm, ctx_norm])
    J = jax.vmap(best_model.j_net)(e_pred_norm, ctx_norm_v)
    R = jax.vmap(best_model.dissipation_net)(e_pred_norm, ctx_norm_v)

    # Structure flow in PHYSICAL units (no extra scaling)
    e_dot_from_structure = jax.vmap(lambda j, r, g: (j - r) @ g)(J, R, grad_H)

    # --- Plot 1: Learned vs Analytical Hamiltonian ---
    print("Comparing learned Hamiltonian with analytical solution...")
    H_learned_norm = jax.vmap(lambda e, ctx: best_model.hamiltonian_net(e, ctx))(e_pred_norm, ctx_norm_v)
    H_learned = denormalize(H_learned_norm, H_mean, H_std)
    H_learned_aligned, sign_H = align_to_reference(H_analytical_vis, H_learned)

    split_start = int(len(t_test) * TRAIN_CONFIG["vis_start_ratio"])
    split_end = int(len(t_test) * TRAIN_CONFIG["vis_end_ratio"])

    plt.figure(figsize=(12, 7))
    plt.plot(t_test[split_start:split_end], H_analytical_vis[split_start:split_end], label='Analytical Hamiltonian',
             color='blue')
    plt.plot(t_test[split_start:split_end], H_learned_aligned[split_start:split_end],
             label='Learned Hamiltonian (Aligned)', color='red')
    plt.title("Time Evolution of Hamiltonians", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Hamiltonian Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'hamiltonian_comparison.png'), dpi=300)
    plt.tight_layout()

    # --- Plot 2: Training, Validation, and Physics Losses ---
    # --- Plot 2: Training, Validation, and Physics Losses ---
    plt.figure(figsize=(14, 9))
    plt.plot(train_losses, label='Total Training Loss', color='black', linewidth=2.5)
    plt.plot(val_losses, label='Total Validation Loss', color='firebrick', linewidth=2.5)

    # Data and Physics Residuals
    plt.plot(data_unified_losses, label='Data Fidelity Loss', color='dodgerblue')
    plt.plot(physics_residual_losses, label='Physics Residual Loss', color='darkorange')

    # PH Structure Losses
    plt.plot(phys_losses, label='PH Structure Loss (phys)', color='purple', alpha=0.9)
    plt.plot(j_structure_losses, label='J Structure Loss', color='brown', alpha=0.7)
    plt.plot(r_structure_losses, label='R Structure Loss', color='magenta', alpha=0.7)

    # Core Physics Losses
    plt.plot(conservative_losses, label='Conservative Loss', color='green', alpha=0.9)
    plt.plot(dissipative_losses, label='Dissipative Loss', color='darkcyan', alpha=0.9)

    # Monitoring Loss
    plt.plot(hamiltonian_losses, label='Hamiltonian Loss (Monitor)', color='gold')

    plt.yscale('log')
    plt.title('Training, Validation, and All Physics Losses Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.legend(fontsize=10, loc='upper right', ncol=2)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'training_losses_detailed.png'), dpi=300)
    plt.tight_layout()
    # --- Plot 3: Derivative Comparison (Physics Fidelity) for Error States ---
    fig, axes = plt.subplots(e_test.shape[1], 1, figsize=(12, 12), sharex=True)
    state_labels_e_dot = [r'$\dot{e}_x$', r'$\dot{e}_y$', r'$\dot{e}_z$', r'$\dot{e}_u$', r'$\dot{e}_\phi$']
    fig.suptitle("Error Derivative Fidelity Comparison (e_dot)", fontsize=18, y=0.99)

    for i in range(e_test.shape[1]):
        axes[i].plot(t_test[split_start:split_end], e_dot_test[split_start:split_end, i], label='True Derivative',
                     color='green', linewidth=2, alpha=0.8)
        axes[i].plot(t_test[split_start:split_end], e_dot_autodiff[split_start:split_end, i], label='Autodiff',
                     color='orange')
        axes[i].plot(t_test[split_start:split_end], e_dot_from_equations[split_start:split_end, i],
                     label='Analytical Eq.', color='purple')
        axes[i].plot(t_test[split_start:split_end], e_dot_from_structure[split_start:split_end, i],
                     label='PH Structure', color='red')

        axes[i].set_ylabel(state_labels_e_dot[i], fontsize=14)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel("Time", fontsize=14)
    fig.savefig(os.path.join(output_dir, 'error_derivative_fidelity.png'), dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --- Plot 4: Derivative Comparison (Physics Fidelity) for HR States ---
    fig, axes = plt.subplots(x_test.shape[1], 1, figsize=(12, 18), sharex=True)
    state_labels_x_dot = [
        r'$\dot{x}_1$', r'$\dot{y}_1$', r'$\dot{z}_1$', r'$\dot{u}_1$', r'$\dot{\phi}_1$',
        r'$\dot{x}_2$', r'$\dot{y}_2$', r'$\dot{z}_2$', r'$\dot{u}_2$', r'$\dot{\phi}_2$'
    ]
    fig.suptitle("HR Derivative Fidelity Comparison (x_dot)", fontsize=18, y=0.99)

    for i in range(x_test.shape[1]):
        axes[i].plot(t_test[split_start:split_end], x_dot_test[split_start:split_end, i], label='True Derivative',
                     color='green', linewidth=2, alpha=0.8)
        axes[i].plot(t_test[split_start:split_end], x_dot_autodiff[split_start:split_end, i], label='Autodiff',
                     color='orange')
        axes[i].plot(t_test[split_start:split_end], x_dot_from_vectorfield_vis[split_start:split_end, i],
                     label='Analytical Eq.', color='purple')

        axes[i].set_ylabel(state_labels_x_dot[i], fontsize=14)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel("Time", fontsize=14)
    fig.savefig(os.path.join(output_dir, 'hr_derivative_fidelity.png'), dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --- Plot 5: Error System State Trajectories (e) ---
    fig, axes = plt.subplots(e_test.shape[1], 1, figsize=(12, 10), sharex=True)
    state_labels_error = [r'$e_x$', r'$e_y$', r'$e_z$', r'$e_u$', r'$e_\phi$']
    fig.suptitle("Error System State 'e' Prediction: True vs. Predicted", fontsize=18, y=0.99)
    for i in range(e_test.shape[1]):
        axes[i].plot(t_test[split_start:split_end], e_test[split_start:split_end, i], 'b', label='True State',
                     alpha=0.9)
        axes[i].plot(t_test[split_start:split_end], e_pred[split_start:split_end, i], 'r', label='Predicted State')
        axes[i].set_ylabel(state_labels_error[i], fontsize=14)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel("Time", fontsize=14)
    fig.savefig(os.path.join(output_dir, 'error_state_e_prediction.png'), dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --- Plot 6: HR System State Trajectories (x) ---
    fig, axes = plt.subplots(x_test.shape[1], 1, figsize=(12, 18), sharex=True)
    state_labels_x = [
        r'$x_1$', r'$y_1$', r'$z_1$', r'$u_1$', r'$\phi_1$',
        r'$x_2$', r'$y_2$', r'$z_2$', r'$u_2$', r'$\phi_2$'
    ]
    fig.suptitle("HR System State 'x' Prediction: True vs. Predicted", fontsize=18, y=0.99)
    for i in range(x_test.shape[1]):
        axes[i].plot(t_test[split_start:split_end], x_test[split_start:split_end, i], 'b', label='True State',
                     alpha=0.9)
        axes[i].plot(t_test[split_start:split_end], x_pred[split_start:split_end, i], 'r', label='Predicted State')
        axes[i].set_ylabel(state_labels_x[i], fontsize=14)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel("Time", fontsize=14)
    fig.savefig(os.path.join(output_dir, 'hr_state_x_prediction.png'), dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # === dH/dt comparison plot ==============================================
    # === Plot 7: dH/dt (analytical vs three predictions) ====================
    # Pull analytical dH/dt from the selected run
    dHdt_analytical_vis = jnp.asarray(vis_results['dHdt'])  # shape [T]

    # dH/dt variants (all in physical units)
    # definitions (physical units)
    dHdt_pred_true = jnp.einsum('ti,ti->t', grad_H, e_dot_test)  # ∇H·ė_true
    dHdt_pred_structure = jnp.einsum('ti,ti->t', grad_H, e_dot_from_structure)  # ∇H·ė_struct
    dHdt_pred_autodiff = jnp.einsum('ti,ti->t', grad_H, e_dot_autodiff)  # ∇H·ė_autodiff
    dHdt_pred_pH = -jnp.einsum('ti,tij,tj->t', grad_H, R, grad_H)  # -∇Hᵀ R ∇H

    # align each to analytical dH/dt
    dHdt_true_aligned, _ = align_to_reference(dHdt_analytical_vis, dHdt_pred_true)
    dHdt_struct_aligned, _ = align_to_reference(dHdt_analytical_vis, dHdt_pred_structure)
    dHdt_autodiff_aligned, _ = align_to_reference(dHdt_analytical_vis, dHdt_pred_autodiff)
    dHdt_pH_aligned, _ = align_to_reference(dHdt_analytical_vis, dHdt_pred_pH)

    plt.figure(figsize=(12, 6))
    plt.plot(t_test[split_start:split_end], np.asarray(dHdt_analytical_vis[split_start:split_end]), label=r'$\dot H$ (analytical)')
    plt.plot(t_test[split_start:split_end], np.asarray(dHdt_true_aligned[split_start:split_end]), label=r'$\nabla H \cdot \dot e_{\mathrm{true}}$')
    # plt.plot(t_test[split_start:split_end], np.asarray(dHdt_struct_aligned[split_start:split_end]), label=r'$\nabla H \cdot \dot e_{\mathrm{structure}}$')
    # plt.plot(t_test[split_start:split_end], np.asarray(dHdt_autodiff_aligned[split_start:split_end]), label=r'$\nabla H \cdot \dot e_{\mathrm{autodiff}}$')
    # plt.plot(t_test[split_start:split_end], np.asarray(dHdt_pH_aligned[split_start:split_end]), label=r'$-(\nabla H)^\top R\,\nabla H$')
    plt.xlabel('Time');
    plt.ylabel(r'$\dot H$')
    plt.title(r'$\dot H$: analytical vs. predictions')
    plt.legend(loc='best', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dHdt_comparison.png'), dpi=300)
    # ======================================================================

    # --- Plot 8 & 9: Mean J and R matrices over the whole dataset ------------
    print("Computing mean J and R matrices over the whole dataset...")

    # Use the model across the entire normalized time set (train + val)
    t_all_norm = jnp.concatenate([t_train_norm, t_val_norm], axis=0)

    # Predict [x, e] for all timestamps
    all_states_pred_norm_full = jax.vmap(best_model.state_net)(t_all_norm)
    x_pred_norm_all = all_states_pred_norm_full[:, :10]
    e_pred_norm_all = all_states_pred_norm_full[:, 10:]

    # Build ctx = [x1, u1, u2, phi1] from normalized x predictions
    x1n_all  = x_pred_norm_all[:, 0]
    u1n_all  = x_pred_norm_all[:, 3]
    u2n_all  = x_pred_norm_all[:, 8]
    phi1n_all = x_pred_norm_all[:, 4]
    ctx_norm_all = jnp.stack([x1n_all, u1n_all, u2n_all, phi1n_all], axis=1)  # [T,4]

    # Evaluate J and R for all samples
    J_stack = jax.vmap(best_model.j_net)(e_pred_norm_all, ctx_norm_all)           # [T, 5, 5]
    R_stack = jax.vmap(best_model.dissipation_net)(e_pred_norm_all, ctx_norm_all) # [T, 5, 5]

    # Mean across the dataset
    J_mean = jnp.mean(J_stack, axis=0)  # [5,5]
    R_mean = jnp.mean(R_stack, axis=0)  # [5,5]

    # Plot helpers
    var_labels = [r'$e_x$', r'$e_y$', r'$e_z$', r'$e_u$', r'$e_\phi$']

    def plot_matrix_with_numbers(mat, title, filename):
        plt.figure(figsize=(6.2, 5.4))
        im = plt.imshow(np.asarray(mat), interpolation='nearest', aspect='equal')  # default colormap
        plt.title(title, fontsize=14)
        plt.xticks(np.arange(5), var_labels, rotation=0)
        plt.yticks(np.arange(5), var_labels)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Mean value', rotation=90, va='center')

        # Add numeric annotations in each cell
        mat_np = np.asarray(mat)
        vmin, vmax = float(np.min(mat_np)), float(np.max(mat_np))
        mid = (vmin + vmax) / 2.0
        for i in range(mat_np.shape[0]):
            for j in range(mat_np.shape[1]):
                val = mat_np[i, j]
                # Choose annotation color for contrast
                txt_color = 'white' if val > mid else 'black'
                plt.text(j, i, f"{val:.3g}", ha='center', va='center', color=txt_color, fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)

    plot_matrix_with_numbers(J_mean, "Mean J matrix (dataset-wide)", "J_mean_matrix.png")
    plot_matrix_with_numbers(R_mean, "Mean R matrix (dataset-wide)", "R_mean_matrix.png")
    # -------------------------------------------------------------------------

    plt.close('all')
    print(f"All plots saved to {output_dir}")

    # ===== SAVE SPECIFIC PLOTTING DATA =====

    def _to_numpy_safe(x):
        """Convert JAX/NumPy array-likes to plain NumPy (no copy if not needed)."""
        import jax.numpy as jnp
        import numpy as np
        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
        return np.asarray(x)

    # Determine repo root and output path
    OUT_PATH = os.path.join(Path(__file__).resolve().parents[2] / "results" / "PINN Data", "pinn_plot_data.pkl")

    print(f"\nSaving specific data required for plots to {OUT_PATH}...")

    # Explicitly define the payload with only the data used in the plots
    payload = {
        # --- Time Vector & Plotting Range ---
        't_test': _to_numpy_safe(t_test),
        'split_start': split_start,
        'split_end': split_end,

        # --- Loss Histories ---
        'train_losses': _to_numpy_safe(train_losses),
        'val_losses': _to_numpy_safe(val_losses),
        'hamiltonian_losses': _to_numpy_safe(hamiltonian_losses),
        'phys_losses': _to_numpy_safe(phys_losses),
        'conservative_losses': _to_numpy_safe(conservative_losses),
        'dissipative_losses': _to_numpy_safe(dissipative_losses),
        'data_unified_losses': _to_numpy_safe(data_unified_losses),
        'physics_residual_losses': _to_numpy_safe(physics_residual_losses),
        'j_structure_losses': _to_numpy_safe(j_structure_losses),
        'r_structure_losses': _to_numpy_safe(r_structure_losses),

        # --- Hamiltonian Plot Data (Plot 1) ---
        'H_analytical_vis': _to_numpy_safe(H_analytical_vis),
        'H_learned_aligned': _to_numpy_safe(H_learned_aligned),

        # --- Derivative Fidelity Plot Data (Plots 3 & 4) ---
        'e_dot_test': _to_numpy_safe(e_dot_test),
        'e_dot_autodiff': _to_numpy_safe(e_dot_autodiff),
        'e_dot_from_equations': _to_numpy_safe(e_dot_from_equations),
        'e_dot_from_structure': _to_numpy_safe(e_dot_from_structure),
        'x_dot_test': _to_numpy_safe(x_dot_test),
        'x_dot_autodiff': _to_numpy_safe(x_dot_autodiff),
        'x_dot_from_vectorfield_vis': _to_numpy_safe(x_dot_from_vectorfield_vis),

        # --- State Trajectory Plot Data (Plots 5 & 6) ---
        'e_test': _to_numpy_safe(e_test),
        'e_pred': _to_numpy_safe(e_pred),
        'x_test': _to_numpy_safe(x_test),
        'x_pred': _to_numpy_safe(x_pred),

        # --- dH/dt Comparison Plot Data (Plot 7) ---
        'dHdt_analytical_vis': _to_numpy_safe(dHdt_analytical_vis),
        'dHdt_pred_true': _to_numpy_safe(dHdt_pred_true),
        'dHdt_pred_structure': _to_numpy_safe(dHdt_pred_structure),
        'dHdt_pred_autodiff': _to_numpy_safe(dHdt_pred_autodiff),
        'dHdt_pred_pH': _to_numpy_safe(dHdt_pred_pH),

        # --- Mean J and R Matrices (as plotted) ---
        'J_mean': _to_numpy_safe(J_mean),
        'R_mean': _to_numpy_safe(R_mean),
    }

    # Save the curated data to the pickle file
    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"✅ Saved plotting data -> {OUT_PATH}")
    # ===== END SAVE BLOCK =====


if __name__ == "__main__":
    main()

