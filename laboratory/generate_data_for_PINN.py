import jax
import jax.numpy as jnp
import diffrax as dfx
import pickle
from src.hr_model.error_system import HRNetworkErrorSystem
from src.hr_model.error_system import DEFAULT_HR_STATES0
from src.hr_model.physics import calculate_H
from src.hr_model.physics import calculate_dHdt
from src.hr_model.physics import calculate_dVdt
from src.hr_model.model import DEFAULT_PARAMS

def generate_data(
    num_runs: int,
    dynamics: str = 'complete',
    param_dict: dict = DEFAULT_PARAMS,
    initial_state_range: tuple = (-2, 2),
    seed: int = None,
    start_time: float = 0.0,
    end_time: float = 1000.0,
    dt_initial: float = 0.01,
    n_points: int = 10000,
    transient_ratio: float = 0,
    max_steps: int = None,
    solver = dfx.Tsit5(),
    stepsize_controller = dfx.PIDController(rtol=1e-10, atol=1e-12),
    I_ext = [0.8, 0.8],
    xi = [[0, 1], [1, 0]],
    output_file: str = "error_system_data.pkl"
):
    """
    Generate data by running the Error_System with multiple random initial conditions.
    For each run, compute the Hamiltonian, its time derivative, and the Lyapunov derivative.

    Parameters:
    - num_runs: Number of different initial conditions to simulate.
    - dynamics: 'complete' or 'simplified' dynamics for the error system.
    - param_dict: Dictionary of model parameters.
    - initial_state_range: Tuple (low, high) for uniform random initial states.
    - seed: Random seed for reproducibility.
    - start_time, end_time: Simulation time span.
    - dt_initial: Initial time step for the solver.
    - n_points: Number of points to save in the time series.
    - transient_ratio: Fraction of initial time series to discard as transient.
    - max_steps: Maximum number of steps for the solver.
    - solver: Diffrax solver to use.
    - stepsize_controller: Controller for adaptive step sizing.
    - I_ext: External currents for the HR neurons.
    - xi: Coupling matrix for the HR neurons.
    - output_file: File path to save the results.

    Returns:
    - Saves a list of result dictionaries to the specified output file.
    """
    # Set random seed
    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(0)  # Default seed

    # Generate keys for each run
    keys = jax.random.split(key, num_runs)

    # Prepare list to store results
    all_results = []

    # Compute max_steps if not provided
    if max_steps is None:
        max_steps = int((end_time - start_time) / dt_initial) * 20

    # Time points for saving
    t_save = jnp.linspace(start_time, end_time, n_points)

    for i in range(num_runs):
        print(f"Running simulation {i+1}/{num_runs}...")

        # Generate random hr_initial_state
        low, high = initial_state_range
        hr_initial_state = jax.random.uniform(keys[i], shape=(10,), minval=low, maxval=high)

        # Create simulator instance
        simulator = HRNetworkErrorSystem(
            params=param_dict,
            dynamics=dynamics,
            hr_initial_state=hr_initial_state,
            I_ext=I_ext,
            hr_xi=xi
        )

        # Run simulation
        simulator.solve(
            solver=solver,
            t0=start_time,
            t1=end_time,
            dt0=dt_initial,
            n_points=dfx.SaveAt(ts=t_save, dense=True),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps
        )

        if not simulator.failed:
            results = simulator.get_results_dict(transient_ratio)

            # Compute Hamiltonian, dHdt, dVdt
            H = calculate_H(results, param_dict)
            dHdt = calculate_dHdt(results, param_dict)
            dVdt = calculate_dVdt(results, param_dict)

            # Add to results
            results['Hamiltonian'] = H
            results['dHdt'] = dHdt
            results['dVdt'] = dVdt

            # Also store initial state
            results['initial_state'] = hr_initial_state

            all_results.append(results)
        else:
            print(f"Simulation {i+1} failed.")

    # Save all results to file
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"Data generation complete. Results saved to {output_file}")



if __name__ == '__main__':
    from visualization.plotting import plot_pinn_data

    sim_params = DEFAULT_PARAMS.copy()
    sim_params['ge'] = 0.62

    import os
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'PINN Data/')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'error_system_data.pkl')
    # Generate data for 5 runs
    generate_data(num_runs=1,
                  dynamics = 'complete',
                  param_dict = sim_params,
                  initial_state_range = (-0.1, 0.1),
                  end_time = 1000,
                  n_points = 100000,
                  seed=42,
                  output_file=output_file
                  )

    # Load the generated data
    with open(output_file, 'rb') as f:
        all_results = pickle.load(f)
    print("Saved on file")

    # Use the imported plotting function to visualize the results
    plot_pinn_data(all_results, save_fig=True)

    print(f"Generated 5 separate plots, each with 8 subplots, saved as 'run_<number>_plots.png'")
