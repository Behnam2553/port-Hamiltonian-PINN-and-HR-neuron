import numpy as np
import jax
import diffrax as dfx
jax.config.update("jax_enable_x64", True)
import copy
from src.hr_model.model import HindmarshRose, DEFAULT_PARAMS, DEFAULT_STATE0
from scipy.signal import find_peaks
import multiprocessing as mp
from functools import partial


# ==============================================================================
# WORKER FUNCTION (for multiprocessing)
# ==============================================================================
def run_one_simulation(current_bif_value, model_instance, initial_state,
                       bifurcation_param_name, start_time, end_time, dt_initial,
                       n_points, max_steps, solver, stepsize_controller,
                       transient_fraction, positive_peaks_only):
    """
    Runs a single simulation for one specific bifurcation parameter value.
    This function is designed to be called by a multiprocessing worker.

    Returns:
        tuple: (parameter_value, list_of_peaks)
    """
    # --- Setup Instance ---
    # Create a deep copy to ensure each process has its own model instance
    current_instance = copy.deepcopy(model_instance)

    # --- FIX IS HERE ---
    # Modify the parameter inside the 'params' dictionary directly.
    current_instance.params[bifurcation_param_name] = current_bif_value

    current_instance.initial_state = np.array(initial_state, dtype=np.float64)

    # --- Run Simulation ---
    current_instance.solve(
        solver=solver, t0=start_time, t1=end_time, dt0=dt_initial, n_points=n_points,
        stepsize_controller=stepsize_controller, max_steps=max_steps)

    # --- Process Results ---
    if current_instance.failed:
        return (current_bif_value, [np.nan])

    results = current_instance.get_results_dict(transient_fraction)
    x_curve = results['x1']

    # Find peaks in the steady-state portion
    peak_idx, _ = find_peaks(x_curve, height=0 if positive_peaks_only else None)
    peaks = x_curve[peak_idx]

    # Mask out negative peaks if requested
    if positive_peaks_only:
        positive_mask = peaks > 0
        peaks = peaks[positive_mask]

    # If no peaks are found, return NaN to maintain array structure
    if peaks.size == 0:
        peaks = np.array([np.nan])

    return (current_bif_value, peaks.tolist())


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Main function to set up and run the parallel bifurcation analysis.
    """
    # --- Simulation Parameters ---
    sim_params = DEFAULT_PARAMS.copy()
    sim_params['rho'] = 0.7
    sim_params['m'] = 1
    model_instance = HindmarshRose(N=1, params=sim_params, initial_state=DEFAULT_STATE0, I_ext=0.8, xi=0)
    bifurcation_param_name = 'k'
    param_range = (-1.3, 0.6, 400)  # Increased steps to show benefit of parallelization

    # --- Integration Settings ---
    start_time = 0
    end_time = 2000
    dt_initial = 0.05
    n_points = int(end_time / dt_initial)
    transient_fraction = 0.5
    max_steps = int((end_time - start_time) / dt_initial) * 20
    solver = dfx.Tsit5()
    stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-10)

    # --- Parallel Execution ---
    param_start, param_end, param_steps = param_range
    bifurcation_values = np.linspace(param_start, param_end, int(param_steps))
    MAX_WORKERS = mp.cpu_count()
    # MAX_WORKERS = 8

    print(f"Starting parallel bifurcation analysis for '{bifurcation_param_name}'...")
    print(f"Running {param_steps} simulations on {MAX_WORKERS} cores.")

    # Use functools.partial to "freeze" the arguments that are the same for all simulations
    worker_func = partial(run_one_simulation,
                          model_instance=model_instance,
                          initial_state=DEFAULT_STATE0,
                          bifurcation_param_name=bifurcation_param_name,
                          start_time=start_time,
                          end_time=end_time,
                          dt_initial=dt_initial,
                          n_points=n_points,
                          max_steps=max_steps,
                          solver=solver,
                          stepsize_controller=stepsize_controller,
                          transient_fraction=transient_fraction,
                          positive_peaks_only=True)

    # Create a pool of worker processes and map the tasks

    with mp.Pool(processes=MAX_WORKERS) as pool:
        # pool.map will distribute bifurcation_values among the workers
        # and collect the results in a list once all are complete.
        results = pool.map(worker_func, bifurcation_values)

    print("All simulations finished. Processing results...")

    # --- Aggregate Results ---
    all_param_values = []
    all_peak_values = []
    for param_val, peaks in results:
        all_param_values.extend([param_val] * len(peaks))
        all_peak_values.extend(peaks)

    # --- Plotting ---
    # from visualization.plotting import plot_bifurcation_diagram

    # plot_bifurcation_diagram(
    #     param_values=all_param_values,
    #     peak_values=all_peak_values,
    #     bifurcation_param_name=bifurcation_param_name,
    #     title="Bifurcation Diagram",
    #     ylabel=r'$x_{max}$',
    #     s=2  # Marker size
    # )

    # --- Optional: Save Data ---
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'Bifurcation/')
    os.makedirs(path, exist_ok=True)

    gen_data = np.vstack([all_param_values, all_peak_values]).T
    np.save(
        path + 'Bif_'+bifurcation_param_name+'_k_'+str(sim_params['k'])+'_m_'+str(sim_params['m']),
        gen_data)


# This check is essential for multiprocessing to work correctly on all platforms
if __name__ == "__main__":
    # On Windows, the 'spawn' start method is the default and safest.
    # Explicitly setting it prevents issues on systems where 'fork' is the default.
    mp.set_start_method("spawn", force=True)
    main()
