
"""
v_dot_parameter_sweep_1D.py
------------------------------------------
Sweep a chosen Hindmarsh-Rose model parameter, simulate for each value,
compute the mean post-transient dV/dt, and plot the result … in parallel.
"""

from __future__ import annotations
import os
import numpy as np
import jax, jax.numpy as jnp
import diffrax as dfx
jax.config.update("jax_enable_x64", True)
import multiprocessing as mp
# ── extra imports (2-D features) ─────────────────────────────────────────
import gc
from typing import Any
# ─────────────────────────────────────────────────────────────────────────
from src.hr_model.error_system import HRNetworkErrorSystem
from laboratory.synchronization_quantities import calculate_dHdt
from laboratory.synchronization_quantities import calculate_dVdt
from src.hr_model.model import DEFAULT_PARAMS


# ───────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────
TARGET_PARAM = "k"  # name of the parameter to sweep
PARAM_MIN, PARAM_MAX = -2.5, 2
NUM_PARAM_POINTS = 5
PARAM_VALUES = jnp.linspace(PARAM_MIN, PARAM_MAX, NUM_PARAM_POINTS)

# ── extra config copied from 2-D script ────────────────────────────────
CORE_NUM   = 5          # number of worker processes
BATCH_SIZE = 5          # params per checkpoint-batch
# ───────────────────────────────────────────────────────────────────────

# integration settings
START_TIME = 0
END_TIME = 250
DT_INITIAL = 0.01
POINT_NUM = 250
TRANSIENT_RATIO = 0.75
N_POINTS = dfx.SaveAt(ts=jnp.linspace(START_TIME, END_TIME, POINT_NUM), dense=True)
MAX_STEPS = int((END_TIME - START_TIME) / DT_INITIAL) * 20
SOLVER = dfx.Tsit5()
STEPSIZE_CONTROLLER = dfx.PIDController(rtol=1e-10, atol=1e-12)
# stepsize_controller = dfx.ConstantStepSize()

# Initial conditions (10 HR-state variables + 5 error-state variables)
INITIAL_HR_STATE0 = [
    0.1, 0.2, 0.3, 0.4, 0.1,    # neuron1
    0.2, 0.3, 0.4, 0.5, 0.2     # neuron2
]

I_EXT = [0.8, 0.8]  # external currents
XI = [[0, 1], [1, 0]]  # electrical coupling

# where to save results (root directory)
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'V_H_DOT_1D/')
os.makedirs(SAVE_DIR, exist_ok=True)


# ───────────────────────────────────────────────────────────────────────
# WORKER FUNCTION (runs in each process)
# ───────────────────────────────────────────────────────────────────────
def run_one_param(param: float) -> float:
    """Simulate for a single parameter value and return mean post-transient dV/dt."""
    # 1– copy default parameters and set the swept one
    current_params = DEFAULT_PARAMS.copy()
    current_params[TARGET_PARAM] = param

    # 2– create simulator
    simulator = HRNetworkErrorSystem(
        params=current_params,
        dynamics='simplified',
        hr_initial_state=INITIAL_HR_STATE0,
        I_ext=I_EXT,
        hr_xi=XI)

    # 3– integrate
    simulator.solve(
        solver=SOLVER,
        t0=START_TIME,
        t1=END_TIME,
        dt0=DT_INITIAL,
        n_points=N_POINTS,
        stepsize_controller=STEPSIZE_CONTROLLER,
        max_steps=MAX_STEPS
    )

    # 4– analyse
    if not simulator.failed:
        results_dict = simulator.get_results_dict(TRANSIENT_RATIO)
        dVdt_timeseries = calculate_dVdt(results_dict, current_params)
        dHdt_timeseries = calculate_dHdt(results_dict, current_params)

        final_result = [float(jnp.nanmean(dVdt_timeseries)), float(jnp.nanmean(dHdt_timeseries))]

        # --- AGGRESSIVE CLEANUP ---
        del simulator, results_dict, dVdt_timeseries, dHdt_timeseries, current_params
        jax.clear_caches()
        gc.collect()

        return final_result
    else:
        # --- AGGRESSIVE CLEANUP (for the failure case too) ---
        del simulator, current_params
        jax.clear_caches()
        gc.collect()
        return [np.nan, np.nan]

# ───────────────────────────────────────────────────────────────────────
# MAIN DRIVER
# ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Starting parallel simulation loop for {TARGET_PARAM} "
          f"from {PARAM_MIN} to {PARAM_MAX}…")

    # ── new resume / checkpoint logic ────────────────────────────────────
    outfile = os.path.join(
        SAVE_DIR, f"V_H_DOT_{TARGET_PARAM}_{PARAM_MIN}_{PARAM_MAX}.npz"
    )

    if os.path.exists(outfile):
        print(f"--- Found existing results file: {outfile}")
        with np.load(outfile) as data:
            if data["mean_dVdt"].shape == (NUM_PARAM_POINTS,):
                mean_dVdt = data["mean_dVdt"].copy()
                mean_dHdt = data["mean_dHdt"].copy()
                print("--- Resuming previous sweep")
            else:
                print("--- Shape mismatch – starting fresh run")
                mean_dVdt = np.full(NUM_PARAM_POINTS, np.nan, dtype=np.float64)
                mean_dHdt = np.full(NUM_PARAM_POINTS, np.nan, dtype=np.float64)
    else:
        print("--- No checkpoint found – starting fresh run")
        mean_dVdt = np.full(NUM_PARAM_POINTS, np.nan, dtype=np.float64)
        mean_dHdt = np.full(NUM_PARAM_POINTS, np.nan, dtype=np.float64)

    # build todo list
    tasks: list[dict[str, Any]] = []
    for idx, p in enumerate(PARAM_VALUES):
        if np.isnan(mean_dVdt[idx]) or np.isnan(mean_dHdt[idx]):
            tasks.append({"param": p, "index": idx})

    if not tasks:
        print("--- All simulations already complete! ---")
        return mean_dVdt, mean_dHdt

    print(f"Total simulations to run: {len(tasks)}")

    # ── batch execution with RAM-safe pool ───────────────────────────────
    with mp.Pool(processes=CORE_NUM, maxtasksperchild=1) as pool:
        for b0 in range(0, len(tasks), BATCH_SIZE):
            batch      = tasks[b0:b0+BATCH_SIZE]
            params     = [d["param"]  for d in batch]
            indices    = [d["index"]  for d in batch]

            print(f"\n--- Batch {b0//BATCH_SIZE + 1}/{-(-len(tasks)//BATCH_SIZE)} "
                  f"({b0+1}–{b0+len(batch)}) ---")

            results    = pool.map(run_one_param, params)

            for (dV, dH), idx in zip(results, indices):
                mean_dVdt[idx] = dV
                mean_dHdt[idx] = dH

            # save checkpoint every batch
            print("--- Saving checkpoint ---")
            np.savez_compressed(
                outfile,
                PARAM_VALUES=PARAM_VALUES,
                mean_dVdt=mean_dVdt,
                mean_dHdt=mean_dHdt,
                TARGET_PARAM=TARGET_PARAM
            )

    print("\nSimulation done")
    print("Saved on a file")

    return mean_dVdt, mean_dHdt




# ───────────────────────────────────────────────────────────────────────
# ENTRY POINT  (required for multiprocessing on Windows & macOS)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # On Windows the default start-method is "spawn" already.
    # Explicitly setting it once avoids accidental "fork" on some *nix installs.
    mp.freeze_support()             # makes scripts double-clickable on Windows
    mp.set_start_method("spawn", force=True)
    mean_dVdt, mean_dHdt = main()

    # ── Plot ───────────────────────────────────────────────────────────
    from visualization.plotting import plot_v_h_dot_1d

    # Ensure the simulation ran and returned valid data before plotting
    if mean_dVdt is not None and mean_dHdt is not None:
        print("Generating plot...")

        # # Define the output path for the plot, same as the data file but with .png
        # import os
        # plot_filename = os.path.join(
        #     SAVE_DIR, f"V_H_DOT_{TARGET_PARAM}_{PARAM_MIN}_{PARAM_MAX}.png"
        # )

        plot_v_h_dot_1d(
            param_values=PARAM_VALUES,
            mean_dvdt=mean_dVdt,
            mean_dhdt=mean_dHdt,
            param_name=TARGET_PARAM,
            title=f"Mean Derivatives vs. Parameter '{TARGET_PARAM}'",
            save_fig=False
        )


