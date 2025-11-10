
"""
v_dot_parameter_sweep_2D.py
--------------------------------------------
Sweep two Hindmarsh‑Rose model parameters, simulate for each pair,
compute the mean post‑transient dV/dt, and plot the result as a colour map.
"""

from __future__ import annotations
import os
import numpy as np
import jax, jax.numpy as jnp
import diffrax as dfx
jax.config.update("jax_enable_x64", True)
import multiprocessing as mp
import gc
from typing import Any
from src.hr_model.error_system import HRNetworkErrorSystem
from laboratory.synchronization_quantities import calculate_dHdt
from laboratory.synchronization_quantities import calculate_dVdt
from src.hr_model.model import DEFAULT_PARAMS


# ───────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────
# ‑‑ choose the two parameters to sweep ‑‑
PARAM_X, PARAM_Y = "ge", "k"  # ← change names as needed

# ranges (inclusive)
X_MIN, X_MAX, NX = 0, 1, 3  # 11 points → 0.3 … 0.9 step 0.06
Y_MIN, Y_MAX, NY = -2.5, 2, 3  # 9  points → 0.1 … 0.5 step 0.05
x_vals = jnp.linspace(X_MIN, X_MAX, NX)
y_vals = jnp.linspace(Y_MIN, Y_MAX, NY)

# Multiprocessing params
CORE_NUM = 3
BATCH_SIZE = 9

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

# Initial conditions (10 HR‑state variables + 5 error‑state variables)
INITIAL_HR_STATE0 = [
    0.1, 0.2, 0.3, 0.4, 0.1,    # neuron1
    0.2, 0.3, 0.4, 0.5, 0.2     # neuron2
]

I_EXT = [0.8, 0.8]  # external currents
XI = [[0, 1], [1, 0]]  # electrical coupling

# where to save results (root directory)
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'V_H_DOT_2D/')
os.makedirs(SAVE_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────
# WORKER FUNCTION
# ───────────────────────────────────────────────────────────────────────
def run_one_pair(params: tuple[float, float]) -> float:
    """Return mean post‑transient dV/dt for (x, y) parameter pair."""
    # 1– copy default parameters and set the swept one
    x_val, y_val = params
    current_params = DEFAULT_PARAMS.copy()
    current_params[PARAM_X] = x_val
    current_params[PARAM_Y] = y_val

    # 2– create simulator
    simulator = HRNetworkErrorSystem(
        params=current_params,
        dynamics='simplified',
        hr_initial_state=INITIAL_HR_STATE0,
        I_ext=I_EXT,
        hr_xi=XI)

    # 3– integrate
    # print('param pair: ',[x_val, y_val])
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
def main() -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep PARAM_X × PARAM_Y, resume from partial runs, and keep RAM bounded.

    Returns
    -------
    (mean_dVdt, mean_dHdt)  – both shape (NX, NY)
    """
    # ── 1. Where to store results ──────────────────────────────────────────
    output_filename = os.path.join(
        SAVE_DIR,
        f"VDOT_{PARAM_X}_{X_MIN}-{X_MAX}_{PARAM_Y}_{Y_MIN}-{Y_MAX}.npz"
    )

    # ── 2. Load / initialise result arrays ─────────────────────────────────
    if os.path.exists(output_filename):
        print(f"--- Found existing results file: {output_filename}")
        with np.load(output_filename) as data:
            if data["mean_dVdt"].shape == (NX, NY):
                # .copy() ⇒ writable even if the NPZ is mem-mapped read-only
                mean_dVdt = data["mean_dVdt"].copy()
                mean_dHdt = data["mean_dHdt"].copy()
                print("--- Resuming previous sweep")
            else:
                print("--- Shape mismatch – starting fresh run")
                mean_dVdt = np.full((NX, NY), np.nan, dtype=np.float64)
                mean_dHdt = np.full((NX, NY), np.nan, dtype=np.float64)
    else:
        print("--- No checkpoint found – starting fresh run")
        mean_dVdt = np.full((NX, NY), np.nan, dtype=np.float64)
        mean_dHdt = np.full((NX, NY), np.nan, dtype=np.float64)

    # ── 3. Build todo list (need *both* matrices filled) ───────────────────
    tasks_to_do: list[dict[str, Any]] = []
    for i in range(NX):           # rows  → PARAM_X
        for j in range(NY):       # cols  → PARAM_Y
            if np.isnan(mean_dVdt[i, j]) or np.isnan(mean_dHdt[i, j]):
                tasks_to_do.append({"params": (x_vals[i], y_vals[j]),
                                   "indices": (i, j)})

    if not tasks_to_do:
        print("--- All simulations already complete! ---")
        return mean_dVdt, mean_dHdt

    print(f"Total simulations to run: {len(tasks_to_do)}")

    # ── 4. Batch-wise execution with ONE persistent worker pool ────────────
    with mp.Pool(processes=CORE_NUM, maxtasksperchild=1) as pool:
        for batch_no, start in enumerate(range(0, len(tasks_to_do), BATCH_SIZE), 1):
            current_batch  = tasks_to_do[start:start + BATCH_SIZE]
            params_batch   = [task["params"]   for task in current_batch]
            indices_batch  = [task["indices"]  for task in current_batch]

            print(f"\n--- Batch {batch_no}/{int(np.ceil(len(tasks_to_do)/BATCH_SIZE))} "
                  f"({start+1}–{min(start+BATCH_SIZE, len(tasks_to_do))}) ---")

            results_batch = pool.map(run_one_pair, params_batch)

            # insert into the big matrices
            for (dV, dH), (row, col) in zip(results_batch, indices_batch):
                mean_dVdt[row, col] = dV
                mean_dHdt[row, col] = dH

            # ── 5. Checkpoint after every batch ────────────────────────────
            print("--- Saving checkpoint ---")
            np.savez_compressed(
                output_filename,
                x_vals=x_vals,
                y_vals=y_vals,
                mean_dVdt=mean_dVdt,
                mean_dHdt=mean_dHdt,
                PARAM_X=PARAM_X,
                PARAM_Y=PARAM_Y
            )

    print("\n--- All simulations finished successfully! ---")
    return mean_dVdt, mean_dHdt



# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    mean_dVdt, mean_dHdt = main()

    # ── Plot colour map ────────────────────────────────────────────────
    from visualization.plotting import plot_v_h_dot_2d


    # Ensure the simulation ran and returned valid data before plotting
    if mean_dVdt is not None and mean_dHdt is not None:
        print("Generating plots...")

        # # Define the output path for the plots, using the data filename as a base
        # import os
        # plot_path_prefix = os.path.join(
        #     SAVE_DIR,
        #     f"VDOT_{PARAM_X}_{X_MIN}-{X_MAX}_{PARAM_Y}_{Y_MIN}-{Y_MAX}"
        # )

        plot_v_h_dot_2d(
            x_vals=x_vals,
            y_vals=y_vals,
            mean_dvdt=mean_dVdt,
            mean_dhdt=mean_dHdt,
            param_x_name=PARAM_X,
            param_y_name=PARAM_Y,
            save_fig=False
        )