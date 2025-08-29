import os
import diffrax as dfx
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from src.hr_model.model import HindmarshRose, DEFAULT_PARAMS, DEFAULT_STATE0


# ==============================================================================
# 1. LOCAL PLOTTING FUNCTIONS
# ==============================================================================

def plot_time_series(t, x, title_suffix, save_path_prefix=None):
    """Plots the time series of variable x after a transient period."""

    plt.figure(figsize=(8, 8), layout='tight')
    plt.suptitle(f'{title_suffix}', fontsize=30, y=0.957)
    plt.plot(t, x, c='k')
    plt.xlabel(r'$t$', fontsize=25)
    plt.ylabel(r'$x$', fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=20)

    if len(t) > 0:
        t_min, t_max = t.min(), t.max()
        plt.xlim([t_min, t_max + (t_max - t_min) * 0.01])

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if save_path_prefix:
        plt.savefig(save_path_prefix + '.png', format='png', dpi=300)
        plt.savefig(save_path_prefix + '.eps', format='eps')
        plt.close()
    else:
        plt.show(block=True)


def plot_phase_portrait(x, y, title_suffix, save_path_prefix=None):
    """Plots the phase portrait (x vs y) after a transient period."""

    plt.figure(figsize=(8, 8), layout='tight')
    plt.suptitle(f'{title_suffix}', fontsize=30, y=0.957)
    plt.plot(x, y, c='k')
    plt.xlabel(r'$x$', fontsize=25)
    plt.ylabel(r'$y$', fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=20)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if save_path_prefix:
        plt.savefig(save_path_prefix + '.png', format='png', dpi=300)
        plt.savefig(save_path_prefix + '.eps', format='eps')
        plt.close()
    else:
        plt.show(block=True)


# ==============================================================================
# 2. MAIN SCRIPT LOGIC
# ==============================================================================

# --- Define the two specific simulation setups ---
setups = [
    {'name': 'rho_fixed', 'rho': 0.7, 'k': 0.36, 'm': 0.25},
    {'name': 'rho_fixed', 'rho': 0.7, 'k': 0.36, 'm': 0.5},
    {'name': 'rho_fixed', 'rho': 0.7, 'k': 0.36, 'm': 1},
    {'name': 'k_fixed', 'rho': 0.7, 'k': 0.87, 'm': 0.25},
    {'name': 'k_fixed', 'rho': 0.7, 'k': 0.87, 'm': 0.5},
    {'name': 'k_fixed', 'rho': 0.7, 'k': 0.87, 'm': 1}
]

# --- Define Output Directory ---
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'Phase Portrait')
os.makedirs(output_dir, exist_ok=True)
print(f"Plots will be saved to: {output_dir}")

# --- Loop Through Setups and Run Simulations ---
for setup_params in setups:
    m_val = setup_params['m']
    setup_name = setup_params['name']
    print(f"\n--- Running simulation for setup: '{setup_name}' (m = {m_val}) ---")

    # --- Use simulation example from model.py ---
    sim_params = DEFAULT_PARAMS.copy()
    sim_params.update(setup_params)  # Update params for the current setup

    # Create the model instance
    hr_model = HindmarshRose(N=1, params=sim_params, initial_state=DEFAULT_STATE0, I_ext=0.8, xi=0)

    # Integration settings
    start_time = 0
    end_time = 2000
    dt_initial = 0.05
    n_points = 20000
    max_steps = int((end_time - start_time) / dt_initial) * 20
    solver = dfx.Tsit5()
    stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-10)

    # Run the solver
    hr_model.solve(
        solver=solver, t0=start_time, t1=end_time, dt0=dt_initial, n_points=n_points,
        stepsize_controller=stepsize_controller, max_steps=max_steps)

    # Get results, discarding the first 75% as transient
    transient_discard_ratio = 0.75
    results = hr_model.get_results_dict(transient_ratio=transient_discard_ratio)
    t_sol, x_sol, y_sol = results['t'], results['x1'], results['y1']

    # --- Use exact plotting logic from the original script ---
    if m_val == 1:
        xt_title_suffix, xy_title_suffix = '(a1)', '(a2)'
    elif m_val == 0.5:
        xt_title_suffix, xy_title_suffix = '(b1)', '(b2)'
    elif m_val == 0.25:
        xt_title_suffix, xy_title_suffix = '(c1)', '(c2)'
    else:
        xt_title_suffix, xy_title_suffix = '(x1)', '(x2)'

    # --- Create filenames and save plots ---
    base_filename_ts = f"timeseries_{setup_name}_{xt_title_suffix.strip('()')}"
    save_path_prefix_ts = os.path.join(output_dir, base_filename_ts)

    base_filename_pp = f"phase_portrait_{setup_name}_{xy_title_suffix.strip('()')}"
    save_path_prefix_pp = os.path.join(output_dir, base_filename_pp)

    # Plot and save the time series using the local function
    plot_time_series(t_sol, x_sol, xt_title_suffix, save_path_prefix=save_path_prefix_ts)

    # Plot and save the phase portrait using the local function
    plot_phase_portrait(x_sol, y_sol, xy_title_suffix, save_path_prefix=save_path_prefix_pp)

print("\nAll simulations and plots saved successfully.")