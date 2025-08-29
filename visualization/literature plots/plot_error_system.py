import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import diffrax as dfx
import jax.numpy as jnp
from src.hr_model.error_system import HRNetworkErrorSystem
from src.hr_model.model import DEFAULT_PARAMS

# ======================================================================
# 1. SIMULATION SETUP & EXECUTION
# ======================================================================

# --- Initial Conditions & Parameters ---
# Initial state (x, y, z, u, φ for each neuron)
INITIAL_HR_STATE0 = [
    0.1, 0.2, 0.3, 0.4, 0.1,   # neuron 1
    0.2, 0.3, 0.4, 0.5, 0.2    # neuron 2
]

# [cite_start]External currents and coupling matrix [cite: 151]
I_ext = [0.8, 0.8]
xi = [[0, 1], [1, 0]]
sim_params = DEFAULT_PARAMS.copy()

# --- Create Simulator Instance ---
# Using 'complete' dynamics as in the example
simulator = HRNetworkErrorSystem(params=sim_params, dynamics='complete',
                                 hr_initial_state=INITIAL_HR_STATE0, I_ext=I_ext, hr_xi=xi)

# --- Integration Settings ---
start_time = 0
end_time = 1000
dt_initial = 0.01
point_num = 10000
transient_ratio = 0
n_points = dfx.SaveAt(ts=jnp.linspace(start_time, end_time, point_num), dense=True)
max_steps = int((end_time - start_time) / dt_initial) * 20
solver = dfx.Tsit5()
stepsize_controller = dfx.PIDController(rtol=1e-10, atol=1e-12)

# --- Run Simulation ---
print("Running simulation...")
simulator.solve(
    solver=solver,
    t0=start_time,
    t1=end_time,
    dt0=dt_initial,
    n_points=n_points,
    stepsize_controller=stepsize_controller,
    max_steps=max_steps
)
print("Simulation finished.")

# --- Get Results ---
results = simulator.get_results_dict(transient_ratio)


# ======================================================================
# 2. PLOTTING
# This section is preserved from the original plot_error_system.py
# ======================================================================

# --- Define the Directory ---
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'Error System Dynamics Dynamics')
os.makedirs(output_dir, exist_ok=True)
print(f"Saving figures to '{output_dir}/'")

# --- Define Plotting Series ---
series = [
    ('e_x', r'$e_x$', 'darkblue', '(a)'),
    ('e_y', r'$e_y$', 'C2', '(b)'),
    ('e_z', r'$e_z$', 'C1', '(c)'),
    ('e_u', r'$e_u$', 'tomato', '(d)'),
    ('e_phi', r'$e_\phi$', 'purple', '(e)'),
]

# --- Generate and Save Plots ---
for key, label, color, title in series:
    # Create a new square figure with high resolution and tight layout
    fig, ax = plt.subplots(
        figsize=(4, 4),
        dpi=200,
        constrained_layout=True)

    # Plot the data
    ax.plot(results['t'], results[key], c=color, linewidth=1.5)

    # Set labels, title, grid, aspect ratio, and margins
    ax.set_xlabel(r'$t$',   fontsize=18)
    ax.set_ylabel(label,    fontsize=18)
    ax.set_title(f'{title.strip("$")}', fontsize=20)
    ax.grid(True)
    ax.set_box_aspect(1.0)
    ax.margins(x=0)

    # Define save paths and save the figure in both PNG and EPS formats
    png_path = os.path.join(output_dir, f'{key}.png')
    eps_path = os.path.join(output_dir, f'{key}.eps')
    fig.savefig(png_path, format='png', dpi=200)
    fig.savefig(eps_path, format='eps')

    plt.close(fig)

print("All plots have been generated and saved successfully.")