import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Define Directory ---
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'V_H_DOT_1D/')
os.makedirs(path, exist_ok=True)

file_name = 'V_H_DOT_rho_0.3_0.9.npz'

data = np.load(path + file_name)
PARAM_VALUES = data['PARAM_VALUES']
mean_dVdt = data['mean_dVdt']
mean_dHdt = data['mean_dHdt']
TARGET_PARAM = data['TARGET_PARAM']

# ── Combined Plot ───────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# mask out any non-finite values
x_vals = np.asarray(PARAM_VALUES, dtype=float)

# compute x-limits (assuming PARAM_VALUES has no NaN/Inf)
x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)

# create the figure and axes
fig, ax = plt.subplots(
    figsize=(4, 4),
    dpi=200,
    constrained_layout=True
)

# plot your two curves
ax.plot(PARAM_VALUES, mean_dVdt, color="blue", label=r'$dV/dt$')
ax.plot(PARAM_VALUES, mean_dHdt, color="red", label=r'$dH/dt$')

# optional: ensure ticks include endpoints
ax.set_xticks([x_min] + list(ax.get_xticks()) + [x_max])

# labels, title, grid, legend
ax.set_xlabel(r'$ρ$', fontsize=18)
ax.set_ylabel(r'$dV/dt, dH/dt$', fontsize=18)
ax.set_title("(c)", fontsize=16)
ax.grid(True)
ax.legend(loc="best", fontsize=14)

# enforce square aspect and remove x-margins
ax.set_box_aspect(1.0)
ax.set_xlim(x_min, x_max)
ax.margins(x=0)

png_path = os.path.join(path, f'rho.png')
eps_path = os.path.join(path, f'rho.eps')
fig.savefig(png_path, format='png', dpi=200)
fig.savefig(eps_path, format='eps')


