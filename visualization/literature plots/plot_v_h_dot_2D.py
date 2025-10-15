# plot_vdot_2d_swapped.py  – PARAM_X on x-axis, PARAM_Y on y-axis
# --------------------------------------------------------------

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# ── Edit these two lines to your file location ─────────────────────────
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'V_H_DOT_2D/')
os.makedirs(path, exist_ok=True)

file_name = 'VDOT_ge_0-1_rho_0.3-0.9.npz'

# ── Load data ──────────────────────────────────────────────────────────
data = np.load(path + file_name)
x_vals     = data['x_vals']                 # PARAM_X values
y_vals     = data['y_vals']                 # PARAM_Y values
mean_dVdt  = data['mean_dVdt']              # shape (NX, NY)
mean_dHdt  = data['mean_dHdt']

# maskout some values
mean_dHdt[np.where(mean_dHdt < -1)] = np.nan
mean_dHdt[np.where(mean_dHdt > 350)] = np.nan

PARAM_X = data['PARAM_X'].item()
PARAM_Y = data['PARAM_Y'].item()

# ── Build grid with x on rows, y on columns (needs transpose) ──────────
Xg, Yg = np.meshgrid(x_vals, y_vals)        # shapes (NY, NX)

# Helper to compute 5 ticks from data grid
def five_ticks(grid):
    mn, mx = np.min(grid), np.max(grid)
    ticks = np.linspace(mn, mx, 5)
    ticks = np.round(ticks, 1)
    return ticks

# ── Plot 1 : 〈dV/dt〉 ───────────────────────────────────────────────────
fig_dV, ax_dV = plt.subplots(figsize=(4, 4), dpi=200, constrained_layout=True)

ax_dV.set_box_aspect(1.0)
pcm1 = ax_dV.pcolormesh(Xg, Yg, mean_dVdt.T, shading='auto', cmap='seismic')
ax_dV.set_xlabel(r'$ge$', fontsize=18)
ax_dV.set_ylabel(r'$\rho$', fontsize=18)
ax_dV.set_title('(a3)', fontsize=18)

# set 5 ticks on x and y
xticks = five_ticks(Xg)
yticks = five_ticks(Yg)
ax_dV.set_xticks(xticks)
ax_dV.set_yticks(yticks)
ax_dV.set_xticklabels([f"{t:.1f}" for t in xticks])
ax_dV.set_yticklabels([f"{t:.1f}" for t in yticks])

# colorbar with title‐label on top
cbar1 = fig_dV.colorbar(pcm1, ax=ax_dV, pad=0.0001, fraction=0.05)
cbar1.ax.set_title(r'$dV/dt$', pad=7, fontdict={'fontsize':15})

# ── Plot 2 : 〈dH/dt〉 ───────────────────────────────────────────────────
fig_dH, ax_dH = plt.subplots(figsize=(4, 4), dpi=200, constrained_layout=True)

ax_dH.set_box_aspect(1.0)
pcm2 = ax_dH.pcolormesh(Xg, Yg, mean_dHdt.T, shading='auto', cmap='seismic')
ax_dH.set_xlabel(r'$ge$', fontsize=18)
ax_dH.set_ylabel(r'$\rho$', fontsize=18)
ax_dH.set_title('(b3)', fontsize=18)

# set 5 ticks on x and y
ax_dH.set_xticks(xticks)
ax_dH.set_yticks(yticks)
ax_dH.set_xticklabels([f"{t:.1f}" for t in xticks])
ax_dH.set_yticklabels([f"{t:.1f}" for t in yticks])

cbar2 = fig_dH.colorbar(pcm2, ax=ax_dH, pad=0.0001, fraction=0.05)
cbar2.ax.set_title(r'$dH/dt$', pad=7, fontdict={'fontsize':15})

png_dV_path = os.path.join(path, f'rho_V.png')
eps_dV_path = os.path.join(path, f'rho_V.eps')
fig_dV.savefig(png_dV_path, format='png', dpi=200)
fig_dV.savefig(eps_dV_path, format='eps')

png_dH_path = os.path.join(path, f'rho_H.png')
eps_dH_path = os.path.join(path, f'rho_H.eps')
fig_dH.savefig(png_dH_path, format='png', dpi=200)
fig_dH.savefig(eps_dH_path, format='eps')


