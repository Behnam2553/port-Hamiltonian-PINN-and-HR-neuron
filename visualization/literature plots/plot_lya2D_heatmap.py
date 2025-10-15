import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import os



import matplotlib
matplotlib.use("Qt5Agg")  # <-- assuming you meant Qt5Agg (q5tgg)
import matplotlib.pyplot as plt


# --- Define the Directory ---
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'Lyapanov 2D/')
os.makedirs(path, exist_ok=True)

file_names = ['Max_Lya_k_m.txt', 'Max_Lya_rho_m.txt', 'Max_Lya_rho_k_m_1.txt',
              'Max_Lya_rho_k_m_0.5.txt', 'Max_Lya_rho_k_m_0.25.txt']

save_names = ['K_M', 'Rho_M', 'Rho_K0_M_1',
              'Rho_K0_M_0.5', 'Rho_K0_M_0.25']

x_names = [r'$k$', r'$\rho$', r'$\rho$', r'$\rho$', r'$\rho$']
y_names = [r'$m$', r'$m$', r'$k$', r'$k$', r'$k$']
titles = ['(a)', '(b)', '(c1)', '(c2)', '(c3)']

global_min = float('inf')
global_max = float('-inf')

# First pass: Determine global min and max across all files
for file_name in file_names:
    df = pd.read_csv(path + file_name, delim_whitespace=True, header=None)
    pivot_table = df.pivot(index=0, columns=2, values=1).fillna(0)
    array_2d = np.rot90(pivot_table.to_numpy(), k=1)
    global_min = min(global_min, np.nanmin(array_2d))
    global_max = max(global_max, np.nanmax(array_2d))

# Iterate through files again to create plots with consistent colorbar
for file_name, save_name, x_name, y_name, title in zip(
        file_names, save_names, x_names, y_names, titles):

    # Read the file into a DataFrame
    df = pd.read_csv(path + file_name, delim_whitespace=True, header=None)

    # Pivot the DataFrame to reshape it into a 2D array
    pivot_table = df.pivot(index=0, columns=2, values=1)

    # Fill NaNs with a value if necessary, h.g., 0
    pivot_table = pivot_table.fillna(0)

    # Convert to a 2D numpy array
    array_2d = pivot_table.to_numpy().T

    # Define custom colormap: black to yellow to red
    colors = ['black', 'yellow', 'red']
    n_bins = int((global_max * 100 / (
                global_max - global_min)) * 128) if global_min != global_max else 256  # Handle case where min == max
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Combine white for negative values with the custom colormap
    if global_min < 0:
        cmap_pos = cmap_custom(np.linspace(0, 1, n_bins))
        cmap_min_len = int(
            (abs(global_min) * 100 / (global_max - global_min)) * 128) if global_min != global_max else 128
        cmap_min = np.array([[1, 1, 1, 1]] * cmap_min_len)
        cmap_combined = np.vstack((cmap_min, cmap_custom(np.linspace(0, 1, n_bins))))
        cmap = mcolors.ListedColormap(cmap_combined)
        vmin_plot = global_min
        vmax_plot = global_max
    else:
        cmap = cmap_custom
        vmin_plot = global_min
        vmax_plot = global_max

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    plt.suptitle(title, fontsize=30, y=0.90)
    img = ax.imshow(array_2d, aspect='auto', cmap=cmap, origin='lower',
                    extent=[pivot_table.index.min(), pivot_table.index.max(),
                            pivot_table.columns.min(), pivot_table.columns.max()],
                    vmin=vmin_plot, vmax=vmax_plot)

    # --- Mouse readout: x, y, z ---
    # Use the same array you plotted:
    arr = array_2d  # if you plotted 'array_2d' in imshow
    # arr = masked.filled(np.nan)  # use this instead if you plotted 'masked'

    # Get the plotted extent from the image (left, right, bottom, top)
    x0, x1, y0, y1 = img.get_extent()

    # Array shape: rows (y), cols (x)
    ny, nx = arr.shape
    dx = (x1 - x0) / (nx - 1) if nx > 1 else 1.0
    dy = (y1 - y0) / (ny - 1) if ny > 1 else 1.0


    def format_coord(x, y):
        col = int(round((x - x0) / dx))
        row = int(round((y - y0) / dy))
        if 0 <= row < ny and 0 <= col < nx:
            z = arr[row, col]
            if np.isnan(z):
                return f"x={x:.3f}, y={y:.3f}, z=NaN"
            return f"x={x:.3f}, y={y:.3f}, z={z:.5g}"
        return f"x={x:.3f}, y={y:.3f}"


    ax.format_coord = format_coord
    # <<< end add this >>>

    # Add colorbar with reduced distance from plot
    cbar = fig.colorbar(img, ax=ax, pad=0.02)

    # Set colorbar ticks
    colorbar_ticks = []
    if global_min < 0:
        colorbar_ticks.append(round(global_min, 2))
    colorbar_ticks.append(0.00)
    if global_max > 0:
        num_intervals = 3
        positive_ticks = np.linspace(0, global_max, num_intervals + 1)[1:]  # Exclude 0
        colorbar_ticks.extend([round(tick, 2) for tick in positive_ticks])
        colorbar_ticks.append(round(global_max, 2))
    elif global_max == 0 and global_min < 0:
        colorbar_ticks.append(round(global_max, 2))

    colorbar_ticks = sorted(list(set(colorbar_ticks)))  # Remove duplicates and sort
    cbar.set_ticks(colorbar_ticks)
    cbar.set_ticklabels([f'{tick:.2f}' for tick in colorbar_ticks])

    # Set colorbar label at the top
    cbar.ax.set_title(r'$\lambda_{\mathrm{max}}$', pad=20, fontsize=25)

    # Set axis labels
    ax.set_xlabel(x_name, fontsize=25)
    ax.set_ylabel(y_name, fontsize=25)

    # Set font size for axis ticks
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Set font size for colorbar ticks
    cbar.ax.tick_params(labelsize=20)

    # --- Set x-axis ticks ---
    x_min, x_max = pivot_table.index.min(), pivot_table.index.max()
    x_ticks = np.linspace(x_min, x_max, 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{tick:.2f}' for tick in x_ticks])

    # --- Set y-axis ticks ---
    y_min, y_max = pivot_table.columns.min(), pivot_table.columns.max()
    y_ticks = np.linspace(y_min, y_max, 5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks])

    # Adjust layout to prevent cropping
    plt.tight_layout()

    # --- Save the figure in both PNG and EPS formats ---
    base_output_path = os.path.join(path, save_name)
    plt.savefig(base_output_path + '.png', format='png', dpi=1000, bbox_inches="tight")
    # plt.savefig(base_output_path + '.eps', format='eps')
    plt.close()

    # plt.show(block=True)  # show in interactive window (PyCharm)
    # plt.close()
