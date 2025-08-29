import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
import os # Add this import



# --- Define paths to the data directories relative to the data ---
path_bif = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'Bifurcation/')
os.makedirs(path_bif, exist_ok=True)

path_lya = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'Lyapanov 1D/')
os.makedirs(path_lya, exist_ok=True)

# --- Define output directory ---
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'Bif_Lya_plots')
os.makedirs(output_dir, exist_ok=True)



bif_names = ['Bif_k_rho_0.7_m_1.npy', 'Bif_k_rho_0.7_m_0.5.npy', 'Bif_k_rho_0.7_m_0.25.npy',
             'Bif_rho_k_0.87_m_1.npy', 'Bif_rho_k_0.87_m_0.5.npy', 'Bif_rho_k_0.87_m_0.25.npy']
lya_names = ['Lya_k_rho_0.7_m_1.txt', 'Lya_k_rho_0.7_m_0.5.txt', 'Lya_k_rho_0.7_m_0.25.txt',
             'Lya_rho_k_0.87_m_1.txt', 'Lya_rho_k_0.87_m_0.5.txt', 'Lya_rho_k_0.87_m_0.25.txt']
save_names = ['k_m_1', 'k_m_0.5', 'k_m_0.25',
              'rho_m_1', 'rho_m_0.5', 'rho_m_0.25']
x_names = [r'$k$', r'$k$', r'$k$', r'$\rho$', r'$\rho$', r'$\rho$']
titles = ['(a)', '(b)', '(c)', '(a)', '(b)', '(c)']

leg_poss = [(0, 0), (0, 0), (0, 0), (0, 0), (0.75, 0), (0, 0)]

for bif_name, lya_name, save_name, title, leg_pos, x_name in zip(
        bif_names, lya_names, save_names, titles, leg_poss, x_names):

    # Load data
    data_lya = np.array(pd.read_csv(path_lya+lya_name, delim_whitespace=True, header=None))
    data_bif = np.load(path_bif+bif_name)


    # Function to format ticks to 2 decimal places
    def format_func(value, tick_number):
        return f'{value:.2f}'


    # Define the size of each subplot (square dimensions)
    subplot_size = 6  # in inches

    # Calculate the total figure size
    total_width = subplot_size
    total_height = 1.25 * subplot_size

    # Create the figure and subplots with adjusted size and spacing
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(total_width, total_height),
        gridspec_kw={'height_ratios': [1, 1]}
    )
    fig.suptitle(title, fontsize=30, y=0.95)

    ### Subplot 1: Scatter Plot
    # Plot the data
    ax1.scatter(data_bif[:, 0], data_bif[:, 1], s=0.05, c='k')

    # Set labels and format
    ax1.set_ylabel(r'$x_{\text{max}}$', fontsize=25)
    ax1.set_xticks([])
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_xlim([data_bif[:, 0].min(), data_bif[:, 0].max()])
    ax1.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax1.yaxis.set_major_formatter(FuncFormatter(format_func))

    # Set y-limits to data range
    y_min_ax1 = data_bif[:, 1].min()
    y_max_ax1 = data_bif[:, 1].max()
    ax1.set_ylim([y_min_ax1, y_max_ax1])

    # Set y-ticks
    ax1.set_yticks(np.linspace(y_min_ax1, y_max_ax1, 5))

    # Adjust aspect ratio
    ax1.set_aspect('auto')

    ### Subplot 2: Line Plot
    # Plot the data
    ax2.plot(data_lya[:, 0], data_lya[:, 1], label="LE 1", linewidth=1.5, c='b')
    ax2.plot(data_lya[:, 0], data_lya[:, 3], label="LE 2", linewidth=1.5, c='r')

    # Set labels and format

    ax2.set_xlabel(x_name, fontsize=25)
    ax2.set_ylabel(r'$LEs$', fontsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_xlim([data_lya[:, 0].min(), data_lya[:, 0].max()])
    ax2.set_xticks(np.linspace(data_lya[:, 0].min(), data_lya[:, 0].max(), 5))
    ax2.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_func))
    ax2.legend(fontsize=12, loc='lower left', bbox_to_anchor=leg_pos)
    ax2.grid(True)

    # Set y-limits to data range

    y_min_ax2 = min(data_lya[:, 1][~np.isnan(data_lya[:, 1])].min(),
                    data_lya[:, 3][~np.isnan(data_lya[:, 3])].min())
    y_max_ax2 = max(data_lya[:, 1][~np.isnan(data_lya[:, 1])].max(),
                    data_lya[:, 3][~np.isnan(data_lya[:, 3])].max())
    ax2.set_ylim([y_min_ax2, y_max_ax2])

    # Set y-ticks
    ax2.set_yticks(np.linspace(y_min_ax2, y_max_ax2, 5))

    # Adjust aspect ratio
    ax2.set_aspect('auto')

    # **Adjust layout to prevent labels from being cut off**
    plt.tight_layout()

    # If labels are still cut off, adjust the margins manually
    # fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=0.3)

    # --- Save the figure in both PNG and EPS formats ---
    base_output_path = os.path.join(output_dir, save_name)
    plt.savefig(base_output_path + '.png', format='png', dpi=100)
    plt.savefig(base_output_path + '.eps', format='eps')
    plt.close()