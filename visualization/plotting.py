import os
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Qt5agg')


def plot_all_time_series(results, N, title="Hindmarsh-Rose Neuron States", save_fig=False):
    """
    Plots the time series of all state variables for each neuron in a grid.

    Args:
        results (dict): The dictionary of simulation results from get_results_dict.
        N (int): The number of neurons in the simulation.
        title (str): The main title for the entire figure.
        save_fig (bool): If True, saves the figure to a file. Otherwise, shows it interactively.
    """
    states = ['x', 'y', 'z', 'u', 'phi']
    state_labels = [r'$x$', r'$y$', r'$z$', r'$u$', r'$\phi$']

    fig, axes = plt.subplots(5, N, figsize=(4 * N, 10), sharex=True, squeeze=False, dpi=300)
    fig.suptitle(title, fontsize=16)

    for i, (state_var, state_label) in enumerate(zip(states, state_labels)):
        for j in range(N):
            ax = axes[i, j]
            data_key = f'{state_var}{j + 1}'

            if data_key in results:
                ax.plot(results['t'], results[data_key])
                ax.grid(True, linestyle='--', alpha=0.6)

            if j == 0:
                ax.set_ylabel(state_label, fontsize=14)

            if i == 0:
                ax.set_title(f'Neuron {j + 1}')

    plt.xlabel('Time', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_fig:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'temp')
        os.makedirs(output_dir, exist_ok=True)
        filename = title.replace(' ', '_').replace(':', '') + '.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Figure saved to {save_path}")
    else:
        plt.show(block=True)


def plot_error_and_state_differences(results, title="Error System and State Difference Dynamics", save_fig=False):
    """
    Plots both the error system states (e.g., e_x) and the direct state
    differences (e.g., x2-x1) on the same subplots for comparison.
    Args:
        results (dict): The dictionary of simulation results.
        title (str): The main title for the figure.
        save_fig (bool): If True, saves the figure to a file. Otherwise, shows it interactively.
    """
    states = ['x', 'y', 'z', 'u', 'phi']
    error_labels = [r'$e_x$', r'$e_y$', r'$e_z$', r'$e_u$', r'$e_\phi$']
    diff_labels = [r'$x_2 - x_1$', r'$y_2 - y_1$', r'$z_2 - z_1$', r'$u_2 - u_1$', r'$\phi_2 - \phi_1$']

    fig, axes = plt.subplots(len(states), 1, figsize=(12, 15), sharex=True)
    fig.suptitle(title, fontsize=16)

    for i, state_var in enumerate(states):
        ax = axes[i]
        error_key = f'e_{state_var}'
        if error_key in results:
            ax.plot(results['t'], results[error_key], label=error_labels[i])

        key1 = f'{state_var}1'
        key2 = f'{state_var}2'
        if key1 in results and key2 in results:
            difference = results[key2] - results[key1]
            ax.plot(results['t'], difference, label=diff_labels[i], alpha=0.8)

        ax.set_ylabel(f"State '{state_var}'", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')

    plt.xlabel('Time', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_fig:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'temp')
        os.makedirs(output_dir, exist_ok=True)
        filename = title.replace(' ', '_').replace(':', '') + '.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Figure saved to {save_path}")
    else:
        plt.show(block=True)


def plot_bifurcation_diagram(param_values, peak_values, bifurcation_param_name, title="Bifurcation Diagram",
                             xlabel=None, ylabel=r'$x_{max}$', marker='.', s=1, save_fig=False):
    """Plots a bifurcation diagram."""

    if xlabel is None:
        greek_letters = {'rho': r'$\rho$', 'alpha': r'$\alpha$', 'beta': r'$\beta$', 'gamma': r'$\gamma$'}
        xlabel = greek_letters.get(bifurcation_param_name.lower(), bifurcation_param_name)

    plt.figure(layout='tight')
    plt.scatter(param_values, peak_values, marker=marker, s=s, alpha=0.8)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"{title} ({bifurcation_param_name})", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_fig:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'temp')
        os.makedirs(output_dir, exist_ok=True)
        filename = f"bifurcation_{bifurcation_param_name}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Figure saved to {save_path}")
    else:
        plt.show(block=True)

def plot_hamiltonian(t, H, save_fig=False):
    """Plots the Hamiltonian (H)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(t, H, linewidth=1, color='blue')

    # Styling to match reference
    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel(r'$H$', fontsize=18)
    ax.grid(True)
    ax.set_title(f'{"(a)".strip("$")}', fontsize=16)
    ax.set_box_aspect(1.0)
    fig.tight_layout()

    if save_fig:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'temp')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'hamiltonian_vs_time.png')  # keep original variable/name
        eps_path  = os.path.join(output_dir, 'hamiltonian_vs_time.eps')

        fig.savefig(save_path, format='png', dpi=200)
        fig.savefig(eps_path,  format='eps')
        plt.close(fig)
        print(f"Figure saved to {save_path} and {eps_path}")
    else:
        plt.show(block=True)


def plot_hamiltonian_derivative(t, dHdt, save_fig=False):
    """Plots the Hamiltonian derivative (dH/dt)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(t, dHdt, linewidth=1, color='red')

    # Styling to match reference
    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel(r'$\mathrm{d}H/\mathrm{d}t$', fontsize=18)
    ax.grid(True)
    ax.set_title(f'{"(b)".strip("$")}', fontsize=16)
    ax.set_box_aspect(1.0)
    fig.tight_layout()

    if save_fig:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'temp')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'dHdt_vs_time.png')  # keep original variable/name
        eps_path  = os.path.join(output_dir, 'dHdt_vs_time.eps')

        fig.savefig(save_path, format='png', dpi=200)
        fig.savefig(eps_path,  format='eps')
        plt.close(fig)
        print(f"Figure saved to {save_path} and {eps_path}")
    else:
        plt.show(block=True)


def plot_lyapunov_derivative(t, dVdt, save_fig=False):
    """Plots the Lyapunov derivative (dV/dt)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, dVdt, label='dV/dt', color='green')

    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel(r'$\mathrm{d}V/\mathrm{d}t$', fontsize=18)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.2)
    ax.legend()

    # Tight layout to reduce clipping
    fig.tight_layout()

    if save_fig:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'temp')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'dVdt_vs_time.png')
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Figure saved to {save_path}")
    else:
        plt.show(block=True)



def plot_pinn_data(all_results, save_fig=False):
    """
    Plots the results from the data generation for the PINN.
    Creates a separate figure for each simulation run, with subplots for key variables.
    Args:
        all_results (list): A list of result dictionaries for each run.
        save_fig (bool): If True, saves the figures to files. Otherwise, shows them interactively.
    """
    variables_to_plot = ['e_x', 'e_y', 'e_z', 'e_u', 'e_phi', 'Hamiltonian', 'dHdt', 'dVdt']
    titles = [r'$e_x$', r'$e_y$', r'$e_z$', r'$e_u$', r'$e_\phi$', r'$H$', r'$dH/dt$', r'$dV/dt$']

    if save_fig:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'temp')
        os.makedirs(output_dir, exist_ok=True)

    for run_idx, results in enumerate(all_results):
        t = results['t']
        fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
        fig.suptitle(f'Run {run_idx + 1}', fontsize=16)

        for ax, var, title in zip(axes.flat, variables_to_plot, titles):
            ax.plot(t, results[var])
            ax.set_xlabel('t')
            ax.set_ylabel(title)
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_fig:
            filename = f'pinn_data_run_{run_idx + 1}.png'
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show(block=True)


    if save_fig:
        print(f"All PINN data plots saved to {output_dir}")


def plot_v_h_dot_1d(param_values, mean_dvdt, mean_dhdt, param_name, title, save_fig=False):
    """
    Plots the mean dV/dt and dH/dt against a single parameter.
    Args:
        param_values (np.ndarray): The values of the swept parameter.
        mean_dvdt (np.ndarray): The corresponding mean dV/dt values.
        mean_dhdt (np.ndarray): The corresponding mean dH/dt values.
        param_name (str): The name of the parameter (e.g., 'k', 'rho').
        title (str): The title for the plot.
        save_fig (bool): If True, saves the figure to a file. Otherwise, shows it interactively.
    """
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(param_values, mean_dvdt, color="blue", label=r'$\langle dV/dt \rangle$', marker='.', linestyle='-')
    ax.plot(param_values, mean_dhdt, color="red", label=r'$\langle dH/dt \rangle$', marker='.', linestyle='-')

    greek_letters = {'rho': r'$\rho$', 'alpha': r'$\alpha$', 'k': r'$k$'}
    xlabel = greek_letters.get(param_name.lower(), param_name)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(r'Mean Value', fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.grid(True)
    ax.legend(loc="best", fontsize=14)
    ax.set_box_aspect(1.0)
    ax.margins(x=0)

    if save_fig:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'temp')
        os.makedirs(output_dir, exist_ok=True)
        filename = f'v_h_dot_vs_{param_name}.png'
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Plot saved to {save_path}")
    else:
        plt.show(block=True)


def plot_v_h_dot_2d(x_vals, y_vals, mean_dvdt, mean_dhdt, param_x_name, param_y_name, save_fig=False):
    """
    Plots 2D heatmaps for mean dV/dt and dH/dt from a parameter sweep.
    Generates and saves two separate figures.

    Args:
        x_vals (np.ndarray): Array of values for the parameter on the y-axis.
        y_vals (np.ndarray): Array of values for the parameter on the x-axis.
        mean_dvdt (np.ndarray): 2D array of mean dV/dt values.
        mean_dhdt (np.ndarray): 2D array of mean dH/dt values.
        param_x_name (str): Name of the parameter for the y-axis.
        param_y_name (str): Name of the parameter for the x-axis.
        save_fig (bool): If True, saves the plots to files. Otherwise, shows them interactively.
    """
    Xg, Yg = np.meshgrid(y_vals, x_vals)
    greek_map = {'ge': r'$g_e$', 'rho': r'$\rho$', 'k': r'$k$', 'm': r'$m$'}
    xlabel = greek_map.get(param_y_name.lower(), param_y_name)
    ylabel = greek_map.get(param_x_name.lower(), param_x_name)

    def five_ticks(grid):
        mn, mx = np.nanmin(grid), np.nanmax(grid)
        return np.round(np.linspace(mn, mx, 5), 2)

    # --- Plot 1: dV/dt ---
    fig_dv, ax_dv = plt.subplots(constrained_layout=True)
    pcm1 = ax_dv.pcolormesh(Xg, Yg, mean_dvdt, shading="auto", cmap='viridis')
    ax_dv.set_xlabel(xlabel, fontsize=14)
    ax_dv.set_ylabel(ylabel, fontsize=14)
    ax_dv.set_title(r"Mean $\langle dV/dt \rangle$ (post-transient)", fontsize=16)
    ax_dv.set_xticks(five_ticks(y_vals))
    ax_dv.set_yticks(five_ticks(x_vals))
    cbar1 = fig_dv.colorbar(pcm1, ax=ax_dv)
    cbar1.set_label(r'$\langle dV/dt \rangle$', fontsize=14)

    # --- Plot 2: dH/dt ---
    fig_dh, ax_dh = plt.subplots(constrained_layout=True)
    pcm2 = ax_dh.pcolormesh(Xg, Yg, mean_dhdt, shading="auto", cmap='inferno')
    ax_dh.set_xlabel(xlabel, fontsize=14)
    ax_dh.set_ylabel(ylabel, fontsize=14)
    ax_dh.set_title(r"Mean $\langle dH/dt \rangle$ (post-transient)", fontsize=16)
    ax_dh.set_xticks(five_ticks(y_vals))
    ax_dh.set_yticks(five_ticks(x_vals))
    cbar2 = fig_dh.colorbar(pcm2, ax=ax_dh)
    cbar2.set_label(r'$\langle dH/dt \rangle$', fontsize=14)

    if save_fig:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'temp')
        os.makedirs(output_dir, exist_ok=True)

        dv_filename = f"v_dot_heatmap_{param_x_name}_vs_{param_y_name}.png"
        dh_filename = f"h_dot_heatmap_{param_x_name}_vs_{param_y_name}.png"

        dv_save_path = os.path.join(output_dir, dv_filename)
        dh_save_path = os.path.join(output_dir, dh_filename)

        fig_dv.savefig(dv_save_path, dpi=300)
        fig_dh.savefig(dh_save_path, dpi=300)
        print(f"Plots saved to {dv_save_path} and {dh_save_path}")
        plt.close(fig_dv)
        plt.close(fig_dh)
    else:
        plt.show(block=True)