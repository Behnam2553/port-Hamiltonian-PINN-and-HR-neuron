import numpy as np
import jax, jax.numpy as jnp
import diffrax as dfx
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Hamiltonian --------------------------------------------------
# ---------------------------------------------------------------------------
def calculate_H(
    results: dict[str, np.ndarray],
    params:  dict[str, float],
    C: float = 0.0,           # integration constant (defaults to 0)
) -> np.ndarray:
    """
    Return the time series of the Hamiltonian **H** evaluated along a trajectory.

    Parameters
    ----------
    results
        Output of ``Error_System.HRNetworkErrorSystem.get_results_dict``.
        Must contain the keys
        ``'x1', 'u1', 'e_x', 'e_y', 'e_z', 'e_u', 'e_phi'``.
    params
        Parameter dictionary (e.g. ``NeuralModel.DEFAULT_PARAMS``) with at least
        ``a, b, d, f, h, k, m, q, r, rho, s, ge``.
        Only ``d, k, f, r, rho, s`` are used here.
    C
        Arbitrary additive constant.  Leave at 0 unless you need a specific
        reference level.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``len(results['x1'])`` – the Hamiltonian value at
        each recorded time step.
    """
    # ------------------------------------------------------------------
    # 1. Unpack parameters we actually need
    # ------------------------------------------------------------------
    d   = params["d"]
    k   = params["k"]
    f   = params["f"]
    r   = params["r"]
    rho = params["rho"]
    s   = params["s"]

    # ------------------------------------------------------------------
    # 2. Extract trajectory arrays
    # ------------------------------------------------------------------
    required = ("x1", "u1", "e_x", "e_y", "e_z", "e_u", "e_phi")
    try:
        x1, u1, e_x, e_y, e_z, e_u, e_phi = (
            jnp.asarray(results[k]) for k in required
        )
    except KeyError as missing:
        raise KeyError(f"results dict is missing key: {missing}") from None

    # Shape validation (same pattern as in calculate_dHdt)
    ref_shape = x1.shape
    for key, arr in zip(required, (x1, u1, e_x, e_y, e_z, e_u, e_phi)):
        if arr.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch for results['{key}'] – expected {ref_shape}, got {arr.shape}"
            )

    # ------------------------------------------------------------------
    # 3. Hamiltonian formula (vectorised)
    # ------------------------------------------------------------------
    H = (
        2 * d * x1 * e_x**2
        + e_y**2
        - 4 * d * k * f * u1 * x1**2 * e_u**2
        - 2 * d * rho * x1**2 * e_phi**2
        + e_z
        - r * s * e_phi
        + C
    )

    # Guard against numerical overflow/underflow
    H = jnp.where(jnp.isfinite(H), H, jnp.nan)
    return H

# ---------------------------------------------------------------------------
# Hamiltonian Rate of Change---------------------------------------
# ---------------------------------------------------------------------------
def calculate_dHdt(results: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:  # noqa: N802 – keep MATLAB-style name
    """Return the time series of **dH/dt** evaluated along a trajectory.

    Parameters
    ----------
    results
        Dictionary produced by :pymeth:`Error_System.HRNetworkErrorSystem.get_results_dict`.
        Must contain the keys ``'x1', 'u1', 'phi1', 'u2', 'e_x', 'e_y', 'e_z', 'e_u', 'e_phi'``.
    params
        Parameter dictionary (e.g. :pydata:`NeuralModel.DEFAULT_PARAMS`).
        Must include at least the entries ::

            a, b, d, f, h, k, m, q, r, rho, s, ge

    Returns
    -------
    numpy.ndarray
        1-D array with ``len(results['x1'])`` elements – the value of dH/dt at
        each recorded time step.
    """
    # ----------------------------------------------------------------------
    # 1. Unpack parameters
    # ----------------------------------------------------------------------
    a   = params["a"]
    b   = params["b"]
    d   = params["d"]
    f   = params["f"]
    h   = params["h"]
    k   = params["k"]
    m   = params["m"]
    q   = params["q"]
    r   = params["r"]
    rho = params["rho"]
    s   = params["s"]
    ge  = params["ge"]

    # ----------------------------------------------------------------------
    # 2. Extract trajectory arrays
    # ----------------------------------------------------------------------
    required = ("x1", "u1", "phi1", "u2", "e_x", "e_y", "e_z", "e_u", "e_phi")
    try:
        x1, u1, phi1, u2, e_x, e_y, e_z, e_u, e_phi = (
            jnp.asarray(results[k]) for k in required
        )
    except KeyError as missing:
        raise KeyError(f"results dict is missing key: {missing}") from None

    # Shape validation ------------------------------------------------------
    ref_shape = x1.shape
    for key, arr in zip(required, (x1, u1, phi1, u2, e_x, e_y, e_z, e_u, e_phi)):
        if arr.shape != ref_shape:
            raise ValueError(f"Shape mismatch for results['{key}'] – expected {ref_shape}, got {arr.shape}")

    # ----------------------------------------------------------------------
    # 3. Helper functions N, alpha, beta (vectorised)
    # ----------------------------------------------------------------------
    N = (
        -3 * a * x1**2 + 2 * b * x1 + k * h + k * f * u1**2 + rho * phi1 - 2 * ge
    )

    conds = [
        (u1 >= 1) & (u2 > -1) & (u2 < 1),
        (u1 >= 1) & (u2 <= -1),
        (u1 > -1) & (u1 < 1) & (u2 >= 1),
        (u1 > -1) & (u1 < 1) & (u2 > -1) & (u2 < 1),
        (u1 > -1) & (u1 < 1) & (u2 <= -1),
        (u1 <= -1) & (u2 >= 1),
        (u1 <= -1) & (u2 > -1) & (u2 < 1),
    ]

    alpha_choices = [2 * m - 1, -1, -1, 2 * m - 1, -1, -1, 2 * m - 1]
    beta_choices  = [
        2 * m * (u1 - 1),
        -4 * m,
        -2 * m * (u1 - 1),
        0,
        -2 * m * (u1 + 1),
        4 * m,
        2 * m * (u1 + 1),
    ]

    alpha = jnp.select(conds, alpha_choices, default=-1)
    beta  = jnp.select(conds, beta_choices,  default=0)

    # ----------------------------------------------------------------------
    # 4. Assemble dH/dt (Eq. Hdot)
    # ----------------------------------------------------------------------
    dHdt = (
        4 * d * x1 * N * e_x**2
        - 2 * e_y**2
        - r * e_z
        - 8 * d * k * f * u1 * x1**2 * alpha * e_u**2
        - 8 * d * k * f * u1 * x1**2 * beta * e_u
        + 4 * d * rho * q * x1**2 * e_phi**2
        + q * r * s * e_phi
    )

    # Guard against numerical overflow/underflow
    dHdt = jnp.where(jnp.isfinite(dHdt), dHdt, jnp.nan)
    return dHdt


# ---------------------------------------------------------------------------
# Lyapunov Rate of change -----------------------------------------
# ---------------------------------------------------------------------------

def calculate_dVdt(results, params):
    """
    Calculates the time series of dV/dt based on simulation results.

    Args:
        results (dict): Dictionary containing the time series output from
                        HRNetworkErrorSystem.get_results_dict(). Must contain keys
                        'x1', 'u1', 'phi1', 'u2', 'e_x', 'e_y', 'e_z', 'e_u', 'e_phi'.
        params (dict): Dictionary containing the model parameters. Must contain
                       keys 'a', 'b', 'k', 'h', 'f', 'rho', 'ge', 'gc', 'lam',
                       'v_syn', 'theta', 'd', 'r', 's', 'q', 'm'.

    Returns:
        np.ndarray: Time series array of dV/dt values. Returns None if
                    required keys are missing.
    """

    # --- Extract Parameters ---
    a = params['a']
    b = params['b']
    k = params['k']
    h = params['h']
    f = params['f']
    rho = params['rho']
    ge = params['ge']
    d = params['d']
    r = params['r']
    s = params['s']
    q = params['q']
    m = params['m']

    # --- Extract Time Series Variables ---
    x1 = results['x1']
    u1 = results['u1']
    phi1 = results['phi1']
    u2 = results['u2']
    e_x = results['e_x']
    e_y = results['e_y']
    e_z = results['e_z']
    e_u = results['e_u']
    e_phi = results['e_phi']

    # Ensure inputs are numpy arrays
    x1, u1, phi1, u2 = jnp.asarray(x1), jnp.asarray(u1), jnp.asarray(phi1), jnp.asarray(u2)
    e_x, e_y, e_z, e_u, e_phi = jnp.asarray(e_x), jnp.asarray(e_y), jnp.asarray(e_z), jnp.asarray(e_u), jnp.asarray(e_phi)

    # --- Calculate dV/dt Components ---
    dVdt_term1 = (((-3 * a * (x1 ** 2)) + (2 * b * x1) + (k * h) +
                  (k * f * (u1 ** 2)) + (rho * phi1) - (2 * ge)) * (e_x ** 2))
    dVdt_term2 = -(e_y ** 2)
    dVdt_term3 = -(r * (e_z ** 2))
    dVdt_term4 = -(q * (e_phi ** 2))
    dVdt_term5 = 2 * k * f * u1 * x1 * e_x * e_u
    dVdt_term6 = (1 - (2 * d * x1)) * e_x * e_y
    dVdt_term7 = r * s * e_x * e_z
    dVdt_term8 = ((rho * x1) + 1) * e_x * e_phi

    # --- Calculate Piecewise Term ---

    conditions = [
        (u1 >= 1) & (-1 < u2) & (u2 < 1),
        (u1 >= 1) & (u2 <= -1),
        (-1 < u1) & (u1 < 1) & (u2 >= 1),
        (-1 < u1) & (u1 < 1) & (-1 < u2) & (u2 < 1),
        (-1 < u1) & (u1 < 1) & (u2 <= -1),
        (u1 <= -1) & (u2 >= 1),
        (u1 <= -1) & (-1 < u2) & (u2 < 1)
    ]
    choices = [
        (e_x * e_u) + ((2 * m - 1) * (e_u ** 2)) + (2 * m * (u1 - 1) * e_u),
        (e_x * e_u) - (e_u ** 2) - (4 * m * e_u),
        (e_x * e_u) - (e_u ** 2) - (2 * m * (u1 - 1) * e_u),
        (e_x * e_u) + ((2 * m - 1) * (e_u ** 2)),
        (e_x * e_u) - (e_u ** 2) - (2 * m * (u1 + 1) * e_u),
        (e_x * e_u) - (e_u ** 2) + (4 * m * e_u),
        (e_x * e_u) + ((2 * m - 1) * (e_u ** 2)) + (2 * m * (u1 + 1) * e_u)
    ]
    default_choice = (e_x * e_u) - (e_u ** 2)
    piecewise_term = jnp.select(conditions, choices, default=default_choice)


    # --- Combine all terms ---
    dVdt = (dVdt_term1 + dVdt_term2 + dVdt_term3 + dVdt_term4 +
            dVdt_term5 + dVdt_term6 + dVdt_term7 + dVdt_term8 +
            piecewise_term)

    dVdt = jnp.where(jnp.isfinite(dVdt), dVdt, jnp.nan)     # replace inf / nan by nan

    return dVdt


# --- Example Usage ---
if __name__ == '__main__':
    from src.hr_model.error_system import HRNetworkErrorSystem
    from src.hr_model.model import DEFAULT_PARAMS
    from visualization.plotting import (
        plot_hamiltonian,
        plot_hamiltonian_derivative,
        plot_lyapunov_derivative
    )

    # initial state (x, y, z, u, φ for each neuron)
    INITIAL_HR_STATE0 = [
        0.1, 0.2, 0.3, 0.4, 0.1,   # neuron 1
        0.2, 0.3, 0.4, 0.5, 0.2    # neuron 2
    ]

    # external currents and coupling matrix
    I_ext = [0.8, 0.8]
    xi = [[0, 1], [1, 0]]

    # Example modification of parameters
    sim_params = DEFAULT_PARAMS.copy()
    # sim_params['ge'] = 0.65

    # Create simulator instance
    simulator = HRNetworkErrorSystem(params=sim_params, dynamics='complete',
                                     hr_initial_state=INITIAL_HR_STATE0, I_ext=I_ext, hr_xi=xi)

    # integration settings
    start_time = 0
    end_time = 1000
    dt_initial = 0.01
    point_num = 10000
    transient_ratio = 0
    n_points = dfx.SaveAt(ts=jnp.linspace(start_time, end_time, point_num), dense=True)
    max_steps = int((end_time - start_time) / dt_initial) * 20

    solver = dfx.Tsit5()
    stepsize_controller = dfx.PIDController(rtol=1e-10, atol=1e-12)
    # stepsize_controller = dfx.ConstantStepSize()

    # run simulation ----------------------------------------------------
    print("Running simulation...")
    import time
    tic = time.perf_counter()

    simulator.solve(
        solver=solver,
        t0=start_time,
        t1=end_time,
        dt0=dt_initial,
        n_points=n_points,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps
    )

    toc = time.perf_counter()
    print(f"Finished in {(toc - tic):.2f} s")

    # Get results dictionary
    results = simulator.get_results_dict(transient_ratio)

    # --- Calculate H, dH/dt, and dV/dt ---
    print("Calculating physical quantities...")
    H = calculate_H(results, sim_params)
    dHdt = calculate_dHdt(results, sim_params)
    dVdt = calculate_dVdt(results, sim_params)
    print("Calculations complete.")

    # --- Plotting ---
    print("Generating plots...")

    # --- Plotting ---
    print("Generating plots...")

    plot_hamiltonian(results['t'], H, save_fig=1)
    plot_hamiltonian_derivative(results['t'], dHdt, save_fig=1)
    plot_lyapunov_derivative(results['t'], dVdt,  save_fig=1)