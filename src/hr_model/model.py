import numpy as np
import jax, jax.numpy as jnp
import diffrax as dfx
jax.config.update("jax_enable_x64", True)

# ──────────────────────────────────────────────────────────────────────────────
# Default parameters dictionary
DEFAULT_PARAMS = {
    # --- Neuron Params ---
    'a': 1.0,
    'b': 3.0,
    'c': 1.0,
    'd': 5.0,
    'f': 0.2,
    'h': 0.3,
    'k': 0.87,
    'm': 0.5,
    'q': 0.005,
    'r': 0.006,
    's': 5.2,
    'x0': -1.56,
    'rho': 0.7,
    'ge': 0.7,
}

# Default initial state for a single neuron
DEFAULT_STATE0 = [0.0, 0.0, 0.0, 1.0, 0.0]

class HindmarshRose:
    """Simulates a network of N coupled Hindmarsh-Rose neurons."""

    def __init__(self, N, params=None, initial_state=None, I_ext=None, xi=None):
        self.N = N
        self.params = params.copy()
        self.initial_state = jnp.array(initial_state, dtype=jnp.float64).flatten()

        # External current
        if isinstance(I_ext, (int, float)):
            self.I_ext = jnp.full(self.N, float(I_ext), dtype=jnp.float64)
        else:
            self.I_ext = jnp.array(I_ext, dtype=jnp.float64)

        # Electrical coupling matrix
        if isinstance(xi, (int, float)):
            self.xi = jnp.full((self.N, self.N), float(xi), dtype=jnp.float64)
        else:
            self.xi = jnp.array(xi, dtype=jnp.float64)
        if self.N > 0:
            self.xi = jnp.fill_diagonal(self.xi, 0, inplace=False)

        # Attributes to store results later
        self.t = None
        self.solution = None
        self.failed = None

    @staticmethod
    def vector_field(t, state, N, params, I_ext, xi):
        """
        Calculates the time derivatives for a network of N coupled Hindmarsh-Rose neurons.
        This static method contains the core physics and can be called from anywhere
        without needing an instance of the class, e.g., HindmarshRose.vector_field(...)
        """
        # Reshape the flat state vector into a 2D array (N neurons x 5 variables)
        state_matrix = state.reshape((N, 5))
        x, y, z, u, phi = state_matrix.T

        # Electrical Coupling
        x_diff = x[jnp.newaxis, :] - x[:, jnp.newaxis]
        electrical_coupling = params['ge'] * jnp.sum(jnp.asarray(xi) * x_diff, axis=1)

        # --- Calculate derivatives ---
        dxdt = (y - (params['a'] * x ** 3) + (params['b'] * x ** 2)
                + (params['k'] * (params['h'] + (params['f'] * (u ** 2))) * x)
                + (params['rho'] * phi * x) + I_ext
                + electrical_coupling
                )
        dydt = params['c'] - (params['d'] * x ** 2) - y
        dzdt = params['r'] * (params['s'] * (x + params['x0']) - z)
        dudt = -u + (params['m'] * (jnp.abs(u + 1.0) - jnp.abs(u - 1.0))) + x
        dphidt = x - (params['q'] * phi)

        # Assign calculated derivative vectors to the output matrix
        d_state_dt_matrix = jnp.zeros_like(state_matrix).at[:, 0].set(dxdt).at[:, 1].set(dydt) \
                                .at[:, 2].set(dzdt).at[:, 3].set(dudt) \
                                .at[:, 4].set(dphidt)

        return d_state_dt_matrix.flatten()

    def _ode_func_internal(self, t, state):
        # This instance method now calls the static method, passing its instance attributes.
        return HindmarshRose.vector_field(t, state, self.N, self.params, self.I_ext, self.xi)

    # ────────────────────────────────────────────────────────────────────────
    def solve(self, solver=None, t0=None, t1=None, dt0=None, n_points=None,
              stepsize_controller=None, max_steps=None):
        """Integrate the model with Diffrax."""

        try:
            sol = dfx.diffeqsolve(
                terms=dfx.ODETerm(lambda t, y, args: self._ode_func_internal(t, y)),
                solver=solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=self.initial_state,
                saveat=dfx.SaveAt(ts=jnp.linspace(t0, t1, n_points)),
                stepsize_controller=stepsize_controller,
                max_steps=max_steps
            )
        except Exception as exc:
            print(f"Solver failed with exception: {exc}")
            self.failed = True
            self.t = self.solution = None
            return np.nan, np.nan

        self.t = jnp.asarray(sol.ts)
        self.solution = jnp.asarray(sol.ys)
        self.failed = False
        self.derivative = jnp.asarray(jax.vmap(self._ode_func_internal)(sol.ts, sol.ys))

    # ────────────────────────────────────────────────────────────────────────
    def get_results_dict(self, transient_ratio: float = 0.0):
        """
        Return the simulated time series as a dict *after* removing the initial
        transient period.
        """
        # <<< 4. MODIFIED: Entire method updated to handle derivatives
        if self.t is None:
            print("Simulation has not been run or has failed. Returning empty dictionary.")
            return {}

        # ── locate cut-off index using the actual time stamps ─
        cutoff_time = self.t[0] + transient_ratio * (self.t[-1] - self.t[0])
        start_idx = jnp.searchsorted(self.t, cutoff_time, side="left")

        # --- Slice all result arrays to remove transient ---
        t_post = self.t[start_idx:]
        sol_post = self.solution[start_idx:]
        deriv_post = self.derivative[start_idx:]

        # --- Create names for state variables and their derivatives ---
        var_names = [f"{v}{i+1}" for i in range(self.N) for v in ("x", "y", "z", "u", "phi")]
        deriv_names = [f"d_{name}" for name in var_names]

        # ── assemble and return dict ────────────────────────────────────
        result = {'t': t_post}
        result.update({name: sol_post[:, i] for i, name in enumerate(var_names)})
        result.update({name: deriv_post[:, i] for i, name in enumerate(deriv_names)})

        return result









# --- Example Usage ---
if __name__ == '__main__':
    from visualization.plotting import plot_all_time_series
    import os

    #=======================================================================================
    # --- Example 1: Single Neuron ---
    #=======================================================================================

    print("Simulating Single Neuron...")
    # Initialize using defaults where possible
    sim_params = DEFAULT_PARAMS.copy()
    sim_params['ge'] = 0.2

    # create the model
    hr_single = HindmarshRose(N=1, params=sim_params, initial_state=DEFAULT_STATE0, I_ext=0.8, xi=0)

    # integration settings
    start_time = 0
    end_time = 1000
    dt_initial = 0.05
    n_points = 10000
    transient_ratio = 0.5
    max_steps  = int((end_time - start_time) / dt_initial) * 20

    solver = dfx.Tsit5()
    stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-10)
    # stepsize_controller = dfx.ConstantStepSize()

    # Run the solver
    import time
    start = time.perf_counter()

    hr_single.solve(
        solver=solver, t0=start_time, t1=end_time, dt0=dt_initial, n_points=n_points,
        stepsize_controller=stepsize_controller, max_steps=max_steps)

    finish = time.perf_counter()
    time = finish - start
    print(time)

    results_single = hr_single.get_results_dict(transient_ratio)

    plot_all_time_series(results_single, N=1, title="One Neuron Simulation", save_fig=0)

    #=======================================================================================
    # --- Example 2: Two Coupled Neurons ---
    #=======================================================================================

    print("\nSimulating Two Coupled Neurons...")

    # Initialize using defaults where possible
    sim_params = DEFAULT_PARAMS.copy()
    sim_params['ge'] = 0.2

    # initial state (x, y, z, u, φ for each neuron)
    state0_coupled = [
        0.1, 0.2, 0.3, 0.4, 0.1,   # neuron 1
        0.2, 0.3, 0.4, 0.5, 0.2    # neuron 2
    ]

    # external currents and coupling matrix
    I_ext_coupled = [0.8, 0.8]
    xi_coupled = [[0, 1], [1, 0]]

    # create the model
    hr_coupled = HindmarshRose(
        N=2,
        params=sim_params,
        initial_state=state0_coupled,
        I_ext=I_ext_coupled,
        xi=xi_coupled
    )

    # integration settings
    start_time = 0
    end_time   = 1000
    dt_initial = 0.05
    n_points   = 10000
    transient_ratio = 0.5
    max_steps  = int((end_time - start_time) / dt_initial) * 20

    solver = dfx.Tsit5()
    stepsize_controller = dfx.PIDController(rtol=1e-10, atol=1e-12)
    # stepsize_controller = dfx.ConstantStepSize()

    # run the solver
    import time
    start = time.perf_counter()
    hr_coupled.solve(
        solver=solver,
        t0=start_time,
        t1=end_time,
        dt0=dt_initial,
        n_points=n_points,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps)
    finish = time.perf_counter()
    time = finish - start
    print(time)

    results_coupled = hr_coupled.get_results_dict(transient_ratio)

    plot_all_time_series(results_coupled, N=2, title="Two Coupled Neurons Simulation", save_fig=False)