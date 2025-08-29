# Error_System.py

import numpy as np
import jax, jax.numpy as jnp
import diffrax as dfx
jax.config.update("jax_enable_x64", True)
from src.hr_model.model import HindmarshRose as HR
from src.hr_model.model import DEFAULT_PARAMS

class HRNetworkErrorSystem:
    """
    Simulates a system combining a 2‑neuron Hindmarsh‑Rose network
    (using the HindmarshRose class) and an associated error system.
    """
    NUM_ERROR_VARS = 5

    def __init__(self, params=None, dynamics=None, hr_initial_state=None, I_ext=None, hr_xi=None):
        self.dynamics = dynamics
        self.params = params.copy()
        self.hr_initial_state = jnp.array(hr_initial_state, dtype=jnp.float64)
        self.error_initial_state = (self.hr_initial_state[len(self.hr_initial_state) // 2:]
                                    - self.hr_initial_state[:len(self.hr_initial_state) // 2])
        self.xi = jnp.array(hr_xi, dtype=jnp.float64)
        self.I_ext = jnp.array(I_ext, dtype=jnp.float64)

        # --- Create the HindmarshRose instance ---
        self.hr_network = HR(
            N=2,
            params=self.params,
            initial_state=self.hr_initial_state,
            I_ext=self.I_ext,
            xi=self.xi
        )

        # --- Combine initial states for the overall system ---
        self.combined_state0 = jnp.concatenate([self.hr_initial_state, self.error_initial_state], dtype=jnp.float64)

        # Attributes to store results
        self.t = None
        self.solution = None
        self.derivative = None
        self.failed = None

    @staticmethod
    def complete_vector_field(current_error_state, current_hr_state, params):
        """
        Calculates derivatives for the 'complete' error system dynamics.
        This static method can be called from anywhere without needing an instance.
        """
        # Unpack HR state variables needed by the error system
        x1, _, _, u1, phi1 = current_hr_state[0:5]
        _, _, _, u2, _ = current_hr_state[5:10]

        # Unpack current error state
        e_x, e_y, e_z, e_u, e_phi = current_error_state

        # --- Complete Error System Dynamics Derivatives ---
        de_xdt = ((((e_y - (params['a'] * ((e_x ** 3) + (3 * (e_x ** 2) * x1) + (3 * e_x * (x1 ** 2))))
                     + (params['b'] * ((e_x ** 2) + (2 * e_x * x1))) + (params['k'] * params['h'] * e_x))
                    + (params['k'] * params['f'] * ((x1 * (e_u ** 2)) + (2 * u1 * x1 * e_u) + (e_x * (e_u ** 2)) +
                                                    (2 * u1 * e_x * e_u) + ((u1 ** 2) * e_x))))
                   + (params['rho'] * ((phi1 * e_x) + (e_x * e_phi) + (x1 * e_phi))) - (2 * params['ge'] * e_x))
        )
        de_ydt = -params['d'] * ((e_x ** 2) + (2 * e_x * x1)) - e_y
        de_zdt = params['r'] * ((params['s'] * e_x) - e_z)

        # --- piece-wise de_u/dt ---
        condlist = [
            (u1 >= 1) & (-1 < u2) & (u2 < 1), (u1 >= 1) & (u2 <= -1),
            (-1 < u1) & (u1 < 1) & (u2 >= 1), (-1 < u1) & (u1 < 1) & (-1 < u2) & (u2 < 1),
            (-1 < u1) & (u1 < 1) & (u2 <= -1), (u1 <= -1) & (u2 >= 1),
            (u1 <= -1) & (-1 < u2) & (u2 < 1),
        ]
        choicelist = [
            e_x + (2 * params['m'] - 1) * e_u + 2 * params['m'] * (u1 - 1), e_x - e_u - 4 * params['m'],
            e_x - e_u - 2 * params['m'] * (u1 - 1), e_x + (2 * params['m'] - 1) * e_u,
            e_x - e_u - 2 * params['m'] * (u1 + 1), e_x - e_u + 4 * params['m'],
            e_x + (2 * params['m'] - 1) * e_u + 2 * params['m'] * (u1 + 1),
        ]
        de_udt = jnp.select(condlist, choicelist, default=e_x - e_u)
        de_phidt = e_x - params['q'] * e_phi

        return jnp.array([de_xdt, de_ydt, de_zdt, de_udt, de_phidt], dtype=jnp.float64)

    @staticmethod
    def simplified_vector_field(current_error_state, current_hr_state, params):
        """
        Calculates derivatives for the 'simplified' error system dynamics.
        This static method can be called from anywhere without needing an instance.
        """
        # Unpack HR state variables needed by the error system
        x1, _, _, u1, phi1 = current_hr_state[0:5]
        _, _, _, u2, _ = current_hr_state[5:10]

        # Unpack current error state
        e_x, e_y, e_z, e_u, e_phi = current_error_state

        # --- Simplified Error System Dynamics Derivatives ---
        de_xdt = (
                (((-3 * params['a'] * (x1 ** 2)) + (2 * params['b'] * x1) + (params['k'] * params['h'])
                  + (params['k'] * params['f'] * (u1 ** 2)) + (params['rho'] * phi1) - (2 * params['ge'])) * e_x)
                + e_y + (2 * params['k'] * params['f'] * u1 * x1 * e_u) + (params['rho'] * x1 * e_phi)
        )
        de_ydt = (-2 * params['d'] * x1 * e_x) - e_y
        de_zdt = params['r'] * ((params['s'] * e_x) - e_z)

        # --- piece-wise de_u/dt ---
        condlist = [
            (u1 >= 1) & (-1 < u2) & (u2 < 1), (u1 >= 1) & (u2 <= -1),
            (-1 < u1) & (u1 < 1) & (u2 >= 1), (-1 < u1) & (u1 < 1) & (-1 < u2) & (u2 < 1),
            (-1 < u1) & (u1 < 1) & (u2 <= -1), (u1 <= -1) & (u2 >= 1),
            (u1 <= -1) & (-1 < u2) & (u2 < 1),
        ]
        choicelist = [
            e_x + (2 * params['m'] - 1) * e_u + 2 * params['m'] * (u1 - 1), e_x - e_u - 4 * params['m'],
            e_x - e_u - 2 * params['m'] * (u1 - 1), e_x + (2 * params['m'] - 1) * e_u,
            e_x - e_u - 2 * params['m'] * (u1 + 1), e_x - e_u + 4 * params['m'],
            e_x + (2 * params['m'] - 1) * e_u + 2 * params['m'] * (u1 + 1),
        ]
        de_udt = jnp.select(condlist, choicelist, default=e_x - e_u)
        de_phidt = e_x - (params['q'] * e_phi)

        return jnp.array([de_xdt, de_ydt, de_zdt, de_udt, de_phidt], dtype=jnp.float64)

    def _error_system_ode(self, t, current_error_state, current_hr_state):
        """Calculates derivatives ONLY for the error system variables by calling the appropriate static method."""
        if self.dynamics == "simplified":
            return HRNetworkErrorSystem.simplified_vector_field(current_error_state, current_hr_state, self.params)
        elif self.dynamics == "complete":
            return HRNetworkErrorSystem.complete_vector_field(current_error_state, current_hr_state, self.params)
        else:
            raise ValueError(f"Unknown dynamics mode: {self.dynamics}")

        # In your HRNetworkErrorSystem class...

    def _clipped_ode_func(self, t, combined_state, args):
        """
        Combined HR-network + error-system ODE for SIMPLIFIED dynamics.
        The ERROR variables are box-constrained to self.error_clip_bound.
        """
        # Add near the top of your class for clarity
        ERROR_SLICE = slice(10, 15)  # indices of e_x … e_phi
        LOW, HIGH = [-5, 5] # box limits

        # ── 1. Unpack raw (unclipped) state ─────────────────────────────────
        hr_state = combined_state[:10]  # x1…phi2
        err_state = combined_state[ERROR_SLICE]  # e_x…e_phi

        # ── 2. Compute raw derivatives ─────────────────────────────────────
        hr_d = self.hr_network._ode_func_internal(t, hr_state)
        err_d = self._error_system_ode(t, err_state, hr_state)
        combined_derivatives = jnp.concatenate([hr_d, err_d])

        # ── 3. Velocity-clamp ONLY the error-state derivatives ─────────────
        err_d_clamped = jnp.where(
            ((err_state >= HIGH) & (err_d > 0.0)) |  # trying to exit above +5
            ((err_state <= LOW) & (err_d < 0.0)),  # trying to exit below −5
            0.0,  # stop outward motion
            err_d  # keep everything else
        )

        final_derivatives = combined_derivatives.at[ERROR_SLICE].set(err_d_clamped)
        return final_derivatives

    def _unclipped_ode_func(self, t, combined_state, args):
        """
        Combined HR-network + error-system ODE for COMPLETE dynamics (no clipping).
        """
        # --- Unpack the combined state vector ---
        current_hr_state = combined_state[0:10]
        current_error_state = combined_state[10:15]

        # --- Calculate HR Derivatives using the HindmarshRose instance ---
        hr_derivatives_flat = self.hr_network._ode_func_internal(t, current_hr_state)

        # --- Calculate Error System Dynamics Derivatives ---
        error_derivatives = self._error_system_ode(t, current_error_state, current_hr_state)

        # --- Combine derivatives into a single vector ---
        combined_derivatives = jnp.concatenate([hr_derivatives_flat, error_derivatives])
        return combined_derivatives

    # ------------------------------------------------------------------
    # Diffrax‑based solver
    # ------------------------------------------------------------------
    def solve(self, solver=None, t0=None, t1=None, dt0=None, n_points=None,
              stepsize_controller=None, max_steps=None):
        """
        Integrate the combined HR + error system with Diffrax.
        """
        # --- Choose the correct vector field function BEFORE calling the solver ---
        if self.dynamics == "simplified":
            vector_field = self._clipped_ode_func
        elif self.dynamics == "complete":
            vector_field = self._unclipped_ode_func
        else:
            # This case should not be reached with the current code
            raise ValueError(f"Unknown dynamics mode: {self.dynamics}")

        try:
            sol = dfx.diffeqsolve(
                terms=dfx.ODETerm(vector_field),  # Use the selected function
                solver=solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=self.combined_state0,
                saveat=n_points,
                stepsize_controller=stepsize_controller,
                max_steps=max_steps,
                args=None
            )
        except Exception as exc:
            print(f"Solver failed with exception: {exc}")
            self.failed = True
            self.t = self.solution = None
            return np.nan, np.nan

        # Normal exit ------------------------------------------------------
        self.failed = False
        self.t = jnp.asarray(sol.ts)
        self.solution = jnp.asarray(sol.ys)
        # Recompute derivative using the selected vector field
        self.derivative = jnp.asarray(jax.vmap(vector_field, in_axes=(0, 0, None))(sol.ts, sol.ys, None))

    # ------------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------------
    def get_results_dict(self, transient_ratio: float = 0.0):
        """
        Return a dict containing the post-transient time vector ('t') and all
        state variables of the combined HR + error system.
        """
        # ── locate cut-off index using the actual (non-uniform) time stamps ─
        cutoff_time = self.t[0] + transient_ratio * (self.t[-1] - self.t[0])
        start_idx = jnp.searchsorted(self.t, cutoff_time, side="left")

        t_post   = self.t[start_idx:]
        sol_post = self.solution[start_idx:]
        deriv_post = self.derivative[start_idx:]

        var_names = [
            'x1', 'y1', 'z1', 'u1', 'phi1',
            'x2', 'y2', 'z2', 'u2', 'phi2',
            'e_x', 'e_y', 'e_z', 'e_u', 'e_phi'
        ]

        deriv_names = [f'd_{name}' for name in var_names]

        result = {'t': t_post}
        result.update({name: sol_post[:, i] for i, name in enumerate(var_names)})
        result.update({name: deriv_post[:, i] for i, name in enumerate(deriv_names)}) # Use pre-saved derivatives
        return result


# --- Example Usage ---
if __name__ == '__main__':
    from visualization.plotting import plot_error_and_state_differences

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

    # Plotting
    plot_error_and_state_differences(results)

    # # # Write to file
    # # import pickle
    # # with open('error_system.pkl', 'wb') as f:
    # #     pickle.dump(results, f)