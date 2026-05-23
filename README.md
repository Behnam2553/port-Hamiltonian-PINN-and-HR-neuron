# A Lyapunov stability proof and a port-Hamiltonian physics-informed neural network for chaotic synchronization in memristive neurons

**Codebase for the paper**  
**“A Lyapunov stability proof and a port-Hamiltonian physics-informed neural network for chaotic synchronization in memristive neurons.”**

---

## Abstract

We study chaotic synchronization in a 5D Hindmarsh--Rose neuron model augmented with electromagnetic induction and a switchable memristive autapse. For two diffusively coupled identical neurons, we derive the transverse error dynamical system and analyze local synchronization via the linearized error system around the synchronization manifold. A quadratic Lyapunov function yields explicit sufficient conditions for (i) asymptotic stability when the memristive switching remains dissipative and (ii) practical stability with an explicit ultimate bound under non-dissipative switching. We complement this with a Hamiltonian-based viewpoint: a Helmholtz decomposition of the linearized error vector field provides a closed-form synchronization Hamiltonian and its rate identity. Numerical simulations corroborate convergence or ultimate boundedness of the synchronization errors and an overall decay of the synchronization Hamiltonian and its instantaneous rate toward zero after transients, and show consistent trends between Lyapunov- and Hamiltonian-based diagnostics across parameters. Finally, we propose the first port-Hamiltonian physics-informed neural network (pH-PINN) that learns this synchronization Hamiltonian and its rate from data while preserving conservative/dissipative structure, achieving close agreement with the analytical expressions.

---

## Contents
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Key directories](#key-directories)
- [Typical pH-PINN workflow](#typical-ph-pinn-workflow)
- [pH-PINN Diagram](#pH-PINN-diagram)
- [Citing](#citing)

---

## Repository structure

```text
├── src/
│   └── hr_model/
│       ├── model.py
│       └── error_system.py
│   └── ph_pinn/
│       ├── optimize_hyperparams.py
│       ├── pH_PINN.py
│       └── read_best_hyperparams.py
├── laboratory/
│   ├── bifurcation_analysis.py
│   ├── generate_data_for_pH_PINN.py
│   ├── synchronization_quantities.py
│   ├── v_dot_parameter_sweep_1D.py
│   ├── v_dot_parameter_sweep_2D.py
│   └── run_loop.py
├── visualization/
│   └── plotting.py
├── Fortran Codes/                 # Lyapunov calculations for the 1D and 2D maps
├── results/
├── pH-PINN diagram.png
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
pip install -r requirements.txt
```

That is enough to run the dynamical-system scripts in `laboratory/` and the models in `src/`.

---

## Key directories

### `src/hr_model/`
- `model.py`: the 5D Hindmarsh–Rose neuron with electromagnetic and memristive extensions (the system described in the paper).
- `error_system.py`: coupled 2-neuron error dynamics + solver; this is what the stability proof and the energy formulas refer to.

### `src/ph_pinn/`
- `pH_PINN.py`: implementation of the port-Hamiltonian physics-informed neural network.
- `optimize_hyperparams.py`: helper to search hyperparameters.
- `read_best_hyperparams.py`: loads the best set and runs the model.

This part **does not** simulate the neurons; it assumes data already exist.

### `laboratory/`
Analysis / experiment scripts around the model:
- `bifurcation_analysis.py`: 1D parameter sweep and bifurcation plot.
- `synchronization_quantities.py`: formulas for \(H\), \(\dot H\), \(\dot V\).
- `v_dot_parameter_sweep_1D.py` & `v_dot_parameter_sweep_2D.py`: stability/energy maps over parameters.
- `generate_data_for_pH_PINN.py`: **creates the dataset** that the pH-PINN will train on.
- `run_loop.py`: helper for long 2D sweeps.

### `visualization/`
- `plotting.py`: common plotting utilities used by the lab scripts. Outputs go to `results/`.

---

## Typical pH-PINN workflow

1. **Generate data from the dynamical system**  
   ```bash
   python laboratory/generate_data_for_pH_PINN.py
   ```
   This integrates the error system, computes \(H\), \(\dot H\), \(\dot V\), and saves everything under `results/`.

2. **Train / run the pH-PINN**  
   ```bash
   python src/ph_pinn/pH_PINN.py
   ```
   or, if you want to tune it first:
   ```bash
   python src/ph_pinn/optimize_hyperparams.py
   python src/ph_pinn/read_best_hyperparams.py
   ```

3. **Inspect** the learned Hamiltonian and energy-rate and compare with the analytical ones (the saved data contain them).

---
## pH-PINN Diagram

<img width="5890" height="5460" alt="Figure_9" src="https://github.com/user-attachments/assets/0484fdc0-ca76-4f67-8ac6-bc2373399c0a" />


---

## Citing

If you use this repository, please cite:

```bibtex
\bibitem{BabaeianYamakou2026}
B.~Babaeian and M.~E. Yamakou,
``A Lyapunov stability proof and a port-Hamiltonian physics-informed neural network for chaotic synchronization in memristive neurons,''
\emph{Applied Mathematics and Computation},
vol.~530, Article~130157, 2026.
doi: \href{https://doi.org/10.1016/j.amc.2026.130157}{10.1016/j.amc.2026.130157}.
```
paper: https://doi.org/10.1016/j.amc.2026.130157
