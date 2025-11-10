# An asymptotic stability proof and a port-Hamiltonian PINN for chaotic synchronization in Hindmarsh–Rose neurons

**Codebase for the paper**  
**“An asymptotic stability proof and a port-Hamiltonian physics-informed neural network approach to chaotic synchronization in Hindmarsh–Rose neurons.”**

---

## Abstract

We study chaotic synchronization in a five-dimensional Hindmarsh-Rose neuron model augmented with electromagnetic induction and a switchable memristive autapse. For two diffusively coupled neurons, we derive the linearized error dynamics and prove global asymptotic stability of the synchronization manifold via a quadratic Lyapunov function. Verifiable sufficient conditions follow from Sylvester’s criterion on the leading principal minors, and convergence is established using Barbalat’s lemma. Leveraging Helmholtz's decomposition, we separate the error field into conservative and dissipative parts and obtain a closed-form expression for the synchronization energy, along with its dissipation law, providing a quantitative measure of the energetic cost of synchrony. Numerical simulations confirm complete synchronization, overall decay of the synchronization energy, and close agreement between Lyapunov and Hamiltonian diagnostics across parameter sweeps. Building on these results, we introduce a port-Hamiltonian physics-informed neural network that embeds the conservative/dissipative structure in training through physically motivated losses and structural priors. The learned Hamiltonian and energy–rate match analytical benchmarks.

---

## Contents
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Key directories](#key-directories)
- [Typical pH-PINN workflow](#typical-ph-pinn-workflow)
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

## Citing

If you use this repository, please cite:

```bibtex
@article{BabaeianYamakou2025HRpHPINN,
  title   = {An asymptotic stability proof and a port-Hamiltonian physics-informed neural network approach to chaotic synchronization in Hindmarsh-Rose neurons},
  author  = {Babaeian, Behnam and Yamakou, Marius E.},
  year    = {2025}
}
```

Preprint: https://arxiv.org/abs/2511.04809
