import optuna
import os


def main():
    """
    Loads an Optuna study from a SQLite database file and prints the
    details of the best trial.
    """
    # --- Define Study Details ---
    # These must match the values used in your optimize_hyperparams.py script.

    # MODIFICATION: Correctly navigate from src/sph_pinn up to the project root ('..', '..')
    # and then down to the results directory.
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data')

    db_name = 'optimize_hyperparams.db'
    db_path = os.path.join(results_dir, db_name)

    storage_name = f"sqlite:///{db_path}"
    study_name = "sphnn_pinn_optimization_study"

    # --- Load the Study ---
    print(f"Loading study '{study_name}' from {storage_name}...")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except Exception as e:
        print(f"\nError loading study: {e}")
        print(f"Please ensure the database file exists at: {db_path}")
        return

    # --- Get the Best Trial and Print Results ---
    best = study.best_trial

    print("\n" + "=" * 40)
    print("         Best Trial Found")
    print("=" * 40)
    print(f"  Trial Number: {best.number}")
    print(f"  Best Value (Loss): {best.value:.6f}")
    print("\n  Best Hyperparameters:")
    for key, value in best.params.items():
        print(f"    '{key}': {value},")
    print("=" * 40)


if __name__ == "__main__":
    main()