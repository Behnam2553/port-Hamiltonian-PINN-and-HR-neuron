import optuna
import os
import argparse


def main():
    """
    Loads a specified Optuna study from a SQLite database file, prints the
    details of the best trial, and saves them to a text file.
    """
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Load an Optuna study and print the best hyperparameters."
    )
    parser.add_argument(
        '--objective',
        type=str,
        default='h_loss',
        choices=['val_loss', 'h_loss'],
        help="The objective metric of the study to load ('val_loss' or 'h_loss')."
    )
    args = parser.parse_args()

    # --- 2. Define Study Details ---
    # These must match the values used in your optimize_hyperparams.py script.
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data')
    db_name = 'optimize_hyperparams.db'
    db_path = os.path.join(results_dir, db_name)
    storage_name = f"sqlite:///{db_path}"

    # Dynamically set the study name based on the chosen objective
    study_name = f"sphnn_pinn_optimization_{args.objective}"

    # --- 3. Load the Study ---
    print(f"Loading study '{study_name}' from {storage_name}...")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        print(f"\nError: Study '{study_name}' not found in the database.")
        print(f"Please ensure you have run the optimization with '--objective {args.objective}'")
        print(f"And that the database file exists at: {db_path}")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return

    # --- 4. Get the Best Trial and Print Results ---
    best = study.best_trial

    print("\n" + "=" * 50)
    print(f"        Best Trial Found for '{study_name}'")
    print("=" * 50)
    print(f"  Trial Number: {best.number}")
    print(f"  Best Value ({args.objective}): {best.value:.6f}")
    print("\n  Best Hyperparameters:")
    for key, value in best.params.items():
        print(f"    '{key}': {value},")
    print("=" * 50)

    # --- 5. Save the Best Hyperparameters to a File ---
    output_txt_path = os.path.join(results_dir, f"best_hyperparams_{args.objective}.txt")
    with open(output_txt_path, 'w') as f:
        f.write(f"Best Hyperparameter Optimization Results for Study: '{study_name}'\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Value ({args.objective}): {best.value}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best.params.items():
            f.write(f"    '{key}': {value},\n")

    print(f"\n✅ Best hyperparameters have been saved to: {output_txt_path}")


if __name__ == "__main__":
    main()
