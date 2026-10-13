#!/usr/bin/env python3
"""
Run Bayesian hyperparameter optimization for the gold fine-tuning stage.

Reuses the CombinedLossOptimizer Bayesian search (gp_minimize) from weak-label
HPO, but each trial reloads the pretrained base model and fine-tunes on the
fixed gold train split, scoring only on gold validation (never AnnualTS test).

Search space (hardcoded):
  - learning_rate: log-uniform [1e-5, 1e-3]
  - batch_size: {2, 4, 6, 8}

Class weights are frozen from training_config.json (no coffee-weight search).
Loss mixture weights are reused from base weak-label HPO (not searched).

After search, winning learning_rate / batch_size are written into
finetuning_config.json. Final fine-tuning is NOT run automatically.
"""

import argparse

from src.combined_loss_system import run_finetuning_optimization


def main():
    parser = argparse.ArgumentParser(
        description='Run gold fine-tuning hyperparameter optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 30 trials, 20 epochs each, ES patience 8
  python run_finetuning_hyperparameter_optimization.py

  # Quick: 15 trials, 10 epochs each
  python run_finetuning_hyperparameter_optimization.py --trials 15 --epochs 10

  # Thorough: 50 trials, patience 12
  python run_finetuning_hyperparameter_optimization.py --trials 50 --patience 12

  # Search only (do not overwrite finetuning_config.json)
  python run_finetuning_hyperparameter_optimization.py --no-update-config
        """
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=30,
        help='Number of Bayesian trials (default: 30)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Max epochs per trial (default: 20)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=8,
        help='Early-stopping patience on val_mean_iou (default: 8)'
    )
    parser.add_argument(
        '--no-update-config',
        action='store_true',
        help='Do not write winning params into finetuning_config.json'
    )

    args = parser.parse_args()

    print(f"\nStarting gold fine-tuning hyperparameter optimization:")
    print(f"  Trials: {args.trials}")
    print(f"  Epochs per trial: {args.epochs}")
    print(f"  Early-stopping patience: {args.patience}")
    print(f"  Update finetuning_config.json: {not args.no_update_config}\n")

    best_params, results = run_finetuning_optimization(
        n_calls=args.trials,
        epochs_per_trial=args.epochs,
        early_stopping_patience=args.patience,
        update_config=not args.no_update_config,
    )

    print(f"\nOptimization complete!")
    print(f"Trial log: models/saved_models/gold_finetune_hpo_results.json")
    if best_params and not args.no_update_config:
        print(f"Winning params written to: finetuning_config.json")
    print(f"\nRun final fine-tuning manually:")
    print(f"  python 07_transfer_learning_finetune.py")


if __name__ == "__main__":
    main()
