#!/usr/bin/env python3
"""CLI for base Bayesian hyperparameter optimization.

Default workflow:
  1) HPO on gold train/val (search LR, batch, loss mix, dropout, gamma)
  2) Save to models/saved_models/combined_loss_optimization_results.json
  3) Write winning batch_size into training_config.json
  4) Train weak-label base model with those params:
       python 02_train_optimized_combined_loss.py

Use --data weak for the legacy weak-subset HPO.
"""

import argparse
from src.combined_loss_system import run_combined_loss_optimization


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Run base hyperparameter optimization. '
            'By default searches on gold labels; apply params to weak training.'
        )
    )
    parser.add_argument('--trials', type=int, default=50, help='Number of trials (default: 50)')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per trial (default: 10)')
    parser.add_argument(
        '--data',
        choices=['gold', 'weak'],
        default='gold',
        help=(
            "HPO data source: 'gold' (default) train/score on gold splits then use "
            "params for weak training; 'weak' = legacy weak-subset HPO"
        ),
    )
    args = parser.parse_args()

    print(
        f"Starting base hyperparameter optimization: "
        f"data={args.data}, trials={args.trials}, epochs/trial={args.epochs}"
    )
    run_combined_loss_optimization(
        n_calls=args.trials,
        epochs_per_trial=args.epochs,
        data_source=args.data,
    )
    print("Results: models/saved_models/combined_loss_optimization_results.json")
    print("Next: python 02_train_optimized_combined_loss.py  # weak-label train with these params")


if __name__ == "__main__":
    main()
