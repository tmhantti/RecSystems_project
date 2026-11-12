"""Train and evaluate the PyTorch two-tower recommender."""

import argparse
from pathlib import Path

import pandas as pd

from .two_towers_nn import TwoTowerRecommender
from utils import evaluate_at_k, filter_ratings, load_movielens, print_evaluation_scores
from utils import temporal_split


def run_two_tower_experiment(
    data_dir: Path,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 5,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    negative_ratio: int = 1,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Train on train, report validation, refit, and report final test scores."""
    _, ratings, _ = load_movielens(data_dir)
    filtered_ratings, filter_stats = filter_ratings(ratings)
    split = temporal_split(filtered_ratings)
    catalogue = filtered_ratings["MovieID"].unique()

    print(
        f"Filtering removed {filter_stats['rows_removed']} rows; "
        f"training on {len(split.train)} interactions."
    )
    model = TwoTowerRecommender(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    model.fit(split.train, catalogue=catalogue)
    validation_scores = evaluate_at_k(
        model.recommend_all(split.validation["UserID"].unique()),
        split.validation,
        catalogue,
        k=10,
    )
    print_evaluation_scores(
        validation_scores, title="--- TWO-TOWER VALIDATION SCORES ---"
    )

    combined_train = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = TwoTowerRecommender(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    final_model.fit(combined_train, catalogue=catalogue)
    test_scores = evaluate_at_k(
        final_model.recommend_all(split.test["UserID"].unique()),
        split.test,
        catalogue,
        k=10,
    )
    print_evaluation_scores(test_scores, title="--- TWO-TOWER TEST SCORES ---")
    return {"validation": validation_scores, "test": test_scores}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch two-tower recommender on MovieLens."
    )
    parser.add_argument("-d", "--data_dir", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the epoch and architecture comparison sweep instead of one run.",
    )
    args = parser.parse_args()

    if args.sweep:
        configurations = [
            {"epochs": 10, "embedding_dim": 64, "hidden_dim": 128},
            {"epochs": 20, "embedding_dim": 64, "hidden_dim": 128},
            {"epochs": 40, "embedding_dim": 64, "hidden_dim": 128},
            {"epochs": 10, "embedding_dim": 256, "hidden_dim": 512},
            {"epochs": 20, "embedding_dim": 256, "hidden_dim": 512},
            {"epochs": 40, "embedding_dim": 256, "hidden_dim": 512},
        ]
        results = []
        for configuration in configurations:
            print(
                "\n=== TWO-TOWER CONFIGURATION: "
                f"epochs={configuration['epochs']}, "
                f"embedding_dim={configuration['embedding_dim']}, "
                f"hidden_dim={configuration['hidden_dim']} ==="
            )
            scores = run_two_tower_experiment(
                data_dir=args.data_dir,
                embedding_dim=configuration["embedding_dim"],
                hidden_dim=configuration["hidden_dim"],
                epochs=configuration["epochs"],
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                negative_ratio=args.negative_ratio,
                seed=args.seed,
            )
            results.append({**configuration, **scores["validation"], **{
                f"test_{metric}": value
                for metric, value in scores["test"].items()
            }})

        print("\n=== TWO-TOWER SWEEP SUMMARY ===")
        for result in results:
            print(
                f"epochs={result['epochs']}, "
                f"embedding_dim={result['embedding_dim']}, "
                f"hidden_dim={result['hidden_dim']}: "
                f"validation_ndcg@10={result['ndcg_at_10']:.4f}, "
                f"test_ndcg@10={result['test_ndcg_at_10']:.4f}, "
                f"test_recall@10={result['test_recall_at_10']:.4f}, "
                f"test_coverage={result['test_catalogue_coverage']:.4f}"
            )
        return

    run_two_tower_experiment(
        data_dir=args.data_dir,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
