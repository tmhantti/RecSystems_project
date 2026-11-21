"""Train and evaluate the PyTorch two-tower recommender."""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from .two_towers_nn import TwoTowerRecommender
from utils import evaluate_at_k, filter_ratings, load_movielens, print_evaluation_scores
from utils import temporal_split

# Configuration selected on validation NDCG@10 by --sweep and --feature-sweep.
SELECTED_CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "epochs": 40,
    "loss": "bpr",
    "feature_mode": "all",
}


def build_feature_tables(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    feature_mode: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Build numeric side-feature tables indexed by MovieLens IDs."""
    if feature_mode not in {
        "none",
        "item_genres",
        "item_metadata",
        "user_attributes",
        "all",
    }:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    user_features = None
    item_features = None
    if feature_mode in {"user_attributes", "all"}:
        user_features = pd.get_dummies(
            users.set_index("UserID")[["Gender", "Age", "Occupation"]].astype(str),
            dtype=float,
        )
    if feature_mode in {"item_genres", "item_metadata", "all"}:
        genre_names = sorted({genre for genres in movies["Genres"] for genre in genres})
        item_features = pd.DataFrame(
            0.0, index=movies["MovieID"], columns=[f"genre_{name}" for name in genre_names]
        )
        for movie_id, genres in movies.set_index("MovieID")["Genres"].items():
            item_features.loc[movie_id, [f"genre_{genre}" for genre in genres]] = 1.0
    if feature_mode in {"item_metadata", "all"}:
        years = movies.set_index("MovieID")["MovieYear"].astype(float)
        year_range = years.max() - years.min()
        item_features["release_year"] = (
            (years - years.min()) / year_range if year_range else 0.0
        )
        item_features = item_features.fillna(0.0)
    return user_features, item_features


def run_two_tower_experiment(
    data_dir: Path,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 5,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    negative_ratio: int = 1,
    loss: str = "bce",
    feature_mode: str = "none",
    seed: int = 42,
    print_scores: bool = True,
    save_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Train on train, report validation, refit, and report final test scores."""
    users, ratings, movies = load_movielens(data_dir)
    user_features, item_features = build_feature_tables(users, movies, feature_mode)
    filtered_ratings, filter_stats = filter_ratings(ratings)
    split = temporal_split(filtered_ratings)
    catalogue = filtered_ratings["MovieID"].unique()

    if print_scores:
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
        loss=loss,
        seed=seed,
    )
    model.fit(
        split.train,
        catalogue=catalogue,
        user_features=user_features,
        item_features=item_features,
    )
    validation_scores = evaluate_at_k(
        model.recommend_all(split.validation["UserID"].unique()),
        split.validation,
        catalogue,
        k=10,
    )
    if print_scores:
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
        loss=loss,
        seed=seed,
    )
    final_model.fit(
        combined_train,
        catalogue=catalogue,
        user_features=user_features,
        item_features=item_features,
    )
    test_scores = evaluate_at_k(
        final_model.recommend_all(split.test["UserID"].unique()),
        split.test,
        catalogue,
        k=10,
    )
    if print_scores:
        print_evaluation_scores(test_scores, title="--- TWO-TOWER TEST SCORES ---")
    if save_path is not None:
        final_model.save(save_path)
        print(f"Saved final two-tower checkpoint to {save_path}")
    return {"validation": validation_scores, "test": test_scores}


def summarize_seed_scores(
    scores_by_seed: list[dict[str, dict[str, float]]],
) -> None:
    """Print means and 95% t-intervals across fixed-configuration seeds."""
    for split_name in ("validation", "test"):
        print(f"\n--- {split_name.upper()} MULTI-SEED SUMMARY ---")
        metric_names = scores_by_seed[0][split_name]
        for metric in metric_names:
            values = pd.Series(
                result[split_name][metric] for result in scores_by_seed
            )
            interval = stats.t.interval(
                0.95,
                df=len(values) - 1,
                loc=values.mean(),
                scale=stats.sem(values),
            )
            print(
                f"{metric}: mean={values.mean():.4f}, "
                f"95% CI=[{interval[0]:.4f}, {interval[1]:.4f}]"
            )


def run_multi_seed(
    data_dir: Path,
    first_seed: int = 42,
    seed_count: int = 10,
    model_path: Path = Path("two_towers/confidence_interval_last_seed.pt"),
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    negative_ratio: int = 1,
) -> list[dict[str, dict[str, float]]]:
    """Train the selected configuration over consecutive seeds and print t-intervals."""
    print(f"Running selected configuration {SELECTED_CONFIG} over {seed_count} seeds.")
    scores_by_seed = []
    for seed in range(first_seed, first_seed + seed_count):
        print(f"\n=== TWO-TOWER SEED {seed} ===")
        scores_by_seed.append(
            run_two_tower_experiment(
                data_dir=data_dir,
                **SELECTED_CONFIG,
                batch_size=batch_size,
                learning_rate=learning_rate,
                negative_ratio=negative_ratio,
                seed=seed,
                print_scores=False,
                # The last seed's final model is kept for two_tower_metrics.
                save_path=model_path if seed == first_seed + seed_count - 1 else None,
            )
        )
    summarize_seed_scores(scores_by_seed)
    return scores_by_seed


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
    parser.add_argument("--loss", choices=("bce", "bpr"), default="bce")
    parser.add_argument(
        "--feature-mode",
        choices=("none", "item_genres", "item_metadata", "user_attributes", "all"),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("two_towers/confidence_interval_last_seed.pt"),
        help="Path for the final checkpoint in multi-seed mode.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="Train the selected configuration across this many consecutive seeds.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the epoch and architecture comparison sweep instead of one run.",
    )
    parser.add_argument(
        "--feature-sweep",
        action="store_true",
        help="Compare side-feature ablations using the fixed BPR 64/128 model.",
    )
    args = parser.parse_args()

    if sum((args.sweep, args.feature_sweep, args.seed_count > 1)) > 1:
        parser.error("--sweep, --feature-sweep, and --seed-count > 1 are mutually exclusive.")

    if args.seed_count < 1:
        parser.error("--seed-count must be positive.")

    if args.seed_count > 1:
        run_multi_seed(
            args.data_dir,
            first_seed=args.seed,
            seed_count=args.seed_count,
            model_path=args.model_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            negative_ratio=args.negative_ratio,
        )
        return

    if args.feature_sweep:
        feature_modes = (
            "none",
            "item_genres",
            "item_metadata",
            "user_attributes",
            "all",
        )
        results = []
        for feature_mode in feature_modes:
            print(f"\n=== TWO-TOWER FEATURE MODE: {feature_mode} ===")
            scores = run_two_tower_experiment(
                data_dir=args.data_dir,
                embedding_dim=64,
                hidden_dim=128,
                epochs=40,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                negative_ratio=args.negative_ratio,
                loss="bpr",
                feature_mode=feature_mode,
                seed=args.seed,
            )
            results.append(
                {
                    "feature_mode": feature_mode,
                    **scores["validation"],
                    **{
                        f"test_{metric}": value
                        for metric, value in scores["test"].items()
                    },
                }
            )

        print("\n=== TWO-TOWER FEATURE SWEEP SUMMARY ===")
        for result in results:
            print(
                f"feature_mode={result['feature_mode']}: "
                f"validation_ndcg@10={result['ndcg_at_10']:.4f}, "
                f"test_ndcg@10={result['test_ndcg_at_10']:.4f}, "
                f"test_recall@10={result['test_recall_at_10']:.4f}, "
                f"test_coverage={result['test_catalogue_coverage']:.4f}"
            )
        return

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
                loss=args.loss,
                feature_mode=args.feature_mode,
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
        loss=args.loss,
        feature_mode=args.feature_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
