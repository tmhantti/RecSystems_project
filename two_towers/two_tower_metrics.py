"""Evaluate the trained two-tower checkpoint for the Part 2 comparison.

Produces overall test metrics, a paired per-user comparison against item-kNN,
disaggregated results, and the 50% / 20% sparsity ablation. Both models follow
the same protocol: fitted on train + validation, evaluated once on test.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from content_recommender import _disaggregated_results
from knn_recommender import ItemKNNRecommender
from utils import (
    RAND_STATES,
    bootstrap_mean_confidence_interval,
    evaluate_at_k,
    filter_ratings,
    load_movielens,
    paired_metric_comparison,
    per_user_metrics_for_relevance,
    per_user_ranking_metrics,
    temporal_split,
)

from .train_tt import SELECTED_CONFIG, build_feature_tables
from .two_towers_nn import TwoTowerRecommender

K = 10
POSITIVE_THRESHOLD = 4.0
# kNN configuration selected on validation in Part 1.
KNN_CONFIG = {"n_neighbors": 10, "metric": "cosine"}
# Items with fewer positive training ratings than this count as cold.
COLD_ITEM_POSITIVES = 5
METRICS = ("precision_at_10", "recall_at_10", "ndcg_at_10", "catalogue_coverage")
DEFAULT_SEED = 51


def load_protocol(data_dir: Path):
    """Rebuild the shared filtering, temporal split and side features."""
    users, ratings, movies = load_movielens(data_dir)
    filtered_ratings, _ = filter_ratings(ratings)
    split = temporal_split(filtered_ratings)
    catalogue = filtered_ratings["MovieID"].unique()
    final_train = pd.concat([split.train, split.validation], ignore_index=True)
    user_features, item_features = build_feature_tables(
        users, movies, SELECTED_CONFIG["feature_mode"]
    )
    return split, final_train, catalogue, user_features, item_features


def fit_two_tower(train, catalogue, user_features, item_features, seed):
    model = TwoTowerRecommender(
        embedding_dim=SELECTED_CONFIG["embedding_dim"],
        hidden_dim=SELECTED_CONFIG["hidden_dim"],
        epochs=SELECTED_CONFIG["epochs"],
        loss=SELECTED_CONFIG["loss"],
        seed=seed,
    )
    return model.fit(
        train,
        catalogue=catalogue,
        user_features=user_features,
        item_features=item_features,
    )


def fit_knn(train):
    return ItemKNNRecommender(
        **KNN_CONFIG, positive_only=True, positive_threshold=POSITIVE_THRESHOLD
    ).fit(train)


def load_or_train_two_tower(
    model_path, final_train, catalogue, user_features, item_features, seed
):
    """Load the saved final model, or retrain and save it if it is missing."""
    if model_path.exists():
        model = TwoTowerRecommender.load(model_path)
        if len(model.train_ratings_) != len(final_train):
            raise ValueError(
                f"{model_path} was trained on {len(model.train_ratings_)} ratings, "
                f"expected {len(final_train)} (train + validation). Delete it to retrain."
            )
        print(f"Loaded two-tower checkpoint {model_path} (seed {model.seed}).")
        return model
    print(f"{model_path} not found; training the selected configuration with seed {seed}.")
    model = fit_two_tower(final_train, catalogue, user_features, item_features, seed)
    model.save(model_path)
    return model


def cold_item_results(model_name, recommendations, test, train):
    """NDCG/recall restricted to relevant test items with few positive training ratings."""
    positive_counts = train[train["Rating"] >= POSITIVE_THRESHOLD]["MovieID"].value_counts()
    relevant = (
        test[test["Rating"] >= POSITIVE_THRESHOLD].groupby("UserID")["MovieID"].agg(set)
    )
    rows = []
    for slice_name, is_member in (
        (f"Cold items (<{COLD_ITEM_POSITIVES} train positives)",
         lambda item: positive_counts.get(item, 0) < COLD_ITEM_POSITIVES),
        (f"Warm items (>={COLD_ITEM_POSITIVES} train positives)",
         lambda item: positive_counts.get(item, 0) >= COLD_ITEM_POSITIVES),
    ):
        group_metrics = per_user_metrics_for_relevance(
            recommendations,
            {
                user_id: {item for item in items if is_member(item)}
                for user_id, items in relevant.items()
            },
            k=K,
        )
        if group_metrics.empty:
            continue
        low_ci, high_ci = bootstrap_mean_confidence_interval(group_metrics["ndcg_at_10"])
        rows.append(
            {
                "model": model_name,
                "slice": slice_name,
                "users": len(group_metrics),
                "recall_at_10": group_metrics["recall_at_10"].mean(),
                "ndcg_at_10": group_metrics["ndcg_at_10"].mean(),
                "ndcg_ci_95": f"[{low_ci:.4f}, {high_ci:.4f}]",
            }
        )
    return rows


def summarize_runs(model_name, training_fraction, run_scores):
    """Mean and 95% t-interval across subsample seeds."""
    frame = pd.DataFrame(run_scores)
    row = {"training_fraction": training_fraction, "model": model_name, "runs": len(frame)}
    for metric in METRICS:
        values = frame[metric]
        row[metric] = values.mean()
        if len(values) > 1:
            low, high = stats.t.interval(
                0.95, df=len(values) - 1, loc=values.mean(), scale=stats.sem(values)
            )
            row[f"{metric}_ci_95"] = f"[{low:.4f}, {high:.4f}]"
        else:
            row[f"{metric}_ci_95"] = "single run"
    return row


def sparsity_ablation(
    split, final_train, catalogue, user_features, item_features, full_scores, seeds
):
    """Refit both models on 50% and 20% subsamples of train + validation."""
    rows = [summarize_runs(name, 1.0, [scores]) for name, scores in full_scores.items()]
    test_users = split.test["UserID"].unique()
    for training_fraction in (0.5, 0.2):
        run_scores = {"two_tower": [], "item_knn": []}
        for seed in seeds:
            print(f"\n=== SPARSITY ABLATION: fraction={training_fraction}, seed={seed} ===")
            # Both models see the identical subsample for each seed.
            sparse_train = final_train.sample(
                frac=training_fraction, random_state=seed
            ).sort_index()
            models = {
                "two_tower": fit_two_tower(
                    sparse_train, catalogue, user_features, item_features, seed
                ),
                "item_knn": fit_knn(sparse_train),
            }
            for name, model in models.items():
                run_scores[name].append(
                    evaluate_at_k(
                        model.recommend_all(test_users, n=K),
                        split.test,
                        catalogue,
                        k=K,
                        positive_threshold=POSITIVE_THRESHOLD,
                    )
                )
        rows.extend(
            summarize_runs(name, training_fraction, scores)
            for name, scores in run_scores.items()
        )
    results = pd.DataFrame(rows)
    results["rank_by_ndcg"] = (
        results.groupby("training_fraction")["ndcg_at_10"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return results


def print_table(title, frame):
    print(f"\n--- {title} ---")
    print(frame.to_string(index=False))


def run_two_tower_metrics(
    data_dir: Path,
    model_path: Path = Path("two_towers/confidence_interval_last_seed.pt"),
    seed: int = DEFAULT_SEED,
    output_dir: Path = Path("report_and_results/two_tower_metrics"),
    ablation_seeds: int = 3,
    skip_ablation: bool = False,
) -> None:
    """Write overall, paired, disaggregated and ablation results as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split, final_train, catalogue, user_features, item_features = load_protocol(data_dir)
    models = {
        "two_tower": load_or_train_two_tower(
            model_path, final_train, catalogue, user_features, item_features, seed
        ),
        "item_knn": fit_knn(final_train),
    }

    test_users = split.test["UserID"].unique()
    overall_rows, user_metrics, disaggregated_rows, full_scores = [], {}, [], {}
    for name, model in models.items():
        recommendations = model.recommend_all(test_users, n=K)
        full_scores[name] = evaluate_at_k(
            recommendations, split.test, catalogue, k=K, positive_threshold=POSITIVE_THRESHOLD
        )
        user_metrics[name] = per_user_ranking_metrics(
            recommendations, split.test, k=K, positive_threshold=POSITIVE_THRESHOLD
        )
        low_ci, high_ci = bootstrap_mean_confidence_interval(user_metrics[name]["ndcg_at_10"])
        overall_rows.append(
            {
                "model": name,
                **full_scores[name],
                "ndcg_ci_95": f"[{low_ci:.4f}, {high_ci:.4f}]",
                "users": len(user_metrics[name]),
            }
        )
        disaggregated_rows.extend(
            _disaggregated_results(name, recommendations, split.test, final_train)
        )
        disaggregated_rows.extend(
            cold_item_results(name, recommendations, split.test, final_train)
        )

    overall = pd.DataFrame(overall_rows)
    per_user = pd.concat(
        frame.assign(model=name) for name, frame in user_metrics.items()
    )
    paired_rows = []
    for metric in ("ndcg_at_10", "recall_at_10", "precision_at_10"):
        comparison = paired_metric_comparison(
            "two_tower", user_metrics["two_tower"], "item_knn", user_metrics["item_knn"],
            metric=metric,
        )
        comparison["mean_difference"] = comparison.pop(f"mean_{metric}_difference")
        paired_rows.append({"metric": metric, **comparison})
    paired = pd.DataFrame(paired_rows)
    disaggregated = pd.DataFrame(disaggregated_rows)

    print_table("OVERALL TEST METRICS (fitted on train + validation)", overall)
    print_table("PAIRED PER-USER COMPARISON (bootstrap CI, Wilcoxon p)", paired)
    print_table("DISAGGREGATED TEST RESULTS (thresholds from training data)", disaggregated)
    overall.to_csv(output_dir / "overall_test.csv", index=False)
    per_user.to_csv(output_dir / "per_user_test.csv", index=False)
    paired.to_csv(output_dir / "paired_comparison.csv", index=False)
    disaggregated.to_csv(output_dir / "disaggregated_test.csv", index=False)

    if not skip_ablation:
        ablation = sparsity_ablation(
            split, final_train, catalogue, user_features, item_features,
            full_scores, RAND_STATES[:ablation_seeds],
        )
        print_table("SPARSITY ABLATION ON TEST", ablation)
        ablation.to_csv(output_dir / "sparsity_ablation.csv", index=False)
    print(f"\nWrote CSV files to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Part 2 metrics for the trained two-tower model."
    )
    parser.add_argument("-d", "--data_dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("two_towers/confidence_interval_last_seed.pt"),
        help="Final two-tower checkpoint; trained and saved here if missing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used only when the checkpoint has to be retrained.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("report_and_results/two_tower_metrics")
    )
    parser.add_argument(
        "--ablation-seeds",
        type=int,
        default=3,
        help="Number of subsample seeds (taken from utils.RAND_STATES) per fraction.",
    )
    parser.add_argument(
        "--skip-ablation", action="store_true", help="Skip the slow sparsity ablation."
    )
    args = parser.parse_args()
    if not 1 <= args.ablation_seeds <= len(RAND_STATES):
        parser.error(f"--ablation-seeds must be between 1 and {len(RAND_STATES)}.")
    run_two_tower_metrics(
        args.data_dir,
        model_path=args.model_path,
        seed=args.seed,
        output_dir=args.output_dir,
        ablation_seeds=args.ablation_seeds,
        skip_ablation=args.skip_ablation,
    )


if __name__ == "__main__":
    main()
