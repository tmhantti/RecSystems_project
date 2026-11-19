import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections.abc import Iterable
from pathlib import Path
import argparse

from data_loader import filter_ratings, load_movielens, temporal_split
from utils import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    RAND_STATES,
    bootstrap_mean_confidence_interval as _bootstrap_mean_ci,
    calc_metrics_stats,
    evaluate_at_k,
    paired_metric_comparison as _paired_comparison,
    per_user_metrics_for_relevance,
    per_user_ranking_metrics as _per_user_metrics,
)


K = 10
POSITIVE_THRESHOLD = 4.0


class ContentRecommender:
    """Content-based recommender using mean-centred rating profiles, genre vectors, and scaled release year."""

    def __init__(self, positive_threshold: float = 4.0, year_weight: float = 0.3, only_positive: bool = False) -> None:
        self.positive_threshold = positive_threshold
        self.year_weight = year_weight
        self.only_positive = only_positive  # If True, only ratings >= positive_threshold are used.
        self._fitted = False

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "ContentRecommender":
        required_cols = {"UserID", "MovieID", "Rating"}
        if not required_cols.issubset(ratings.columns):
            raise ValueError(f"ratings is missing required columns: {required_cols - set(ratings.columns)}")

        self.train_ratings_ = ratings.copy()
        self.movies_ = movies.copy()
        self.user_rating_positions_ = {
            user_id: positions
            for user_id, positions in self.train_ratings_.groupby("UserID", sort=False).indices.items()
        }

        # 1. Build Multi-hot genre feature vectors
        all_genres = sorted(list(set(g for genres in movies["Genres"] for g in genres)))
        self.genres_ = all_genres

        # 2. Determine release year column and calculate min/max for Min-Max scaling
        year_col = next(
            (column for column in ("MovieYear", "Year", "ReleaseYear") if column in movies.columns),
            None,
        )
        
        if year_col and year_col in movies.columns:
            valid_years = movies[year_col].dropna()
            self.min_year_ = valid_years.min()
            self.max_year_ = valid_years.max()
            self.has_year_ = True
        else:
            self.has_year_ = False
            self.min_year_ = 0
            self.max_year_ = 1

        # 3. Build unified feature matrix for items (Genres + Scaled Year)
        movie_features = []
        for _, row in movies.iterrows():
            movie_id = row["MovieID"]
            
            movie_genres = set(row["Genres"]) if isinstance(row["Genres"], list) else set()
            genre_vec = [1 if g in movie_genres else 0 for g in all_genres]
            
            if self.has_year_ and year_col in row and pd.notna(row[year_col]) and (self.max_year_ > self.min_year_):
                # Min-max scaling puts release years on a comparable 0-1 scale.
                norm_year = (row[year_col] - self.min_year_) / (self.max_year_ - self.min_year_)
            else:
                norm_year = 0.5
                
            scaled_year_feature = [norm_year * self.year_weight]
            combined_vec = genre_vec + scaled_year_feature
            movie_features.append({"MovieID": movie_id, "vector": combined_vec})

        feature_df = pd.DataFrame(movie_features).set_index("MovieID")
        self.movie_features_matrix_ = np.vstack(feature_df["vector"].values)
        self.movie_ids_ = feature_df.index.to_numpy()
        self.movie_id_to_idx_ = {mid: i for i, mid in enumerate(self.movie_ids_)}

        # 4. Compute user average rating (avg(u)) for mean-centring
        self.user_avg_ = self.train_ratings_.groupby("UserID")["Rating"].mean()

        self._fitted = True
        return self

    def _get_user_profile(self, user_id: int) -> np.ndarray:
        """Compute mean-centred user profile vector, optionally filtering for positive ratings only."""
        positions = self.user_rating_positions_.get(user_id)
        if positions is None:
            user_ratings = self.train_ratings_.iloc[0:0]
        else:
            user_ratings = self.train_ratings_.iloc[positions]
        if user_ratings.empty:
            feature_dim = len(self.genres_) + 1
            return np.zeros(feature_dim)

        # When enabled, discard ratings below the positive threshold.
        if self.only_positive:
            user_ratings = user_ratings[user_ratings["Rating"] >= self.positive_threshold]
            if user_ratings.empty:
                feature_dim = len(self.genres_) + 1
                return np.zeros(feature_dim)

        avg_u = self.user_avg_.get(user_id, self.train_ratings_["Rating"].mean())
        movie_indices = user_ratings["MovieID"].map(self.movie_id_to_idx_)
        valid = movie_indices.notna()
        if not valid.any():
            return np.zeros(len(self.genres_) + 1)

        # Mean-centering captures whether a rating is above or below the user's usual rating.
        centred_ratings = user_ratings.loc[valid, "Rating"].to_numpy() - avg_u
        features = self.movie_features_matrix_[movie_indices.loc[valid].to_numpy(dtype=int)]
        return centred_ratings @ features

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        if not self._fitted:
            raise RuntimeError("Fit the recommender before requesting recommendations.")
        if n < 1:
            raise ValueError("n must be positive.")

        positions = self.user_rating_positions_.get(user_id)
        user_ratings = (
            self.train_ratings_.iloc[positions]
            if positions is not None
            else self.train_ratings_.iloc[0:0]
        )
        seen_items = set(user_ratings["MovieID"])

        profile = self._get_user_profile(user_id)
        
        candidate_indices = [i for i, mid in enumerate(self.movie_ids_) if mid not in seen_items]
        if not candidate_indices:
            return []

        candidate_matrix = self.movie_features_matrix_[candidate_indices]
        similarities = cosine_similarity(profile.reshape(1, -1), candidate_matrix).flatten()

        ranked_local_indices = np.argsort(similarities)[::-1]
        ranked_candidates = [self.movie_ids_[candidate_indices[i]] for i in ranked_local_indices]

        return ranked_candidates[:n]

    def recommend_all(self, user_ids: Iterable[int], n: int = 10) -> dict[int, list[int]]:
        if not self._fitted:
            raise RuntimeError("Fit the recommender before requesting recommendations.")
        if n < 1:
            raise ValueError("n must be positive.")

        user_ids = list(user_ids)
        if not user_ids:
            return {}

        profiles = np.vstack([self._get_user_profile(user_id) for user_id in user_ids])
        profile_norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        normalized_profiles = np.divide(
            profiles,
            profile_norms,
            out=np.zeros_like(profiles),
            where=profile_norms != 0,
        )
        item_norms = np.linalg.norm(self.movie_features_matrix_, axis=1, keepdims=True)
        normalized_items = np.divide(
            self.movie_features_matrix_,
            item_norms,
            out=np.zeros_like(self.movie_features_matrix_, dtype=float),
            where=item_norms != 0,
        )

        recommendations: dict[int, list[int]] = {}
        batch_size = 128
        for start in range(0, len(user_ids), batch_size):
            batch_end = min(start + batch_size, len(user_ids))
            scores = normalized_profiles[start:batch_end] @ normalized_items.T
            for local_index, user_id in enumerate(user_ids[start:batch_end]):
                positions = self.user_rating_positions_.get(user_id)
                seen_items = (
                    set(self.train_ratings_.iloc[positions]["MovieID"])
                    if positions is not None
                    else set()
                )
                seen_indices = [
                    self.movie_id_to_idx_[movie_id]
                    for movie_id in seen_items
                    if movie_id in self.movie_id_to_idx_
                ]
                if seen_indices:
                    scores[local_index, seen_indices] = -np.inf
                available_count = len(self.movie_ids_) - len(seen_indices)
                result_count = min(n, available_count)
                if result_count <= 0:
                    recommendations[user_id] = []
                    continue
                top_indices = np.argpartition(scores[local_index], -result_count)[-result_count:]
                top_indices = top_indices[np.argsort(scores[local_index, top_indices])[::-1]]
                recommendations[user_id] = self.movie_ids_[top_indices].tolist()
        return recommendations


def _fit_model(model_name: str, params: dict, train: pd.DataFrame, movies: pd.DataFrame):
    if model_name == "content_all":
        return ContentRecommender(
            positive_threshold=POSITIVE_THRESHOLD,
            year_weight=params["year_weight"],
            only_positive=False,
        ).fit(train, movies)
    if model_name == "content_positive":
        return ContentRecommender(
            positive_threshold=params["positive_threshold"],
            year_weight=params["year_weight"],
            only_positive=True,
        ).fit(train, movies)
    raise ValueError(f"Unknown model: {model_name}")


def _evaluate_model(
    model_name: str,
    params: dict,
    train: pd.DataFrame,
    movies: pd.DataFrame,
    users: Iterable[int],
) -> dict[int, list[int]]:
    model = _fit_model(model_name, params, train, movies)
    return model.recommend_all(users, n=K)


def _tune_model(
    model_name: str,
    candidates: list[dict],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    movies: pd.DataFrame,
    catalogue: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    validation_users = validation["UserID"].unique()
    rows = []
    for params in candidates:
        recommendations = _evaluate_model(
            model_name, params, train, movies, validation_users
        )
        scores = evaluate_at_k(
            recommendations,
            validation,
            catalogue,
            k=K,
            positive_threshold=POSITIVE_THRESHOLD,
        )
        rows.append({**params, **scores})

    results = pd.DataFrame(rows).sort_values(
        ["ndcg_at_10", "recall_at_10"], ascending=False
    )
    best_params = {
        key: results.iloc[0][key]
        for key in candidates[0]
    }
    return best_params, results


def _item_bucket_metrics(
    recommendations: dict[int, list[int]],
    held_out: pd.DataFrame,
    item_counts: pd.Series,
    popularity_cutoff: float,
) -> dict[str, pd.DataFrame]:
    relevant = (
        held_out[held_out["Rating"] >= POSITIVE_THRESHOLD]
        .groupby("UserID")["MovieID"]
        .agg(set)
    )
    return {
        "Popular items": per_user_metrics_for_relevance(
            recommendations,
            {
                user_id: {
                    item
                    for item in user_relevant
                    if item_counts.get(item, 0) >= popularity_cutoff
                }
                for user_id, user_relevant in relevant.items()
            },
            k=K,
        ),
        "Long-tail items": per_user_metrics_for_relevance(
            recommendations,
            {
                user_id: {
                    item
                    for item in user_relevant
                    if item_counts.get(item, 0) < popularity_cutoff
                }
                for user_id, user_relevant in relevant.items()
            },
            k=K,
        ),
    }


def _disaggregated_results(
    model_name: str,
    recommendations: dict[int, list[int]],
    test: pd.DataFrame,
    train: pd.DataFrame,
) -> list[dict]:
    results = []
    user_metrics = _per_user_metrics(recommendations, test)
    user_counts = train["UserID"].value_counts()
    activity_cutoff = user_counts.median()
    for group_name, user_ids in (
        ("Low-activity users", set(user_counts[user_counts < activity_cutoff].index)),
        ("High-activity users", set(user_counts[user_counts >= activity_cutoff].index)),
    ):
        group_metrics = user_metrics[user_metrics["UserID"].isin(user_ids)]
        if group_metrics.empty:
            continue
        low_ci, high_ci = _bootstrap_mean_ci(group_metrics["ndcg_at_10"])
        results.append(
            {
                "model": model_name,
                "slice": group_name,
                "users": len(group_metrics),
                "recall_at_10": group_metrics["recall_at_10"].mean(),
                "ndcg_at_10": group_metrics["ndcg_at_10"].mean(),
                "ndcg_ci_95": f"[{low_ci:.4f}, {high_ci:.4f}]",
            }
        )

    item_counts = train["MovieID"].value_counts()
    popularity_cutoff = item_counts.median()
    for group_name, group_metrics in _item_bucket_metrics(
        recommendations, test, item_counts, popularity_cutoff
    ).items():
        if group_metrics.empty:
            continue
        low_ci, high_ci = _bootstrap_mean_ci(group_metrics["ndcg_at_10"])
        results.append(
            {
                "model": model_name,
                "slice": group_name,
                "users": len(group_metrics),
                "recall_at_10": group_metrics["recall_at_10"].mean(),
                "ndcg_at_10": group_metrics["ndcg_at_10"].mean(),
                "ndcg_ci_95": f"[{low_ci:.4f}, {high_ci:.4f}]",
            }
        )
    return results


def _summarize_ablation(
    model_name: str,
    training_fraction: float,
    seed_scores: list[dict[str, float]],
) -> dict:
    score_frame = pd.DataFrame(seed_scores)
    row = {
        "training_fraction": training_fraction,
        "model": model_name,
        "seeds": len(score_frame),
    }
    metrics = (
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "catalogue_coverage",
    )
    _, metric_stats = calc_metrics_stats(score_frame, metrics, confidence=0.95)
    for metric in metrics:
        values = score_frame[metric].to_numpy(dtype=float)
        row[metric] = float(values.mean())
        if len(values) > 1:
            interval = metric_stats.loc[
                metric_stats["metric"] == metric, "conf_int"
            ].iloc[0]
            row[f"{metric}_ci_95"] = f"[{interval[0]:.4f}, {interval[1]:.4f}]"
        else:
            row[f"{metric}_ci_95"] = "not estimated from one deterministic run"
    return row


def run_content_experiment(data_dir: Path) -> None:
    """
    Run the content-based recommender experiment, including hyperparameter tuning on the validation set and evaluation on the test set. The results are printed to the console.
    """
    _, ratings, movies = load_movielens(data_dir)
    filtered_ratings, filter_stats = filter_ratings(ratings)
    split = temporal_split(filtered_ratings)
    catalogue = filtered_ratings["MovieID"].unique()
    print(
        f"Filtered {filter_stats['rows_removed']} ratings; "
        f"train={len(split.train)}, validation={len(split.validation)}, test={len(split.test)}."
    )

    content_year_weights = [float(value) for value in np.linspace(0.0, 1.0, 20)]
    positive_year_weights = [0.0, 0.1, 0.3, 0.5, 1.0]
    positive_thresholds = [3.0, 3.5, 4.0, 4.5]
    content_all_candidates = [{"year_weight": value} for value in content_year_weights]
    content_positive_candidates = [
        {"year_weight": weight, "positive_threshold": threshold}
        for weight in positive_year_weights
        for threshold in positive_thresholds
    ]
    candidates_by_model = {
        "content_all": content_all_candidates,
        "content_positive": content_positive_candidates,
    }

    print(
        "Validation-only tuning budget: 20 configurations per model "
        "(40 configurations total)."
    )
    print(
        "Search spaces: content_all year_weight=20 evenly spaced values in [0, 1]; "
        "content_positive year_weight={0, 0.1, 0.3, 0.5, 1} x "
        "positive_threshold={3, 3.5, 4, 4.5}."
    )
    best_params = {}
    validation_results = {}
    validation_rows = []
    for model_name, candidates in candidates_by_model.items():
        print(f"Tuning {model_name}: {len(candidates)} configurations.")
        best_params[model_name], validation_results[model_name] = _tune_model(
            model_name,
            candidates,
            split.train,
            split.validation,
            movies,
            catalogue,
        )
        validation_rows.append(
            validation_results[model_name].assign(model=model_name)
        )
        print(
            f"Best validation NDCG@10 for {model_name}: "
            f"{validation_results[model_name].iloc[0]['ndcg_at_10']:.4f}; "
            f"parameters={best_params[model_name]}"
        )
    print("\nValidation search results for all configurations:")
    print(pd.concat(validation_rows, ignore_index=True).to_string(index=False))

    final_train = split.train
    test_users = split.test["UserID"].unique()
    test_user_metrics = {}
    overall_rows = []
    disaggregated_rows = []
    for model_name, params in best_params.items():
        recommendations = _evaluate_model(
            model_name, params, final_train, movies, test_users
        )
        scores = evaluate_at_k(
            recommendations,
            split.test,
            catalogue,
            k=K,
            positive_threshold=POSITIVE_THRESHOLD,
        )
        user_metrics = _per_user_metrics(recommendations, split.test)
        test_user_metrics[model_name] = user_metrics
        ci_low, ci_high = _bootstrap_mean_ci(user_metrics["ndcg_at_10"])
        overall_rows.append(
            {
                "model": model_name,
                **scores,
                "ndcg_ci_95": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "users": len(user_metrics),
            }
        )
        disaggregated_rows.extend(
            _disaggregated_results(
                model_name, recommendations, split.test, final_train
            )
        )

    print("\nOverall test metrics (models fitted on train only, matching LF-SGD):")
    print(pd.DataFrame(overall_rows).to_string(index=False))

    print("\nDisaggregated test results (all group thresholds use training data only):")
    print(pd.DataFrame(disaggregated_rows).to_string(index=False))

    print("\nPaired test-set comparison in per-user NDCG@10:")
    print(
        f"Confidence intervals use {DEFAULT_BOOTSTRAP_ITERATIONS} paired user-bootstrap "
        "resamples; p-values use the paired Wilcoxon signed-rank test."
    )
    paired_rows = [
        _paired_comparison(
            "content_all",
            test_user_metrics["content_all"],
            "content_positive",
            test_user_metrics["content_positive"],
        )
    ]
    print(pd.DataFrame(paired_rows).to_string(index=False))

    ablation_rows = []
    base_scores = {
        row["model"]: {
            metric: row[metric]
            for metric in (
                "precision_at_10",
                "recall_at_10",
                "ndcg_at_10",
                "catalogue_coverage",
            )
        }
        for row in overall_rows
    }
    for model_name in best_params:
        ablation_rows.append(
            _summarize_ablation(
                model_name,
                1.0,
                [base_scores[model_name]],
            )
        )

    for training_fraction in (0.5, 0.2):
        for model_name, params in best_params.items():
            seed_scores = []
            for seed in RAND_STATES:
                sparse_train = final_train.sample(
                    frac=training_fraction, random_state=seed
                ).sort_index()
                recommendations = _evaluate_model(
                    model_name, params, sparse_train, movies, test_users
                )
                seed_scores.append(
                    evaluate_at_k(
                        recommendations,
                        split.test,
                        catalogue,
                        k=K,
                        positive_threshold=POSITIVE_THRESHOLD,
                    )
                )
            ablation_rows.append(
                _summarize_ablation(
                    model_name, training_fraction, seed_scores
                )
            )

    ablation_results = pd.DataFrame(ablation_rows)
    ablation_results["rank"] = ablation_results.groupby("training_fraction")[
        "ndcg_at_10"
    ].rank(method="min", ascending=False).astype(int)
    full_ranks = ablation_results[ablation_results["training_fraction"] == 1.0].set_index(
        "model"
    )["rank"]
    ablation_results["rank_change_vs_full"] = [
        int(row["rank"] - full_ranks[row["model"]])
        for _, row in ablation_results.iterrows()
    ]
    print(
        "\nSparsity ablation on test data (same 10 training subsamples for all models):"
    )
    print(ablation_results.to_string(index=False))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Tune and compare content-based recommender variants."
    )
    parser.add_argument("-d", "--data_dir", type=Path, required=True, help="path to data")
    args = parser.parse_args()
    run_content_experiment(args.data_dir)


if __name__ == "__main__":
    main()