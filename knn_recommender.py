"""Item-based k-nearest-neighbour recommender."""

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from data_loader import filter_ratings, load_movielens, temporal_split
from plot_routines import plot_knn_validation_results


class ItemKNNRecommender:
    """Recommend unseen items using item-item neighbourhood similarity."""

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "cosine",
        positive_only: bool = False,
        positive_threshold: float = 4.0,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        if metric not in {"cosine", "euclidean", "l2"}:
            raise ValueError(
                "metric must be cosine or euclidean."
            )
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.positive_only = positive_only
        self.positive_threshold = positive_threshold
        self._fitted = False

    def fit(self, ratings: pd.DataFrame) -> "ItemKNNRecommender":
        """Fit item similarities using a ratings dataframe."""
        required_columns = {"UserID", "MovieID", "Rating"}
        missing = required_columns.difference(ratings.columns)
        if missing:
            raise ValueError(f"ratings is missing columns: {sorted(missing)}")

        all_train_ratings = ratings.copy()
        interactions = ratings
        if self.positive_only:
            interactions = interactions[interactions["Rating"] >= self.positive_threshold]
        if interactions.empty:
            raise ValueError("Cannot fit kNN with no interactions.")

        item_ids = pd.Index(interactions["MovieID"].unique(), name="MovieID")
        user_ids = pd.Index(interactions["UserID"].unique(), name="UserID")
        item_index = {item_id: row for row, item_id in enumerate(item_ids)}
        user_index = {user_id: column for column, user_id in enumerate(user_ids)}

        rows = interactions["MovieID"].map(item_index).to_numpy()
        columns = interactions["UserID"].map(user_index).to_numpy()
        values = interactions["Rating"].to_numpy(dtype=float)
        matrix = csr_matrix(
            (values, (rows, columns)),
            shape=(len(item_ids), len(user_ids)),
        )

        neighbor_count = min(self.n_neighbors + 1, len(item_ids))
        model = NearestNeighbors(
            n_neighbors=neighbor_count,
            metric=self.metric,
            algorithm="brute",
        )
        model.fit(matrix)
        distances, neighbors = model.kneighbors(matrix)

        self.item_ids_ = item_ids.to_numpy()
        self.item_index_ = item_index
        self.neighbors_ = neighbors[:, 1:]
        self.similarities_ = self._distance_to_similarity(distances[:, 1:])
        self.model_ = model
        self.train_ratings_ = all_train_ratings
        self._fitted = True
        return self

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        """Return up to n unseen item IDs for one user."""
        self._check_fitted()
        if n < 1:
            raise ValueError("n must be positive.")

        user_ratings = self.train_ratings_[self.train_ratings_["UserID"] == user_id]
        seen_items = set(user_ratings["MovieID"])
        candidate_scores: dict[int, float] = {}

        for item_id in seen_items:
            item_row = self.item_index_.get(item_id)
            if item_row is None:
                continue
            for neighbor_row, similarity in zip(
                self.neighbors_[item_row],
                self.similarities_[item_row],
            ):
                candidate_id = int(self.item_ids_[neighbor_row])
                if candidate_id not in seen_items:
                    candidate_scores[candidate_id] = (
                        candidate_scores.get(candidate_id, 0.0) + float(similarity)
                    )

        ranked = sorted(candidate_scores, key=candidate_scores.get, reverse=True)
        return ranked[:n]

    def recommend_all(self, user_ids: Iterable[int], n: int = 10) -> dict[int, list[int]]:
        """Return recommendations for each requested user."""
        return {user_id: self.recommend(user_id, n=n) for user_id in user_ids}

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Fit the recommender before requesting recommendations.")

    def _distance_to_similarity(self, distances: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return np.maximum(1.0 - distances, 0.0)
        if self.metric in {"euclidean", "l2"}:
            return 1.0 / (1.0 + distances)
        raise ValueError(
            "metric must be 'cosine' or 'euclidean' for recommender scoring."
        )


def evaluate_at_k(
    recommendations: dict[int, list[int]],
    held_out: pd.DataFrame,
    catalogue: Iterable[int],
    k: int = 10,
    positive_threshold: float = 4.0,
) -> dict[str, float]:
    """Evaluate ranked recommendations against positive held-out ratings."""
    if k < 1:
        raise ValueError("k must be positive.")

    relevant = (
        held_out[held_out["Rating"] >= positive_threshold]
        .groupby("UserID")["MovieID"]
        .agg(set)
    )
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    recommended_items: set[int] = set()

    for user_id, relevant_items in relevant.items():
        ranked = recommendations.get(user_id, [])[:k]
        recommended_items.update(ranked)
        hits = [item for item in ranked if item in relevant_items]
        precisions.append(len(hits) / k)
        recalls.append(len(hits) / len(relevant_items))

        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, item in enumerate(ranked)
            if item in relevant_items
        )
        ideal_hits = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    catalogue_size = len(set(catalogue))
    return {
        "precision_at_10": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "catalogue_coverage": (
            len(recommended_items) / catalogue_size if catalogue_size else 0.0
        ),
    }


def run_knn_experiment(
    data_dir: Path,
    neighborhood_sizes: Iterable[int] = (1, 2, 5, 10, 15, 20, 30, 40, 60, 80),
    metrics: Iterable[str] = ("cosine", "euclidean"),
    plot_path: Path | None = Path("plots"),
) -> pd.DataFrame:
    """Run the kNN validation grid and print the best configuration."""
    _, ratings, _ = load_movielens(data_dir)
    filtered_ratings, filter_stats = filter_ratings(ratings)
    split = temporal_split(filtered_ratings)
    validation_users = split.validation["UserID"].unique()
    results: list[dict[str, float | int | str]] = []

    print(
        f"Filtering removed {filter_stats['rows_removed']} rows; "
        f"evaluating {len(validation_users)} users."
    )
    for metric in metrics:
        for n_neighbors in neighborhood_sizes:
            model = ItemKNNRecommender(
                n_neighbors=n_neighbors,
                metric=metric,
                positive_only=True,
                positive_threshold=4.0,
            ).fit(split.train)
            recommendations = model.recommend_all(validation_users, n=10)
            scores = evaluate_at_k(
                recommendations,
                split.validation,
                filtered_ratings["MovieID"].unique(),
                k=10,
                positive_threshold=4.0,
            )
            results.append(
                {"metric": metric, "n_neighbors": n_neighbors, **scores}
            )
            print(
                f"{metric=}, {n_neighbors=}: "
                f"NDCG@10={scores['ndcg_at_10']:.4f}, "
                f"Recall@10={scores['recall_at_10']:.4f}, "
                f"Coverage={scores['catalogue_coverage']:.4f}"
            )

    results_df = pd.DataFrame(results).sort_values(
        "ndcg_at_10", ascending=False
    )
    if plot_path is not None:
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_knn_validation_results(results_df, plot_path=plot_path)
    print("\nValidation results:")
    print(results_df.to_string(index=False))
    print("\nBest configuration:")
    print(results_df.iloc[0].to_string())
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune an item-based kNN recommender on MovieLens validation data."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        required=True,
        help="subfolder containing input data files",
    )
    args = parser.parse_args()
    run_knn_experiment(args.data_dir)


if __name__ == "__main__":
    main()
