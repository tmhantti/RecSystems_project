"""

Elokuvilla on Genres-kenttä (esim. ['Action', 'Drama']). Voimme esittää jokaisen elokuvan monikuuma-vektorina (multi-hot encoding) kaikkien genrien joukosta:
Genre-vektori: Jokaista tunnettua genreä (Action, Comedy, Drama, Thriller jne.) kohden on oma sarakkeensa (arvo 1, jos elokuvalla on kyseinen genre, muuten 0).

Haluat jakaa käyttäjät personoituihin ryhmiin demografioiden (Gender, Age, Occupation, Zip-code) mukaan ja painottaa suosituksia sen mukaan. Voit toteuttaa tämän kahdella tavalla:


. Matching & Prediction (Yhteensopivuus ja ennuste)

    Lasketaan käyttäjäprofiilin ja ehdolla olevien elokuvien välille kosinislaskenta (Cosine Similarity).

    Ennuste saadaan saman painotetun keskiarvon kaavalla kuin kNN:ssä, mutta kalibroituna piirrevälillä.

Elokuvien vektorisointi (Item Features):
Muuntaa elokuvien genret (Genres) monikuuma-vektoreiksi (multi-hot encoding), joissa jokainen tunnettu genre muodostaa oman ulottuvuutensa.Käyttäjäprofiilin laskenta (User Profile):
Laskee jokaiselle käyttäjälle profiilivektorin kaavalla:$$\text{profile}(u) = \sum \left(r(u,i) - \text{avg}(u)\right) \cdot \text{features}(i)$$Mean-centring varmistaa, että käyttäjän keskiarvon alapuolelle jäävät arvosanat työntävät profiilia poispäin kyseisistä genreistä ja keskiarvon yläpuoliset tuovat lähemmäs.Yhteensopivuus ja suosittelu (Matching & Prediction):
Laskkee käyttäjäprofiilin ja tuntemattomien elokuvien piirrevektoreiden välisen kosinisiirron (Cosine Similarity). Suosittelee käyttäjälle parhaiten vastaavat (korkeimman samankaltaisuuden saaneet) elokuvat.Arviointi (evaluate_at_k):





"""




"""Content-based recommender using user history profile and genre feature vectors."""

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import filter_ratings, load_movielens, temporal_split
from knn_recommender import evaluate_at_k


class ContentRecommender:
    """Content-based recommender using mean-centred rating profiles and genre feature vectors."""

    def __init__(self, positive_threshold: float = 4.0) -> None:
        self.positive_threshold = positive_threshold
        self._fitted = False

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "ContentRecommender":
        """Fit the content-based recommender by building item feature vectors and user average ratings."""
        required_cols = {"UserID", "MovieID", "Rating"}
        if not required_cols.issubset(ratings.columns):
            raise ValueError(f"ratings is missing required columns: {required_cols - set(ratings.columns)}")

        self.train_ratings_ = ratings.copy()
        self.movies_ = movies.copy()

        # 1. Build Multi-hot genre feature vectors for movies
        all_genres = sorted(list(set(g for genres in movies["Genres"] for g in genres)))
        self.genres_ = all_genres

        movie_features = []
        for _, row in movies.iterrows():
            movie_id = row["MovieID"]
            movie_genres = set(row["Genres"]) if isinstance(row["Genres"], list) else set()
            vec = [1 if g in movie_genres else 0 for g in all_genres]
            movie_features.append({"MovieID": movie_id, **dict(zip(all_genres, vec))})

        self.movie_features_df_ = pd.DataFrame(movie_features).set_index("MovieID")
        
        # 2. Compute user average rating (avg(u)) for mean-centring
        self.user_avg_ = self.train_ratings_.groupby("UserID")["Rating"].mean()

        self._fitted = True
        return self

    def _get_user_profile(self, user_id: int) -> np.ndarray:
        """Compute mean-centred user profile vector: profile(u) = sum (r(u,i) - avg(u)) * features(i)."""
        user_ratings = self.train_ratings_[self.train_ratings_["UserID"] == user_id]
        if user_ratings.empty:
            return np.zeros(len(self.genres_))

        avg_u = self.user_avg_.get(user_id, self.train_ratings_["Rating"].mean())
        
        profile = np.zeros(len(self.genres_))
        for _, row in user_ratings.iterrows():
            movie_id = row["MovieID"]
            rating = row["Rating"]
            if movie_id in self.movie_features_df_.index:
                features = self.movie_features_df_.loc[movie_id].to_numpy(dtype=float)
                centred_rating = rating - avg_u
                profile += centred_rating * features

        return profile

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        """Return up to n unseen item IDs for one user ranked by cosine similarity to user profile."""
        if not self._fitted:
            raise RuntimeError("Fit the recommender before requesting recommendations.")
        if n < 1:
            raise ValueError("n must be positive.")

        user_ratings = self.train_ratings_[self.train_ratings_["UserID"] == user_id]
        seen_items = set(user_ratings["MovieID"])

        profile = self._get_user_profile(user_id)
        if np.all(profile == 0):
            unseen_movies = [m for m in self.movie_features_df_.index if m not in seen_items]
            return unseen_movies[:n]

        candidate_ids = [m for m in self.movie_features_df_.index if m not in seen_items]
        if not candidate_ids:
            return []

        candidate_matrix = self.movie_features_df_.loc[candidate_ids].to_numpy(dtype=float)
        similarities = cosine_similarity(profile.reshape(1, -1), candidate_matrix).flatten()

        ranked_indices = np.argsort(similarities)[::-1]
        ranked_candidates = [candidate_ids[i] for i in ranked_indices]

        return ranked_candidates[:n]

    def recommend_all(self, user_ids: Iterable[int], n: int = 10) -> dict[int, list[int]]:
        """Return recommendations for each requested user."""
        return {user_id: self.recommend(user_id, n=n) for user_id in user_ids}


def run_content_experiment(data_dir: Path) -> None:
    """Run content-based recommender experiment and evaluate on validation data."""
    print("\n*** Running Content-Based Recommender Experiment (User Profile & Genres) ***\n")
    users, ratings, movies = load_movielens(data_dir)
    filtered_ratings, filter_stats = filter_ratings(ratings)
    split = temporal_split(filtered_ratings)
    validation_users = split.validation["UserID"].unique()

    print(
        f"Filtering removed {filter_stats['rows_removed']} rows; "
        f"evaluating {len(validation_users)} users."
    )

    recommender = ContentRecommender(positive_threshold=4.0)
    recommender.fit(split.train, movies)

    recommendations = recommender.recommend_all(validation_users, n=10)
    scores = evaluate_at_k(
        recommendations,
        split.validation,
        filtered_ratings["MovieID"].unique(),
        k=10,
        positive_threshold=4.0,
    )

    print("\nContent-Based Recommender Validation Scores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run content-based movie recommender based on user profiles and genre features."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        required=True,
        help="subfolder containing input data files",
    )
    args = parser.parse_args()
    run_content_experiment(args.data_dir)


if __name__ == "__main__":
    main()