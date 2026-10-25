"""Data loading, filtering, and temporal splitting for MovieLens ratings."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RATING_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
USER_COLUMNS = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
MOVIE_COLUMNS = ["MovieID", "Title", "Genres"]


@dataclass(frozen=True)
class TemporalSplit:
    """Per-user chronological train, validation, and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_movielens(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and lightly parse the MovieLens .dat files."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Subfolder '{data_dir}' was not found.")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path '{data_dir}' is not a directory.")

    users = pd.read_csv(data_dir / "users.dat", sep="::", header=None, engine="python")
    ratings = pd.read_csv(data_dir / "ratings.dat", sep="::", header=None, engine="python")
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        header=None,
        engine="python",
        encoding="latin1",
    )

    users.columns = USER_COLUMNS
    ratings.columns = RATING_COLUMNS
    movies.columns = MOVIE_COLUMNS

    title_pattern = r"(.*) ?\(([0-9][0-9][0-9][0-9])\)"
    movies[["MovieName", "MovieYear"]] = movies["Title"].str.extract(title_pattern)
    movies["MovieYear"] = movies["MovieYear"].astype("Int64")
    movies["MovieName"] = movies["MovieName"].str.strip()
    movies["Genres"] = movies["Genres"].str.split("|")

    nan_rows = movies[movies.isna().any(axis=1)]
    if not nan_rows.empty:
        print("* Warning: the following rows in movies df contain NaN values:")
        print(nan_rows)

    ratings["Timestamp"] = pd.to_datetime(ratings["Timestamp"], unit="s")
    return users, ratings, movies


def filter_ratings(
    ratings: pd.DataFrame,
    min_ratings_per_user: int = 25,
    min_ratings_per_movie: int = 5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Iteratively enforce minimum user and movie rating counts."""
    filtered = ratings.copy()
    rows_before = len(filtered)
    users_before = filtered["UserID"].nunique()
    movies_before = filtered["MovieID"].nunique()

    while True:
        user_counts = filtered["UserID"].value_counts()
        movie_counts = filtered["MovieID"].value_counts()
        valid_users = user_counts[user_counts >= min_ratings_per_user].index
        valid_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
        next_filtered = filtered[
            filtered["UserID"].isin(valid_users)
            & filtered["MovieID"].isin(valid_movies)
        ]
        if len(next_filtered) == len(filtered):
            break
        filtered = next_filtered

    stats = {
        "users_removed": users_before - filtered["UserID"].nunique(),
        "movies_removed": movies_before - filtered["MovieID"].nunique(),
        "rows_removed": rows_before - len(filtered),
    }
    return filtered, stats


def temporal_split(
    ratings: pd.DataFrame,
    validation_size: int = 2,
    test_size: int = 2,
) -> TemporalSplit:
    """Split each user's ratings chronologically into train/validation/test."""
    if validation_size < 1 or test_size < 1:
        raise ValueError("validation_size and test_size must both be positive.")

    required_ratings = validation_size + test_size + 1
    counts = ratings["UserID"].value_counts()
    users_with_history = counts[counts >= required_ratings].index
    usable = ratings[ratings["UserID"].isin(users_with_history)].copy()
    usable = usable.sort_values(["UserID", "Timestamp"], kind="mergesort")

    test = usable.groupby("UserID", group_keys=False).tail(test_size)
    remaining = usable.drop(test.index)
    validation = remaining.groupby("UserID", group_keys=False).tail(validation_size)
    train = remaining.drop(validation.index)

    return TemporalSplit(
        train=train.sort_index(),
        validation=validation.sort_index(),
        test=test.sort_index(),
    )
