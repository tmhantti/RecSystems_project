"""Misc. utility routines : statistics, data loading, filtering, and temporal splitting for MovieLens ratings."""

from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
import numpy as np
import copy

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

    users = pd.read_csv(data_dir / "users.dat", 
        sep="::", 
        header=None, 
        engine="python")
    ratings = pd.read_csv(data_dir / "ratings.dat", 
        sep="::", 
        header=None, 
        engine="python")
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

def print_dataset_statistics(ratings, title=None):
    """Calculate and display dataset statistics, distribution details, and quantiles."""
    rat_mat = ratings.pivot(index="UserID", columns="MovieID", values="Rating")
    rat_mat = rat_mat.to_numpy()

    n = rat_mat.size
    n_users = rat_mat.shape[0]
    n_items = rat_mat.shape[1]
    
    # Number of interactions (= number of non-empty elements in the matrix)
    n_interact = n - np.count_nonzero(np.isnan(rat_mat))
    # Density (= fraction of non-empty elements in the matrix)
    dens = 100 * n_interact / n 
    
    # Dates for the first and last ratings given:
    first_rating = ratings['Timestamp'].min()
    last_rating = ratings['Timestamp'].max()
    
    user_counts = ratings['UserID'].value_counts()    
    item_counts = ratings['MovieID'].value_counts()   
    ratings_count = ratings['Rating'].value_counts().sort_index()
    
    quantiles = [0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    quantiles_df = pd.DataFrame({
        "Users": user_counts.quantile(quantiles),
        "Movies": item_counts.quantile(quantiles),
    })
    quantiles_df.index = [f"{q:.1%}" for q in quantiles]
    quantiles_df.index.name = "quantile"

    if title:
        print(f"\n{title}")
    print(f"* Number of users: {n_users}")
    print(f"* Mean number of ratings per user: {(user_counts.mean()):.2f}")
    print(f"* Number of items (= movies): {n_items}")
    print(f"* Mean number of ratings per movie: {(item_counts.mean()):.2f}")
    print(f"* Number of interactions: {n_interact}")
    print(f"* Density: {dens:.2f}%")
    print(f"* Rating time span : {first_rating} - {last_rating}")
    print(f"* Quantiles: ratings per user and per movie:\n")
    pprint(quantiles_df)
    
    return user_counts, item_counts, ratings_count

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

def split_by_user_activity(ds, ratings, cutoff = None, use_all_data = False):
    ''' Splits TemporalSplit dataset (ds) into two based on user activity.
        Input cutoff is the number of ratings/user used as criterion in splitting.
        Default cutoff value is the median value of ratings/user count.
        If use_all_data is True, split is done using the whole data set,
        otherwise training data set is used.   

        NOTE: The split is only applied to test and validation sets.
    '''
    if use_all_data:
        user_counts = ratings['UserID'].value_counts()
    else:
        user_counts = ds.train['UserID'].value_counts()  
    if cutoff is None:
        cutoff = user_counts.quantile(0.5)

    low_act_users = list(user_counts[user_counts < cutoff].index)
    high_act_users = list(user_counts[user_counts >= cutoff].index)

    ds_low = TemporalSplit(
        train=ds.train, 
        test=ds.test[ds.test['UserID'].isin(low_act_users)],
        validation= ds.validation[ds.validation['UserID'].isin(low_act_users)]
        )
    ds_high = TemporalSplit(
        train=ds.train, 
        test=ds.test[ds.test['UserID'].isin(high_act_users)],
        validation= ds.validation[ds.validation['UserID'].isin(low_act_users)]
        )

    return ds_low, ds_high

def split_by_popularity(ds, ratings, cutoff = 1000, use_all_data = True):
    ''' Splits TemporalSplit dataset (ds) into two based on the movie popularity: 
        highly popular items versus long-tail items.
        Input cutoff is the number of ratings/movie used as criterion in splitting.
        Default cutoff value is based on visual investigation of ratings/movie histogram 
        (see hist_user_item_count.png in plots folder).  
        If use_all_data is True, split is done using the whole data set,
        otherwise training data set is used.   

        NOTE: The split is only applied to test and validation sets.
        NOTE: Popularity is here taken to mean popularity in terms of ratings received 
              (both negative and positive)
    '''
    if use_all_data:
        item_counts = ratings['MovieID'].value_counts()
    else:
        item_counts = ds.train['MovieID'].value_counts()  

    popular_items = list(item_counts[item_counts >= cutoff].index)
    tail_items = list(item_counts[item_counts < cutoff].index)

    ds_pop= TemporalSplit(
        train=ds.train, 
        test=ds.test[ds.test['MovieID'].isin(popular_items)],
        validation= ds.validation[ds.validation['UserID'].isin(popular_items)]
        )
    ds_tail = TemporalSplit(
        train=ds.train, 
        test=ds.test[ds.test['MovieID'].isin(tail_items)],
        validation= ds.validation[ds.validation['UserID'].isin(tail_items)]
        )

    return ds_pop, ds_tail