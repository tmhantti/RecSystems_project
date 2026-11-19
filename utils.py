"""Misc. utility routines : statistics, data loading, filtering, and temporal splitting for MovieLens ratings."""

from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from collections.abc import Iterable
import numpy as np
import copy
from scipy import stats

import pandas as pd

RATING_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
USER_COLUMNS = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
MOVIE_COLUMNS = ["MovieID", "Title", "Genres"]
# random states used to calculate variability of modeling results:
RAND_STATES = list(range(0, 10))
DEFAULT_BOOTSTRAP_ITERATIONS = 2000


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

def calc_metrics_stats(df, metrics, confidence=0.95):
    """Calculate mean and confidence interval across RNG seeds for different metrics."""

    summary = []
    for metric in metrics:
        values = df[metric].dropna()
        n = len(values)
        mean = values.mean()
        # standard error of mean:
        if n > 1:
            sem = stats.sem(values) 
        else: 
            sem = 0.0
        # Student's t confidence interval:
        if sem > 1e-9:
            ci_low, ci_high = stats.t.interval(
                confidence,
                df=n - 1,
                loc=mean,
                scale=sem
            )
        else:
            ci_low, ci_high = [mean, mean]
        # store results:
        summary.append({
            "metric": metric,
            "mean": mean,
            "conf_int": [ci_low, ci_high],
        })
    return summary, pd.DataFrame(summary)


def per_user_metrics_for_relevance(
    recommendations: dict[int, list[int]],
    relevant_by_user: dict[int, set[int]],
    k: int = 10,
) -> pd.DataFrame:
    """Calculate Precision@k, Recall@k, and NDCG@k for each user."""
    if k < 1:
        raise ValueError("k must be positive.")

    rows = []
    for user_id, relevant_items in relevant_by_user.items():
        if not relevant_items:
            continue
        ranked = recommendations.get(user_id, [])[:k]
        hits = [item for item in ranked if item in relevant_items]
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, item in enumerate(ranked)
            if item in relevant_items
        )
        ideal_hits = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        rows.append(
            {
                "UserID": user_id,
                f"precision_at_{k}": len(hits) / k,
                f"recall_at_{k}": len(hits) / len(relevant_items),
                f"ndcg_at_{k}": dcg / idcg if idcg else 0.0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "UserID",
            f"precision_at_{k}",
            f"recall_at_{k}",
            f"ndcg_at_{k}",
        ],
    )


def per_user_ranking_metrics(
    recommendations: dict[int, list[int]],
    held_out: pd.DataFrame,
    k: int = 10,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """Calculate per-user ranking metrics for positive held-out ratings."""
    relevant_by_user = (
        held_out[held_out["Rating"] >= positive_threshold]
        .groupby("UserID")["MovieID"]
        .agg(set)
        .to_dict()
    )
    return per_user_metrics_for_relevance(recommendations, relevant_by_user, k=k)


def bootstrap_mean_confidence_interval(
    values: Iterable[float],
    confidence: float = 0.95,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_state: int = 42,
) -> tuple[float, float]:
    """Estimate a bootstrap confidence interval for the mean of independent values."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(random_state)
    bootstrap_means = np.empty(n_bootstrap)
    for iteration in range(n_bootstrap):
        bootstrap_means[iteration] = rng.choice(
            values, size=len(values), replace=True
        ).mean()
    tail_probability = (1.0 - confidence) / 2.0
    low, high = np.quantile(
        bootstrap_means, [tail_probability, 1.0 - tail_probability]
    )
    return float(low), float(high)


def paired_metric_comparison(
    model_a: str,
    metrics_a: pd.DataFrame,
    model_b: str,
    metrics_b: pd.DataFrame,
    metric: str = "ndcg_at_10",
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    """Compare two models on paired per-user metric values."""
    paired = metrics_a[["UserID", metric]].merge(
        metrics_b[["UserID", metric]],
        on="UserID",
        suffixes=("_a", "_b"),
    )
    differences = (paired[f"{metric}_a"] - paired[f"{metric}_b"]).to_numpy()
    ci_low, ci_high = bootstrap_mean_confidence_interval(
        differences,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    if not len(differences):
        p_value = float("nan")
        mean_difference = float("nan")
    elif np.allclose(differences, 0):
        p_value = 1.0
        mean_difference = 0.0
    else:
        p_value = float(stats.wilcoxon(differences).pvalue)
        mean_difference = float(np.mean(differences))
    return {
        "comparison": f"{model_a} - {model_b}",
        "users": len(differences),
        f"mean_{metric}_difference": mean_difference,
        "difference_ci_95": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "wilcoxon_p_value": p_value,
    }


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

        NOTE: split is only applied to test and validation sets.
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
        validation= ds.validation[ds.validation['UserID'].isin(high_act_users)]
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

def subsample(ds, x, random_state = None):
    ''' Remove fraction x of rows from training data in TemporalSplit object. '''

    ds_sparse = TemporalSplit(
        train=ds.train.sample(frac=1-x, random_state=random_state).sort_index(),
        test=ds.test.copy(),
        validation= ds.validation.copy()
    )

    return ds_sparse

def evaluate_at_k(
    recommendations: dict[int, list[int]],
    held_out: pd.DataFrame,
    catalogue: Iterable[int],
    k: int = 10,
    positive_threshold: float = 4.0,
) -> dict[str, float]:
    """Evaluate ranked recommendations against positive held-out ratings universally for all models."""
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

def print_evaluation_scores(scores: dict, title: str = None):
    """Prints evaluation metrics dictionary in a clean, readable format without dict syntax."""
    if title:
        print(f"\n{title}")
    for key, value in scores.items():
        print(f"- {key}: {value}")