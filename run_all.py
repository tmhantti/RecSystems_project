import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pprint import pprint

from data_loader import filter_ratings, load_movielens, temporal_split
from plot_routines import (hist_ratings, plot_cum_ratings, hist_user_item_count, plot_long_tail, plot_long_tail_with_threshold) 
from most_popular import get_max_timestamp, get_min_timestamp, get_most_popular_movies, print_most_popular_movies
from knn_recommender import run_knn_experiment

# subfolder for plots:
# (set to None to display plots on screen)
plot_path = Path.cwd() / "plots" 

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
    
    quantiles = [0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    quantiles_df = pd.DataFrame({
        "Users": user_counts.quantile(quantiles),
        "Movies": item_counts.quantile(quantiles)
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
    
    return user_counts, item_counts, rating_counts_func(ratings)

def rating_counts_func(ratings):
    return ratings['Rating'].value_counts().sort_index()

def main():
    # set up argument parser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=Path, required=True,
                        help="subfolder containing input data files")
    args = parser.parse_args()
    
    _users, ratings, movies = load_movielens(args.data_dir)

    print("*** Calculating descriptive statistics (Part I) ***\n")
    # Calculate and display initial statistics:
    user_counts, item_counts, rating_counts = print_dataset_statistics(ratings, title="--- INITIAL DATASET STATISTICS ---")
    
    # Cumulative sum of number of ratings given over time (scaled to 0-1 interval):
    timestamp_counts = ratings["Timestamp"].value_counts().sort_index()
    rating_cumsum = timestamp_counts.cumsum() / ratings["Timestamp"].count()

    # plot results:
    if plot_path is not None:
        plot_path.mkdir(parents=True, exist_ok=True)
    plot_cum_ratings(rating_cumsum, plot_path=plot_path)
    hist_user_item_count(user_counts, item_counts, semilog_y=True, plot_path=plot_path)
    hist_ratings(ratings, rating_counts, plot_path=plot_path)
    plot_long_tail(user_counts, item_counts, plot_path=plot_path)
    plot_long_tail_with_threshold(user_counts, item_counts, user_threshold=25, item_threshold=5, plot_path=plot_path)

   
    # # # # #
    # For other contributors
    # If you see that different kind of filtering is needed, feel free to change the thresholds and/or the filtering logic below.
    # # # # #

    print("\n***Getting most popular items (Part I) ***\n")
    min_ratings_per_user = 25
    min_ratings_per_movie = 5
    ratings, filter_stats = filter_ratings(
        ratings,
        min_ratings_per_user=min_ratings_per_user,
        min_ratings_per_movie=min_ratings_per_movie,
    )

    print(f"* Filtering applied: Removed {filter_stats['users_removed']} users (< {min_ratings_per_user} ratings) and {filter_stats['movies_removed']} movies (< {min_ratings_per_movie} ratings).")
    print(f"* Removed rows (interactions): {filter_stats['rows_removed']}")

    # Calculate and display updated statistics after filtering:
    user_counts, item_counts, rating_counts = print_dataset_statistics(ratings, title=f"--- UPDATED DATASET STATISTICS (Users >= {min_ratings_per_user}, Movies >= {min_ratings_per_movie}) ---")

    # # # # #
    # Most popular movies
    # # # # #
    
    # Get the maximum and minimum timestamps from the ratings dataset
    max_timestamp = get_max_timestamp(ratings)
    min_timestamp = get_min_timestamp(ratings)
    
    # Most popular movies (recent 1 month):
    start_date_recent = max_timestamp - pd.DateOffset(months=1)

    # Recent top popular movies
    recent_top = get_most_popular_movies(ratings, movies, start_date=start_date_recent, end_date=max_timestamp, top_n=5)
    print_most_popular_movies(recent_top, title="--- MOST POPULAR MOVIES (RECENT 1 MONTH) ---", start_time=start_date_recent, end_time=max_timestamp)

    # All time top popular movies
    all_time_top = get_most_popular_movies(ratings, movies, start_date=None, end_date=None, top_n=5)
    print_most_popular_movies(all_time_top, title="--- MOST POPULAR MOVIES (ALL-TIME) ---", start_time=min_timestamp, end_time=max_timestamp)

    split = temporal_split(ratings)
    print(
        f"\n* Temporal split: {len(split.train)} train, "
        f"{len(split.validation)} validation, {len(split.test)} test ratings."
    )

    # Run kNN - recommender:
    print("\n*** Running kNN-recommender (Part I) ***\n")

    run_knn_experiment(args.data_dir)

if __name__ == "__main__":
    main()
