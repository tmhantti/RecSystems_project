import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pprint import pprint
import sys
import datetime as dt

from utils import filter_ratings, load_movielens, temporal_split, print_dataset_statistics
from utils import split_by_user_activity, split_by_popularity, subsample
from plot_routines import (hist_ratings, plot_cum_ratings, hist_user_item_count, 
                           plot_long_tail, plot_long_tail_with_threshold) 
from most_popular import (get_max_timestamp, get_min_timestamp, get_most_popular_movies, 
                          print_most_popular_movies)
from knn_recommender import run_knn_experiment

# subfolder for plots:
# (set to None to display plots on screen)
plot_path = Path.cwd() / "plots" 

# random states used to calculate variability of modeling results:
rand_states = list(range(1, 10))

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
