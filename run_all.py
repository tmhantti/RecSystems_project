import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from utils import (
    load_movielens, 
    print_dataset_statistics, 
    filter_ratings, 
    temporal_split
)
from plot_routines import (
    hist_ratings, 
    plot_cum_ratings, 
    hist_user_item_count, 
    plot_long_tail, 
    plot_long_tail_with_threshold
) 
from most_popular import (
    get_max_timestamp, 
    get_min_timestamp, 
    run_most_popular_experiment
)
from knn_recommender import run_knn_experiment

# subfolder for plots:
plot_path = Path.cwd() / "plots" 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=Path, required=True,
                        help="subfolder containing input data files")
    args = parser.parse_args()
    
    _users, ratings, movies = load_movielens(args.data_dir)

    print("*** Calculating descriptive statistics (Part I) ***\n")
    user_counts, item_counts, rating_counts = print_dataset_statistics(ratings, title="--- INITIAL DATASET STATISTICS ---")
    
    # Cumulative sum of ratings over time
    timestamp_counts = ratings["Timestamp"].value_counts().sort_index()    
    rating_cumsum = timestamp_counts.cumsum() / ratings["Timestamp"].count()

    # Plot initial results
    if plot_path is not None:
        plot_path.mkdir(parents=True, exist_ok=True)
    plot_cum_ratings(rating_cumsum, plot_path=plot_path)
    hist_user_item_count(user_counts, item_counts, semilog_y=True, plot_path=plot_path)
    hist_ratings(ratings, rating_counts, plot_path=plot_path)
    plot_long_tail(user_counts, item_counts, plot_path=plot_path)
    plot_long_tail_with_threshold(user_counts, item_counts, user_threshold=25, item_threshold=5, plot_path=plot_path)

    # Filtering data
    min_ratings_per_user = 25
    min_ratings_per_movie = 5
    ratings, filter_stats = filter_ratings(
        ratings,
        min_ratings_per_user=min_ratings_per_user,
        min_ratings_per_movie=min_ratings_per_movie,
    )

    print(f"\n* Filtering applied: Removed {filter_stats['users_removed']} users (< {min_ratings_per_user} ratings) and {filter_stats['movies_removed']} movies (< {min_ratings_per_movie} ratings).")
    print(f"* Removed rows (interactions): {filter_stats['rows_removed']}")

    # Updated dataset statistics
    user_counts, item_counts, rating_counts = print_dataset_statistics(ratings, title=f"--- UPDATED DATASET STATISTICS (Users >= {min_ratings_per_user}, Movies >= {min_ratings_per_movie}) ---")

    # Temporal split
    split = temporal_split(ratings)
    print(
        f"\n* Temporal split: {len(split.train)} train, "
        f"{len(split.validation)} validation, {len(split.test)} test ratings."
    )

    # Run Most Popular experiment
    max_timestamp = get_max_timestamp(ratings)
    min_timestamp = get_min_timestamp(ratings)
    validation_users = split.validation["UserID"].unique()
    catalogue = ratings["MovieID"].unique()

    print("\n*** Running most popular-recommender (Part I) ***\n")
    run_most_popular_experiment(ratings, movies, validation_users, catalogue, max_timestamp, min_timestamp)

    print("\n*** Running kNN-recommender (Part I) ***\n")
    run_knn_experiment(args.data_dir)

if __name__ == "__main__":
    main()