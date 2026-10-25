import os
import sys
import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint

from plot_routines import (hist_ratings, plot_cum_ratings, hist_user_item_count, plot_long_tail, plot_long_tail_with_threshold) 

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
    parser.add_argument("-d", "--data_dir", 
                        type= Path, 
                        help="subfolder containing input data files")
    args = parser.parse_args()
    
    # read input data sets and set column names:
    rel_path = args.data_dir
    if rel_path.exists():
        users = pd.read_csv(rel_path / "users.dat", sep='::', header=None, engine='python')
        ratings = pd.read_csv(rel_path / "ratings.dat", sep='::', header=None, engine='python')
        movies = pd.read_csv(rel_path / "movies.dat", sep='::', header=None, engine='python', encoding='latin1')
    else: 
        raise FileNotFoundError(f"Subfolder '{rel_path}' was not found.") 
    users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    movies.columns = ['MovieID', 'Title', 'Genres']
    
    # add movie names and publishing year as separate columns, 
    # convert genres column to list:
    rp = r'(.*) ?\(([0-9][0-9][0-9][0-9])\)' # use regex for splitting 
    movies[['MovieName', 'MovieYear']] = movies['Title'].str.extract(rp)
    movies['MovieYear'] = movies['MovieYear'].astype("Int64")
    movies['MovieName'] = movies['MovieName'].str.strip()
    movies['Genres'] = movies['Genres'].str.split("|")
    # check for NaN values:
    nan_rows = movies[movies.isna().any(axis=1)]
    if len(nan_rows) > 0:
        print("* Warning: the following rows in movies df contain NaN values:")
        print(nan_rows)
    
    # convert timestamp column to datetime
    # (timestamp : when rating was submitted, in seconds since 1970-01-01 00:00 UTC) 
    ratings["Timestamp"] = pd.to_datetime(ratings['Timestamp'], unit='s')

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


    # Filter out users with less than 25 ratings and movies with less than 5 ratings based on long tail analysis:
    min_ratings_per_user = 25
    min_ratings_per_movie = 5
    
    rows_before_filter = len(ratings)
    users_before_filter = ratings['UserID'].nunique()
    movies_before_filter = ratings['MovieID'].nunique()

    # Iterative filtering to ensure clean removal of both user and item constraints
    while True:
        u_counts = ratings['UserID'].value_counts()
        i_counts = ratings['MovieID'].value_counts()
        
        valid_users = u_counts[u_counts >= min_ratings_per_user].index
        valid_movies = i_counts[i_counts >= min_ratings_per_movie].index
        
        filtered_ratings = ratings[ratings['UserID'].isin(valid_users) & ratings['MovieID'].isin(valid_movies)]
        
        if len(filtered_ratings) == len(ratings):
            break
        ratings = filtered_ratings

    users_removed = users_before_filter - ratings['UserID'].nunique()
    movies_removed = movies_before_filter - ratings['MovieID'].nunique()
    rows_removed = rows_before_filter - len(ratings)

    print(f"\n* Filtering applied: Removed {users_removed} users (< {min_ratings_per_user} ratings) and {movies_removed} movies (< {min_ratings_per_movie} ratings).")
    print(f"* Removed rows (interactions): {rows_removed}")

    # Calculate and display updated statistics after filtering:
    user_counts, item_counts, rating_counts = print_dataset_statistics(ratings, title=f"--- UPDATED DATASET STATISTICS (Users >= {min_ratings_per_user}, Movies >= {min_ratings_per_movie}) ---")

# ----------------------------------------------------------------------------------------------------------------------------

    # TODO (part 1, DL 8.11):
    '''
    - The data loaded and characterized: 
        * users, items, interactions, density; 
        * distribution of interactions per user and per item, with a plot: 
          this is the long tail from Lecture 5.1, slide 5; 
        * the rating distribution; the time span and how activity is distributed across it.
    
    - A temporal split into train/validation/test (Lecture 3.1, slide 4). Document exactly where you
    cut and why. If you filter out low-activity users or items, state the threshold and how many rows
    it removed.

    - Two baselines:
        1. Item-based kNN (Lecture 1.2, slides 5–10). Tune the neighborhood size k and the
        similarity measure.
        2. Most popular. There is nothing to tune here, but there are choices: all-time popularity or
        a recent window; count all ratings or only positive ones. State which choices you made,
        and report at least two variants.

    - Metrics: Precision@10, Recall@10, NDCG@10 (Lecture 3.1, slides 7–8), plus catalogue
    coverage (Lecture 3.1, slide 9). Ranked against the full catalogue.
    
    - Include in the report a justification for each protocol decision: the split, the feedback
    treatment, the filtering, the candidate set, and the choice of k. One or two sentences each are
    sufficient, referring to the relevant lecture material
    '''

if __name__ == "__main__":
    main()