from pprint import pprint
import pandas as pd
from utils import evaluate_at_k, print_evaluation_scores

def get_max_timestamp(ratings: pd.DataFrame) -> pd.Timestamp:
    """Returns the maximum timestamp in the ratings dataset."""
    return ratings["Timestamp"].max()

def get_min_timestamp(ratings: pd.DataFrame) -> pd.Timestamp:
    """Returns the minimum timestamp in the ratings dataset."""
    return ratings["Timestamp"].min()

def get_most_popular_movies(ratings: pd.DataFrame, movies_df: pd.DataFrame, 
                           start_date: pd.Timestamp = None, 
                           end_date: pd.Timestamp = None, 
                           top_n: int = 5) -> pd.DataFrame:
    """Calculates the top N most popular movies for a given date range (or all-time if None)."""
    df = ratings.copy()
    
    if start_date is not None and end_date is not None:
        df = df[(df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)]
    
    movie_stats = df.groupby("MovieID").agg(
        RatingCount=("Rating", "count"),
        AverageRating=("Rating", "mean")
    ).reset_index()
    
    top_movies = movie_stats.sort_values(by="RatingCount", ascending=False).head(top_n)
    result = pd.merge(top_movies, movies_df, on="MovieID", how="inner")
    
    return result

def print_most_popular_movies(most_popular_movies: pd.DataFrame, title: str = None, start_time=None, end_time=None):
    """Prints the most popular movies in a clean and readable format, including time span."""
    if title:
        print(f"\n{title}")
    if start_time and end_time:
        print(f"Time span: {start_time} - {end_time}")
    
    for rank, (_, row) in enumerate(most_popular_movies.iterrows(), start=1):
        genres = ", ".join(row['Genres']) if isinstance(row['Genres'], list) else row['Genres']
        
        print(
            f"  {rank}. {row['Title']} "
            f"(ID: {row['MovieID']}) | "
            f"Ratings: {row['RatingCount']} | "
            f"Avg Rating: {row['AverageRating']:.2f} | "
            f"Genres: {genres}"
        )

def run_most_popular_experiment(ratings, movies, validation_users, catalogue, max_timestamp, min_timestamp):
    """Runs both recent 1-month and all-time most popular experiments."""
    print("\n*** Getting most popular items (Part I) ***\n")
    
    # 1. Recent 1 month
    start_date_recent = max_timestamp - pd.DateOffset(months=1)
    recent_top_5 = get_most_popular_movies(ratings, movies, start_date=start_date_recent, end_date=max_timestamp, top_n=5)
    print_most_popular_movies(recent_top_5, title="--- MOST POPULAR MOVIES (RECENT 1 MONTH) ---", start_time=start_date_recent, end_time=max_timestamp)

    recent_top_10 = get_most_popular_movies(ratings, movies, start_date=start_date_recent, end_date=max_timestamp, top_n=10)
    recent_scores = evaluate_at_k({u: recent_top_10["MovieID"].tolist() for u in validation_users}, ratings, catalogue, k=10)

    # using utils function to print evaluation scores
    print_evaluation_scores(recent_scores, title="--- RECENT 1 MONTH EVALUATION SCORES ---")

    # 2. All-time
    all_time_top_5 = get_most_popular_movies(ratings, movies, start_date=None, end_date=None, top_n=5)
    print_most_popular_movies(all_time_top_5, title="--- MOST POPULAR MOVIES (ALL-TIME) ---", start_time=min_timestamp, end_time=max_timestamp)

    all_time_top_10 = get_most_popular_movies(ratings, movies, start_date=None, end_date=None, top_n=10)
    all_time_scores = evaluate_at_k({u: all_time_top_10["MovieID"].tolist() for u in validation_users}, ratings, catalogue, k=10)
    
    print_evaluation_scores(all_time_scores, title="--- ALL-TIME EVALUATION SCORES ---")