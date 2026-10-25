import pandas as pd

def get_max_timestamp(ratings: pd.DataFrame) -> pd.Timestamp:
    """
    Returns the maximum timestamp in the ratings dataset.
    """
    return ratings["Timestamp"].max()

def get_min_timestamp(ratings: pd.DataFrame) -> pd.Timestamp:
    """
    Returns the minimum timestamp in the ratings dataset.
    """
    return ratings["Timestamp"].min()

def get_most_popular_movies(ratings: pd.DataFrame, movies_df: pd.DataFrame, 
                           start_date: pd.Timestamp = None, 
                           end_date: pd.Timestamp = None, 
                           top_n: int = 5) -> pd.DataFrame:
    """
    Calculates the top N most popular movies for a given date range (or all-time if None),
    along with their rating counts and average ratings.
    Most popular method is done for whole dataset, not only for positive ratings.
    It can be calculated for a specific date range by providing start_date and end_date parameters or for all-time if both are None. 
    """
    df = ratings.copy()
    
    # Filter by date range if provided
    if start_date is not None and end_date is not None:
        df = df[(df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)]
    
    # Group by MovieID to calculate count of ratings and average rating
    movie_stats = df.groupby("MovieID").agg(
        RatingCount=("Rating", "count"),
        AverageRating=("Rating", "mean")
    ).reset_index()
    
    # Sort by rating count descending and pick top N
    top_movies = movie_stats.sort_values(by="RatingCount", ascending=False).head(top_n)
    
    # Merge with movies dataframe to get movie titles and genres
    result = pd.merge(top_movies, movies_df, on="MovieID", how="inner")
    
    return result

def print_most_popular_movies(most_popular_movies: pd.DataFrame, title: str = None, start_time=None, end_time=None):
    """
    Prints the most popular movies in a clean and readable format, including time span.
    """
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