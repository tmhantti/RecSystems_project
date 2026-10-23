
from matplotlib import pyplot as plt
from pathlib import Path

def plot_ratings_count(rating_cumsum, plot_path= None):
    ''' Plot relative cumulative ratings over time.'''
    plt.figure(figsize=(11, 5))
    plt.plot(rating_cumsum.index, rating_cumsum.values,
            label="Cumulative proportion of ratings")
    plt.xlabel("Date")
    plt.ylabel("Cumulative proportion of ratings")
    plt.title("Cumulative ratings over time")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path / "ratings_cum_count.png")
    else:
        plt.show()

def hist_user_count(user_counts, semilog_y= False, plot_path= None):
    ''' Plots histogram: number of ratings per user. '''
    plt.figure(figsize=(9, 5))
    plt.hist(user_counts, bins=50, edgecolor="black")
    plt.axvline(user_counts.median(), linestyle="--",
                label=f"Median = {user_counts.median():.1f}")
    plt.axvline(user_counts.mean(), linestyle="-.",
                label=f"Mean = {user_counts.mean():.1f}")
    plt.xlabel("Number of ratings per user")
    plt.ylabel("Number of users")
    plt.title("Distribution of ratings per user")
    plt.legend()
    if semilog_y:
        plt.yscale("log")
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path / "hist_user_count.png")
    else:
        plt.show()

def hist_item_count(item_counts, semilog_y= False, plot_path= None):
    ''' Plot histogram: number of ratings per movie. '''
    plt.figure(figsize=(9, 5))

    plt.hist(item_counts, bins=50, edgecolor="black")
    plt.axvline(item_counts.median(), linestyle="--",
                label=f"Median = {item_counts.median():.1f}")
    plt.axvline(item_counts.mean(), linestyle="-.",
                label=f"Mean = {item_counts.mean():.1f}")
    plt.xlabel("Number of ratings per movie")
    plt.ylabel("Number of movies")
    plt.title("Distribution of ratings per movie")
    plt.legend()
    if semilog_y:
        plt.yscale("log")
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path / "hist_item_count.png")
    else:
        plt.show()

def hist_ratings(ratings, rating_counts, plot_path= None):
    ''' Plot histogram of rating values. '''

    plt.figure(figsize=(9, 5))
    plt.bar(rating_counts.index, rating_counts.values,
            width=0.7, edgecolor="black")
    #plt.axvline(ratings["Rating"].median(), linestyle="--",
    #            label=f"Median = {ratings['Rating'].median():.1f}")
    #plt.axvline(ratings["Rating"].mean(), linestyle="-.",
    #            label=f"Mean = {ratings['Rating'].mean():.2f}")
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.title("Distribution of rating values")
    # lt.legend()
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path / "hist_ratings.png")
    else:
        plt.show()
