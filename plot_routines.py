
from matplotlib import pyplot as plt
from pathlib import Path

def hist_user_item_count(user_counts, item_counts,
                         semilog_y=False, plot_path=None):
    '''Plots histograms of ratings per user and per movie.'''
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))

    # top : ratings per user
    axes[0].hist(user_counts, bins=75, edgecolor="black")
    axes[0].axvline(
        user_counts.median(), linestyle="-", color="red",
        label=f"Median = {user_counts.median():.1f}"
    )
    axes[0].axvline(
        user_counts.mean(), linestyle="--", color="red",
        label=f"Mean = {user_counts.mean():.1f}"
    )
    axes[0].set_xlabel("Number of ratings per user")
    axes[0].set_ylabel("Number of users")
    axes[0].set_title("Distribution of ratings per user")
    axes[0].set_xlim(left=0)
    axes[0].legend()

    # bottom: ratings per movie
    axes[1].hist(item_counts, bins=75, edgecolor="black")
    axes[1].axvline(
        item_counts.median(), linestyle="-", color="red",
        label=f"Median = {item_counts.median():.1f}"
    )
    axes[1].axvline(
        item_counts.mean(), linestyle="--", color="red",
        label=f"Mean = {item_counts.mean():.1f}"
    )
    axes[1].set_xlabel("Number of ratings per movie")
    axes[1].set_ylabel("Number of movies")
    axes[1].set_title("Distribution of ratings per movie")
    axes[1].set_xlim(left=0)
    axes[1].legend()

    if semilog_y:
        for ax in axes:
            ax.set_yscale("log")
    for ax in axes:
        ax.grid()
    fig.tight_layout()
    if plot_path is not None:
        fig.savefig(plot_path / "hist_user_item_count.png")
        plt.close(fig)
    else:
        plt.show()

def plot_cum_ratings(rating_cumsum, plot_path= None):
    ''' Plot cumulative ratings (scaled to 0-1 interval) over time.'''
    plt.figure(figsize=(11, 5))
    plt.plot(rating_cumsum.index, rating_cumsum.values,
            label="Cumulative proportion of ratings")
    plt.xlabel("Date")
    plt.ylabel("Cumulative proportion of ratings")
    plt.title("Cumulative ratings over time")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path / "ratings_cum_count.png")
    else:
        plt.show()

def hist_ratings(ratings, rating_counts, plot_path= None):
    ''' Plot histogram of rating distribution. '''

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
    # plt.legend()
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path / "hist_ratings.png")
    else:
        plt.show()

def plot_long_tail(user_counts, item_counts, plot_path=None):
    '''Plots long tail of ratings per user and per movie.'''
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))

    # top : ratings per user
    axes[0].plot(user_counts.sort_values(ascending=False).values)
    axes[0].set_xlabel("User rank")
    axes[0].set_ylabel("Number of ratings")
    axes[0].set_title("Long tail of ratings per user")
    axes[0].set_xlim(left=0)
    axes[0].grid()

    # bottom: ratings per movie
    axes[1].plot(item_counts.sort_values(ascending=False).values)
    axes[1].set_xlabel("Movie rank")
    axes[1].set_ylabel("Number of ratings")
    axes[1].set_title("Long tail of ratings per movie")
    axes[1].set_xlim(left=0)
    axes[1].grid()

    fig.tight_layout()
    if plot_path is not None:
        fig.savefig(plot_path / "long_tail_user_item_count.png")
        plt.close(fig)
    else:
        plt.show()


def plot_long_tail_with_threshold(user_counts, item_counts, user_threshold=25, item_threshold=5, plot_path=None):
    '''Plots long tail of ratings per user and per movie with thresholds.
    Thresholds are drawn as horizontal lines, to show which users and movies would be filtered out.'''
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))

    # top : ratings per user
    axes[0].plot(user_counts.sort_values(ascending=False).values)
    axes[0].axhline(user_threshold, color='red', linestyle='--', label=f'Threshold = {user_threshold}')
    axes[0].set_xlabel("User rank")
    axes[0].set_ylabel("Number of ratings")
    axes[0].set_title("Long tail of ratings per user with threshold")
    axes[0].set_xlim(left=0)
    axes[0].grid()
    axes[0].legend()

    # bottom: ratings per movie
    axes[1].plot(item_counts.sort_values(ascending=False).values)
    axes[1].axhline(item_threshold, color='red', linestyle='--', label=f'Threshold = {item_threshold}')
    axes[1].set_xlabel("Movie rank")
    axes[1].set_ylabel("Number of ratings")
    axes[1].set_title("Long tail of ratings per movie with threshold")
    axes[1].set_xlim(left=0)
    axes[1].grid()
    axes[1].legend()

    fig.tight_layout()
    if plot_path is not None:
        fig.savefig(plot_path / "long_tail_user_item_count_with_threshold.png")
        plt.close(fig)
    else:
        plt.show()


def plot_knn_validation_results(results, plot_path=None):
    """Plot validation quality and catalogue coverage over neighborhood size."""
    metrics = [
        ("ndcg_at_10", "NDCG@10"),
        ("recall_at_10", "Recall@10"),
        ("catalogue_coverage", "Catalogue coverage"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 12), sharex=True)

    for axis, (column, label) in zip(axes, metrics):
        for metric, group in results.groupby("metric"):
            ordered = group.sort_values("n_neighbors")
            axis.plot(
                ordered["n_neighbors"],
                ordered[column],
                marker="o",
                label=metric,
            )
        axis.set_ylabel(label)
        axis.grid()
        axis.legend()

    axes[-1].set_xlabel("Neighborhood size (k)")
    fig.suptitle("Item-based kNN validation results")
    fig.tight_layout()
    if plot_path is not None:
        fig.savefig(Path(plot_path) / "knn_validation_results.png")
        plt.close(fig)
    else:
        plt.show()