import numpy as np
import pandas as pd
import sys
from itertools import islice
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from matplotlib import pyplot as plt

from utils import load_movielens # , filter_ratings,  temporal_split
from two_towers.two_tower_metrics import (load_protocol, 
                                          load_or_train_two_tower, 
                                          DEFAULT_SEED, 
                                          POSITIVE_THRESHOLD)
from two_towers.two_tower_metrics import evaluate_at_k as tt_evaluate_at_k

# number of recommendations per user retrieved from the original recommender
N_RECS = 2000
# number of recommendations per user given by reranker:
K = 10 

def rerank_with_mmr(data_dir, plot_path):
    _, _, movies = load_movielens(data_dir)

    # load / train the final two towers model:
    split, final_train, catalogue, user_features, item_features = load_protocol(data_dir)
    test_users = split.test["UserID"].unique()
    model_path = Path("two_towers/confidence_interval_last_seed.pt")
    seed = DEFAULT_SEED
    tt_model = load_or_train_two_tower(
            model_path, final_train, catalogue, user_features, item_features, seed
        )

    # get N_RECS most relevant items + their scores for each user as dict ( {userId : {movieID: score} }
    user_rel = tt_model.recommend_all_with_scores(test_users, n=N_RECS)
    # get all recommended items and remove users without any recommendations:
    item_ids = set()
    no_recs_users = []
    for u, r in user_rel.items():
        if type(r) is dict: 
            for id in list(r.keys()):
                item_ids.add(id)
        else:
            no_recs_users.append(u)
    user_rel = {k: v for k, v in user_rel.items() if k not in no_recs_users}
    item_ids = sorted(item_ids)
    print(f"* MMR reranking : removed {len(no_recs_users)} test users with no recommendations by the original recommender.")
    # convert to numpy array
    rel_m, user_ids = dict_to_matrix(user_rel, item_ids)
    # get item genres as binary numpy array (rows = item genre vectors):
    movies_filtered = movies[movies["MovieID"].isin(item_ids)]
    G, _, genre_names = genres_to_matrix(movies_filtered)
    # calculate item-item similarities:
    S = cosine_similarity(G)      
    # calculate genre distributions for users using training data:
    user_hist_dist = user_genre_distribution(final_train, G, item_ids, user_ids)
    n_users, n_items = rel_m.shape
    n_items_tot = len(set(catalogue))
    print(f"* MMR reranking : original pool of recommendations ({N_RECS} items / user) contain {n_items} out of {n_items_tot} items.")
    lams = np.linspace(0, 1, 11)
    rows = []
    for lam in lams: 
        # re-rank recommendations using MMR:
        lists = [mmr(rel_m[u], S, K, lam) for u in range(n_users)]
        # get recommendations for each user: 
        recs = {u: [item_ids[j] for j in l] for u, l in zip(user_ids, lists)}
        # calculate performace metrics:
        scores = tt_evaluate_at_k(
            recs, split.test, catalogue, k=K, positive_threshold=POSITIVE_THRESHOLD
        )   
        result = {
            "lambda": lam,
            "precision_at_10": scores["precision_at_10"],
            "recall_at_10": scores["recall_at_10"],
            "ndcg_at_10": scores["ndcg_at_10"],
            "intra_list_diversity": ild(lists, S),
            "coverage": coverage(lists, n_items_tot),
            "gini": gini(lists, n_items),
            "calibration_error": calibration_error(lists, user_hist_dist, G),
        }
        rows.append(result)
        print(f"\n* MMR reranking : Results for trade-off parameter lambda = {lam}:" )    
        print(f" * Precision@10           : {scores['precision_at_10']:.4f}")
        print(f" * Recall@10              : {scores['recall_at_10']:.4f}")
        print(f" * NDCG@10                : {scores['ndcg_at_10']:.4f}")
        print(f" * Intra-list diversity   : {ild(lists, S):.4f}")
        print(f" * Coverage               : {coverage(lists, n_items_tot):.4f}")
        print(f" * Gini coefficient       : {gini(lists, n_items):.4f}")
        print(f" * Calibration error      : {calibration_error(lists, user_hist_dist, G):.4f}")

    results_df = pd.DataFrame(rows).set_index("lambda")
    print("\n* MMR reranking : Results summary:")
    print(results_df.round(4))
    # plot accuracy vs diversity for lambda values over [0,1]:
    plot_acc_div_frontier(rel_m, S, n_users, K, plot_path)
    if plot_path is not None:
        print(f"\n* MMR reranking : Stored accuracy vs. diversity plot to path: {plot_path} / acc_div_frontier.png")
        
def mmr(rel_u, S, k, lam):
    ''' Calculate MMR for given value of trade-off parameter lambda (lam). '''
    cand = np.flatnonzero(~np.isnan(rel_u)).tolist()  # get candidate indices
    lo, hi = np.nanmin(rel_u), np.nanmax(rel_u)
    r = (rel_u - lo) / (hi - lo + 1e-12) # normalise to [0, 1]
    chosen = []
    # choose k recommendations by picking canditates with highest MMR one at time:
    for _ in range(min(k, len(cand))):
        # calculate maximum similarity to the already picked up items:
        if chosen:
            red = S[np.ix_(cand, chosen)].max(axis=1)
        else:
            red = np.zeros(len(cand))
        score = lam * r[cand] - (1 - lam) * red
        best = cand[int(np.argmax(score))]
        chosen.append(best)
        cand.remove(best)
    return chosen

def ild(lists, S):
    ''' Calculate average pairwise distance between items in input list (lists). '''
    out = []
    for l in lists:
        n = len(l)
        if n < 2:
            continue
        # mean distance over all non-identical item pairs:
        out.append((1 - S[np.ix_(l, l)]).sum() / (n * (n - 1)))
    return np.mean(out) # average over all users

def coverage(lists, n_items):
    return len(set(i for l in lists for i in l)) / n_items

def gini(lists, n_items):
    c = np.bincount(np.concatenate(lists), minlength=n_items).astype(float)
    c.sort()
    n = len(c)
    return (2 * np.arange(1, n + 1) - n - 1).dot(c) / (n * c.sum())

def calibration_error(lists, user_hist_dist, G, eps=1e-2):
    errs = []
    for u, l in enumerate(lists):
        p = user_hist_dist[u]   # rows must follow user_ids order
        q = G[l].sum(0).astype(float)
        q = q / q.sum()
        q = (1 - eps) * q + eps * p
        m = p > 0
        errs.append(np.sum(p[m] * np.log(p[m] / q[m])))
    return np.mean(errs)

def plot_acc_div_frontier(rel_m, S, n_users, K=10, plot_path=None):
    ''' calculate and plot accuracy vs diversity over a range of lambda values 
        (lambda = trade-off parameter). '''
    lams = np.linspace(0, 1, 11)
    acc, div = [], []
    for lam in lams:
        lists = [mmr(rel_m[u], S, K, lam) for u in range(n_users)]        
        acc.append(np.mean([rel_m[u, l].sum() for u, l in enumerate(lists)]))
        div.append(ild(lists, S))

    plt.plot(div, acc, "o-")
    for lam, x, y in zip(lams, div, acc):
        plt.annotate(f"{lam:.1f}", (x, y))
    plt.xlabel("Intra-list diversity"); plt.ylabel("Accuracy (sum of prediction scores @ 10, averaged over users)")
    plt.grid()
    if plot_path is not None:
        plt.savefig(plot_path / "acc_div_frontier.png")
        plt.close()
    else:
        plt.show()

def user_genre_distribution(train, G, item_ids, user_ids, weight_col=None):
    """
    Returns user_hist_dist: (n_users, n_genres), rows follow user_ids order,
    each row sums to 1 (all zeros if the user has no usable history).
    weight_col: optional column name (e.g. "Rating") to weight items;
                None = every interacted item counts 1.
    """
    item_index = {item: j for j, item in enumerate(item_ids)}
    user_index = {u: r for r, u in enumerate(user_ids)}

    # keep only users and items we have rows for
    df = train[train["UserID"].isin(user_index) & train["MovieID"].isin(item_index)]
    rows = df["UserID"].map(user_index).to_numpy()
    cols = df["MovieID"].map(item_index).to_numpy()
    w = np.ones(len(df)) if weight_col is None else df[weight_col].to_numpy(float)

    # sum of (weight * genre vector) over each user's items
    counts = np.zeros((len(user_ids), G.shape[1]))
    np.add.at(counts, rows, w[:, None] * G[cols])

    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

def genres_to_matrix(movies):
    """
    movies: DataFrame with columns "MovieId" and "Genres" (list of strings).
    Returns
      G         : (n_items, n_genres) binary numpy array
      item_ids  : (n_items,) array, row j of G belongs to item_ids[j]
      genre_names : list of genre names, column j of G belongs to genre_names[j]
    """
    mlb = MultiLabelBinarizer()
    G = mlb.fit_transform(movies["Genres"]).astype(float)
    item_ids = movies["MovieID"].to_numpy()
    return G, item_ids, list(mlb.classes_)

def dict_to_matrix(user_rel, item_ids):
    """Returns relevance matrix (n_users, n_items) with NaN where an item has no score, plus the user_ids in row order."""
    user_ids = sorted(user_rel)
    item_index = {item: j for j, item in enumerate(item_ids)}
    rel = np.full((len(user_ids), len(item_ids)), np.nan)
    for r, u in enumerate(user_ids):
        for item, score in user_rel[u].items():
            rel[r, item_index[item]] = score
    return rel, user_ids