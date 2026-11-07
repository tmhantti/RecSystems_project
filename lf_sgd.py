import sys
from itertools import product
import warnings

from cornac.models import MF
from cornac.metrics import Precision, Recall, NDCG
from cornac.eval_methods import BaseMethod
import pandas as pd

from utils import (RAND_STATES, calc_metrics_stats, split_by_popularity, split_by_user_activity,
                   subsample)

def run_lf_sgd(split, ratings):
    '''Collaboratorive filtering using latent factors with stochastic gradient search for optimal solution.
       Implemented using Cornac library, see:
       https://cornac.readthedocs.io/en/v3.1.0/, and
       https://cornac.readthedocs.io/en/latest/api_ref/models.html#module-cornac.models.mf.recom_mf 
    '''

    # create evaluation against test data:
    # NOTE: exclude_unknowns = True => 
    #       users/items appearing in test but unseen in training are excluded from evaluation.
    #       This can be checked with routine check_unknowns (see above).    
    eval_method = BaseMethod.from_splits(
            train_data=conv_to_list(split.train),
            test_data=conv_to_list(split.test),
            rating_threshold=4.0, # threshold to decide positive or negative preferences
            exclude_unknowns=True,
    )



    # skip hyperparameter evaluation and use previously found optimal hyperparameter values?
    skip_eval = False

    if not skip_eval:
        # ------------ Hyperparameter evaluation -------------------------------
        # hyperparameter evaluation is done against validation data set:
        eval_method_val = BaseMethod.from_splits(
            train_data=conv_to_list(split.train),
            test_data=conv_to_list(split.validation),
            rating_threshold=4.0,
            exclude_unknowns=True,
        )
        # set hyperparameter grid:
        # NOTE: number of combinations should add up to 20 (as in Part I)
        combos = {
            "k" : [10, 25, 50, 75, 100], # dimension of latent factors
            "lr" : [1e-2, 5e-3],         # learning rate
            "lambda_reg": [1e-2, 1e-3]   # regularization parameter
        }

        param_combos = [
            dict(zip(combos.keys(), combination))
            for combination in product(*combos.values())
        ]
        print("* Latent factor CF: Hyperparameter search - this may take some tens of minutes... \n")
        res = pd.DataFrame()
        # loop over hyperparameter combinations:
        for i, c in enumerate(param_combos):
            varied_params = {"k": c["k"],
                             "lr": c["lr"],
                             "lambda": c["lambda_reg"]
                             }
            _, stats_df = eval_model(varied_params, eval_method_val)
            stats_df["k"], stats_df["lr"], stats_df["lambda"]  = c["k"], c["lr"], c["lambda_reg"] 
            res= pd.concat([res, stats_df], ignore_index=True)
            print(f" Parameter combination {i+1} / 20, results:")
            print(f"{stats_df}")

        # get the optimal parameter combination (ranked by mean value of NDCG@10)
        tmp = res[res["metric"] == "NDCG@10"].sort_values(by=["mean"], ascending=False)
        k_opt = tmp["k"].iloc[0]
        lr_opt = tmp["lr"].iloc[0] 
        lambda_opt = tmp["lambda"].iloc[0]
        best_res = res[(res["k"] == k_opt) & (res["lr"] == lr_opt) & (res["lambda"] == lambda_opt)]
        print("\n* Latent factor CF: Hyperparameter search finished - optimal parameter values:\n" 
              f"  k = {k_opt}, lambda = {lambda_opt}, learning rate = {lr_opt}")
        opt_params = {"k": k_opt, "lr": lr_opt, "lambda": lambda_opt}
        print("\n* Latent factor CF: Hyperparameter search finished - error metrics for case with the largest NDCG@10:\n")
        print(best_res)

        # ------------ Baseline (using default parameter values) ---------------
        default_params = {"k": 10, "lr": 1e-2,"lambda": 1e-3}
        _, summ_df = eval_model(default_params, eval_method) # evaluation done against test data set
        print("\n* Latent factor CF: Baseline with default parameter values (against test data):\n")
        print(f"{summ_df}\n")

        # ------------ Optimal case --------------------------------------------
        # uses optimal hyperparameter values
        _, summ_df = eval_model(opt_params, eval_method) # evaluation done against test data set
        print("* Latent factor CF: Tuned model results (against test data):\n")
        print(f"{summ_df}\n")
    else: 
        # ---------- Hyperparameter evaluation skipped --------------
        # use tuned hyperparameter values:
        k_opt = 25
        lambda_opt = 0.001 
        lr_opt = 0.005
        opt_params = {"k": k_opt, "lr": lr_opt, "lambda": lambda_opt}
        
    # -------------- Segregated data: high / low activity users ------
    high_act_ds, low_act_ds = split_by_user_activity(split, ratings)
    eval_method_high = BaseMethod.from_splits(
        train_data=conv_to_list(split.train),
        test_data=conv_to_list(high_act_ds.test),
        rating_threshold=4.0,
        exclude_unknowns=True,
    )
    eval_method_low = BaseMethod.from_splits(
        train_data=conv_to_list(split.train),
        test_data=conv_to_list(low_act_ds.test),
        rating_threshold=4.0,
        exclude_unknowns=True,
    )
    _, summ_df_high = eval_model(opt_params, eval_method_high)
    _, summ_df_low = eval_model(opt_params, eval_method_low)
    
    print("* Latent factor CF, data segregation - Tuned model results (against test data with high-active users):\n")
    print(f"{summ_df_high}\n")
    print("* Latent factor CF, data segregation - Tuned model results (against test data with low-active users):\n")
    print(f"{summ_df_low}\n")

    # -------------- Segregated data: popular / long-tail items -------------------
    pop_ds, tail_ds = split_by_popularity(split, ratings)
    eval_method_pop = BaseMethod.from_splits(
        train_data=conv_to_list(split.train),
        test_data=conv_to_list(pop_ds.test),
        rating_threshold=4.0,
        exclude_unknowns=True,
    )
    eval_method_tail = BaseMethod.from_splits(
        train_data=conv_to_list(split.train),
        test_data=conv_to_list(tail_ds.test),
        rating_threshold=4.0,
        exclude_unknowns=True,
    )
    _, summ_df_pop = eval_model(opt_params, eval_method_pop)
    _, summ_df_tail = eval_model(opt_params, eval_method_tail)

    print("* Latent factor CF, data segregation - Tuned model results (against test data with popular items):\n")
    print(f"{summ_df_pop}\n")
    print("* Latent factor CF, data segregation - Tuned model results (against test data with long-tail items):\n")
    print(f"{summ_df_tail}\n")

    # -------------- Subsampling training data -------------------------------------
    sample_ratios = [0.5, 0.2]
    for i, ratio in enumerate(sample_ratios):
        ds_sparse = subsample(split, ratio, i) # use index i as random state
        eval_method_sparse = BaseMethod.from_splits(
            train_data=conv_to_list(ds_sparse.train),
            test_data=conv_to_list(split.test),
            rating_threshold=4.0,
            exclude_unknowns=True,
        )
        _, summ_df = eval_model(opt_params, eval_method_sparse)
        print(f"* Latent factor CF, subsampling - Tuned model results (training data pruned by {100*ratio}%, evaluation against test data):\n")
        print(f"{summ_df}\n")

    # check_unknowns(split.train, split.test)

def eval_model(varied_params, eval_method):
    ''' Evaluate model over a set of seed values for fixed hyperparameters.
        Returns summary of results for each seed as a df.  
    '''
    # set hyperparameter values:
    k = varied_params["k"]
    learning_rate = varied_params["lr"]
    lambda_reg = varied_params["lambda"]
    # Cornac ranking metrics:
    metrics = [
        Precision(k=10),
        Recall(k=10),
        NDCG(k=10)
    ]
    # summarized metrics:
    summ_metrics=[
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "coverage"
    ]
    # suppress refitting warnings:
    warnings.filterwarnings(
        "ignore",
        module=r"cornac\..*"
    )
    # loop over random seed values:
    res = []
    for seed in RAND_STATES:
        # init MF model:
        model = MF(   
            k=k,
            learning_rate=learning_rate,
            lambda_reg=lambda_reg,
            use_bias=True,
            seed=seed
        )
        # fit model and evaluate against test data:
        model.fit(eval_method.train_set)
        test_result, user_results = eval_method.evaluate(
            model=model,
            metrics=metrics,
            user_based=True
        )
        # calculate error metrics + coverage:
        precision = test_result.metric_avg_results["Precision@10"]
        recall    = test_result.metric_avg_results["Recall@10"]
        ndcg      = test_result.metric_avg_results["NDCG@10"]
        coverage = calc_coverage(
            model=model,
            train_set=eval_method.train_set,
            k=10
        )
        # store results:
        res.append({
            "k": k,
            "learning_rate": learning_rate,
            "lambda_reg": lambda_reg,
            "seed": seed,
            "Precision@10": precision,
            "Recall@10": recall,
            "NDCG@10": ndcg,
            "coverage": coverage
        })
    # calculate error metrics:
    res = pd.DataFrame(res)
    _, stats_df = calc_metrics_stats(
        res,
        metrics=summ_metrics,
        confidence=0.95
    )

    return pd.DataFrame(res), stats_df

def conv_to_list(df):
    ''' convert training/test/validation data set to list. '''

    as_list = list(
        df[["UserID", "MovieID", "Rating"]]
        .itertuples(index=False, name=None)
    )

    return as_list

def check_unknowns(df_train, df_test):
    ''' Check if there are users/items in training data set that are not in test data set. '''
    train_users = set(df_train["UserID"])
    test_users  = set(df_test["UserID"])
    train_items = set(df_train["MovieID"])
    test_items  = set(df_test["MovieID"])

    unknown_users = test_users - train_users
    unknown_items = test_items - train_items

    print("Unknown test users:", len(unknown_users))
    print("Unknown test items:", len(unknown_items))

def calc_coverage(model, train_set, k=10):
    """Calculate catalogue coverage@k across all known users."""
    recommended_items = set()

    user_ids = train_set.uid_map.keys()
    for user_id in user_ids:
        recs = model.recommend(
            user_id=user_id,
            k=k,
            remove_seen=True,
            train_set=train_set
        )
        recommended_items.update(recs)

    n_items = train_set.num_items
    coverage = len(recommended_items) / n_items

    return coverage