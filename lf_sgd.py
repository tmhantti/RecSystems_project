import sys
from itertools import product
import warnings

from cornac.models import MF
from cornac.metrics import Precision, Recall, NDCG
from cornac.eval_methods import BaseMethod
import pandas as pd

from utils import RAND_STATES, calc_metrics_stats

def run_lf_sgd(split):
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
            # rating_threshold=4.0, # threshold to decide positive or negative preferences
            exclude_unknowns=True,
    )

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

    # ------------ Hyperparameter evaluation ------------
    
    # evaluation is done against validation data set:
    eval_method_val = BaseMethod.from_splits(
        train_data=conv_to_list(split.train),
        test_data=conv_to_list(split.validation),
        exclude_unknowns=True,
    )
    # set hyperparameter grid:
    # NOTE: number of combinations fixed to 20
    combos = {
        "k" : [10, 25, 50, 75, 100], # dimension of latent factors
        "lr" : [1e-2, 5e-3],       # learning rate
        "lambda_reg": [1e-2, 1e-3] # regularization parameter
    }

    param_combos = [
        dict(zip(combos.keys(), combination))
        for combination in product(*combos.values())
    ]
    print("* Latent factor CF: Hyperparameter search - this may take some tens of minutes... \n")
    res = pd.DataFrame()
    # loop over hyperparameter combinations:
    for c in param_combos:
        res_df = eval_model(
            k=c["k"], 
            learning_rate=c["lr"],
            lambda_reg=c["lambda_reg"],
            metrics=metrics,
            eval_method=eval_method_val,
        )
        _, stats_df = calc_metrics_stats(
            res_df,
            metrics=summ_metrics,
            confidence=0.95
        )
        stats_df["k"] = c["k"]
        stats_df["lr"] = c["lr"]
        stats_df["lambda"] = c["lambda_reg"] 
        res= pd.concat([res, stats_df], ignore_index=True)
        print(f" Results for k= {c["k"]}, lambda = {c["lambda_reg"]}, learning rate = {c["lr"]}:")
        print(stats_df)
        print()
    #print("\n* Latent factor CF: Hyperparameter search, results: \n")
    #print(res)
    # get the optimal parameter combination
    # (ranked by mean value of NDCG@10)
    tmp = res[res["metric"] == "NDCG@10"].sort_values(by=["mean"], ascending=False)
    k_opt = tmp["k"].iloc[0]
    lr_opt = tmp["lr"].iloc[0] 
    lambda_opt = tmp["lambda"].iloc[0]
    best_res = res[(res["k"] == k_opt) & (res["lr"] == lr_opt) & (res["lambda"] == lambda_opt)]
    print("\n* Latent factor CF: Hyperparameter search, case with the largest NDCG@10:\n")
    print(best_res)

    # ------------ Baseline ---------------
        
    # evaluation done against test data set
    base_res_df = eval_model(
        k=10, 
        learning_rate=1e-2,
        lambda_reg=1e-3,
        metrics=metrics,
        eval_method=eval_method,
    )
    summary, summ_df = calc_metrics_stats(
        base_res_df,
        metrics=summ_metrics,
        confidence=0.95
    )
    print("* Latent factor CF: Baseline results (against test data):\n")
    print(summ_df)
    print()

    # ------------ Optimal case ---------
    # uses optimal hyperparameter values
    
    # evaluation done against test data set
    opt_res_df = eval_model(
        k=k_opt, 
        learning_rate=lr_opt,
        lambda_reg=lambda_opt,
        metrics=metrics,
        eval_method=eval_method,
    )
    summary, summ_df = calc_metrics_stats(
        opt_res_df,
        metrics=summ_metrics,
        confidence=0.95
    )
    print("* Latent factor CF: Tuned model results (against test data):\n")
    print(summ_df)
    print()

    # check_unknowns(split.train, split.test)

def eval_model(k, learning_rate, lambda_reg, metrics, eval_method):
    ''' Evaluate model over a set of seed values for fixed hyperparameters.
        Returns summary of results for each seed as a df.  
    '''
    # suppress refitting warnings:
    warnings.filterwarnings(
        "ignore",
        module=r"cornac\..*"
    )
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
    return pd.DataFrame(res)

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