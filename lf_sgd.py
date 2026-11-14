import sys
from itertools import product
import warnings

from cornac.models import MF, BaselineOnly
from cornac.metrics import Precision, Recall, NDCG
from cornac.eval_methods import BaseMethod
import pandas as pd
import numpy as np
import copy

from utils import (RAND_STATES, calc_metrics_stats, split_by_popularity, split_by_user_activity,
                   subsample)

def run_lf_sgd(split, ratings):
    '''Collaboratorive filtering using latent factors with stochastic gradient search for optimal solution.
       Implemented using Cornac library, see:
       https://cornac.readthedocs.io/en/v3.1.0/, and
       https://cornac.readthedocs.io/en/latest/api_ref/models.html#module-cornac.models.mf.recom_mf 
    '''
    # skip hyperparameter evaluation and use previously found optimal hyperparameter values?
    skip_eval = False

    if not skip_eval:
        # ------------ Hyperparameter evaluation -------------------------------

        # set hyperparameter grid for MF model:
        # NOTE: number of combinations should add up to 20 (as in Part I)
        combos = {
            "k" : [10, 25, 50, 75, 100], # dimension of latent factors
            "lr" : [1e-2, 1e-3],         # learning rate
            "lambda_reg": [1e-2, 1e-3]   # regularization parameter
        }
        combos_MF = [
            dict(zip(combos.keys(), combination))
            for combination in product(*combos.values())
        ]
        # set hyperparameter grid for baseline model:
        # NOTE: the number of combinations should be the same as above
        combos = {
            "lr": [1e-3, 5e-3, 1e-2, 2e-2],
            "lambda_reg": [1e-3, 1e-2, 2e-2, 5e-2, 0.1]
        }
        combos_base = [
            dict(zip(combos.keys(), combination))
            for combination in product(*combos.values())
        ]
        
        # evaluation is done against validation data set:
        # NOTE: exclude_unknowns = True => 
        #       users/items appearing in test but unseen in training are excluded from evaluation.
        #       This can be checked with routine check_unknowns (see below).    
        eval_method_val = BaseMethod.from_splits(
            train_data=conv_to_list(split.train),
            test_data=conv_to_list(split.validation),
            rating_threshold=4.0,
            exclude_unknowns=True,
        )
        print("* Latent factor CF: Performing hyperparameter search - this may take some tens of minutes... \n")
        res_MF, res_base = pd.DataFrame(), pd.DataFrame()

        # loop over hyperparameter combinations for baseline model:
        for i, c in enumerate(combos_base):
            varied_params = {"lr": c["lr"], "lambda": c["lambda_reg"]}
            _, stats_df = eval_model(varied_params, eval_method_val, model_type="base")
            stats_df["lr"], stats_df["lambda"]  = c["lr"], c["lambda_reg"] 
            res_base= pd.concat([res_base, stats_df], ignore_index=True)
            print(f"  Baseline model: Parameter combination {i+1} / 20 evaluated")
            #print(f"{stats_df}")
        # get the optimal parameter combination (ranked by mean value of NDCG@10):
        opt_params_base = get_opt_params(res_base)
        print("\n* Latent factor CF: Hyperparameter search finished for baseline model - optimal parameter values:\n" 
              f"  lambda = {opt_params_base["lambda"]}, learning rate = {opt_params_base["lr"]}\n")
        #best_res_base = res_base[(res_base["lr"] == opt_params_base["lr"]) & 
        #                         (res_base["lambda"] == opt_params_base["lambda"])]
        #print("\n* Latent factor CF: Hyperparameter search finished for baseline - error metrics for case with the largest NDCG@10:\n")
        #print(best_res_base)

        # loop over hyperparameter combinations for MF model:
        for i, c in enumerate(combos_MF):
            varied_params = {"k": c["k"], "lr": c["lr"], "lambda": c["lambda_reg"]}
            _, stats_df = eval_model(varied_params, eval_method_val)
            stats_df["k"], stats_df["lr"], stats_df["lambda"]  = c["k"], c["lr"], c["lambda_reg"] 
            res_MF= pd.concat([res_MF, stats_df], ignore_index=True)
            print(f"  MF model: Parameter combination {i+1} / 20 evaluated")
            #print(f"{stats_df}")
        # get the optimal parameter combination (ranked by mean value of NDCG@10):
        opt_params_MF = get_opt_params(res_MF)
        print("\n* Latent factor CF: Hyperparameter search finished for MF model - optimal parameter values:\n" 
              f"  k = {opt_params_MF["k"]}, lambda = {opt_params_MF["lambda"]}, learning rate = {opt_params_MF["lr"]}\n")
        #best_res_MF = res_MF[(res_MF["k"] == opt_params_MF["k"]) & 
        #                  (res_MF["lr"] == opt_params_MF["lr"]) & 
        #                  (res_MF["lambda"] == opt_params_MF["lambda"])]
        #print("\n* Latent factor CF: Hyperparameter search finished - error metrics for model with the largest NDCG@10:\n")
        #print(best_res_MF)
    else: 
        # skip hyperparameter evaluation, use pre-calculated optimal hyperparameter values:
        opt_params_base = {"lr": 0.001, "lambda": 0.001}
        opt_params_MF = {"k": 25, "lr": 0.005, "lambda": 0.001 }

    # ------------ Tuned model vs baseline comparison ----------------
    eval_method_test = BaseMethod.from_splits(
            train_data=conv_to_list(split.train),
            test_data=conv_to_list(split.test),
            rating_threshold=4.0, 
            exclude_unknowns=True,
    )
    print("\n* Latent factor CF: Evaluating tuned baseline model (against test data)...\n")
    _, summ_df_opt = eval_model(opt_params_base, eval_method_test, model_type="base")
    print(f"{summ_df_opt}\n")
    print("* Latent factor CF: Evaluating tuned MF model (against test data)...\n")
    _, summ_df_opt = eval_model(opt_params_MF, eval_method_test) 
    print(f"{summ_df_opt}\n")

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
    
    print("* Latent factor CF: data segregation - Evaluating tuned MF model (against test data with high-active users)...\n")
    _, summ_df_high = eval_model(opt_params_MF, eval_method_high)
    print(f"{summ_df_high}\n")
    print("* Latent factor CF: data segregation - Evaluating tuned MF model (against test data with low-active users)...\n")
    _, summ_df_low = eval_model(opt_params_MF, eval_method_low)
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
    print("* Latent factor CF: data segregation - Evaluating tuned MF model (against test data with popular items)...\n")
    _, summ_df_pop = eval_model(opt_params_MF, eval_method_pop)
    print(f"{summ_df_pop}\n")
    print("* Latent factor CF: data segregation - Evaluating tuned MF model (against test data with long-tail items)...\n")
    _, summ_df_tail = eval_model(opt_params_MF, eval_method_tail)
    print(f"{summ_df_tail}\n")

    # -------------- Subsampling training data -------------------------------------
    sample_ratios = [0.5, 0.2] # remaining fraction of original sample size
    summ_df_sub = [None, None]
    for i, ratio in enumerate(sample_ratios):
        ds_sparse = subsample(split, 1-ratio, i) # use index i as random state
        eval_method_sparse = BaseMethod.from_splits(
            train_data=conv_to_list(ds_sparse.train),
            test_data=conv_to_list(split.test),
            rating_threshold=4.0,
            exclude_unknowns=True,
        )
        print(f"* Latent factor CF: subsampling - Evaluating tuned MF model (training data pruned by {100*(1-ratio)}%, evaluation against test data):\n")
        _, summ_df_sub[i] = eval_model(opt_params_MF, eval_method_sparse)
        print(f"{summ_df_sub[i]}\n")

    print("* Latent factor CF: Model calculations completed.\n")

    # check_unknowns(split.train, split.test)

def eval_model(varied_params, 
               eval_method, 
               model_type= "MF",
               metrics= [Precision(k=10), Recall(k=10), NDCG(k=10)], 
               summ_metrics=["Precision@10", "Recall@10", "NDCG@10", "coverage"]):
    ''' Evaluate latent factor CF model / baseline recommender over a set of seed values for fixed hyperparameters.
        For API documentation, see:
        https://cornac.readthedocs.io/en/stable/api_ref/models.html#module-cornac.models.mf.recom_mf
        For API documentation of the baseline method, see:
        https://cornac.readthedocs.io/en/stable/api_ref/models.html#baseline-only, and docstrings in:
        https://github.com/PreferredAI/cornac/blob/v3.1.0/cornac/models/baseline_only/recom_bo.pyx

        * Input: 
          - varied_params : None / dict containing varied hyperparameters and their values
          - eval_method : Cornac BaseMethod object containing test and training data
          - model_type : Matrix Factorization model ("MF"), or baseline model ("base")
          - metrics : Cornac evaluation metrics
          - summ_metrics : metrics columns in output dataframe
        * Output: 
          - res : list where an element contains results from a single model run 
          - stats_df : overall statistics of the model runs
    '''
    if model_type not in ["MF", "base"]: 
        raise ValueError("Input argument 'model_type' must be 'MF' (= Matrix Factorization) or 'base' (= baseline model).")
    # set hyperparameter values:
    if varied_params is not None:
        if "k" in varied_params:
            k = varied_params["k"]
        learning_rate = varied_params["lr"]
        lambda_reg = varied_params["lambda"]    
    # suppress refitting warnings:
    warnings.filterwarnings(
        "ignore",
        module=r"cornac\..*"
    )
    # loop over random seed values:
    res_all = []
    for seed in RAND_STATES:
        if model_type == "MF":
            # init MF model:
            # NOTE: fixing seed ensures single threaded execution
            model = MF(   
                k=k,
                learning_rate=learning_rate,
                lambda_reg=lambda_reg,
                use_bias=True,
                seed=seed,
            )
            eval_method_eval = copy.copy(eval_method)
            model.fit(eval_method_eval.train_set)
        elif model_type == "base":
            # init baseline model 
            # NOTE: uses single threaded execution to ensure reproducibility of the results
            if varied_params is None:
                model = BaselineOnly(num_threads=1)
            else:
                model = BaselineOnly(   
                    learning_rate=learning_rate,
                    lambda_reg=lambda_reg,
                    seed=seed,
                )
            eval_method_eval = copy.copy(eval_method)
            # convert training data types to avoid type errors:
            eval_method_eval.train_set = copy.copy(eval_method.train_set)
            rid, cid, val = eval_method_eval.train_set.uir_tuple
            eval_method_eval.train_set.uir_tuple = (
                rid.astype(np.int32),
                cid.astype(np.int32),
                val.astype(np.float32)
            )
        # evaluate model against test data:
        test_result, user_results = eval_method_eval.evaluate(
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
            train_set=eval_method_eval.train_set,
            k=10
        )
        # store results:
        res = {
            "Precision@10": precision,
            "Recall@10": recall,
            "NDCG@10": ndcg,
            "coverage": coverage
        }
        res_all.append(res)
        # baseline results do not depend on the seed value:
        if varied_params is None or model_type == "base": 
            #return pd.DataFrame(res_all), None
            break
    # calculate error metrics:
    res_all = pd.DataFrame(res_all)
    _, stats_df = calc_metrics_stats(
        res_all,
        metrics=summ_metrics,
        confidence=0.95
    )

    return res_all, stats_df

def get_opt_params(res):
    ''' Get optimal set of parameters from df containing results from a set of model runs. 
        Here models are ranked in terms of performance by metric NDCG@10. 
    '''
    tmp = res[res["metric"] == "NDCG@10"].sort_values(by=["mean"], ascending=False)
    lr_opt = tmp["lr"].iloc[0] 
    lambda_opt = tmp["lambda"].iloc[0]
    if "k" in tmp.columns:
        k_opt = tmp["k"].iloc[0]
    else:
        k_opt = None
    return {"k": k_opt, "lr": lr_opt, "lambda": lambda_opt}


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