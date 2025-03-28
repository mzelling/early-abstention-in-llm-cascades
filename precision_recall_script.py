import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import plot_precision_recall_curve
from early_abs_setup import setup_data, PRETTY_NAMES
import numpy as np
from itertools import product
import pickle

# Optionally write results to file "./precision_recall_data/pr-data.pkl"
# WARNING: this will overwrite any existing file with this path
SAVE_DATA_TO_FILE = False

ALL_PR_DATA = []

benchmarks = ["mmlu", "medmcqa", "gsm8k", "truthfulqa", "triviaqa", "xsum"]
cascades = ["llama", "openai", "qwen"]
total_iter = len(benchmarks) * len(cascades)

for iter_idx, (BENCHMARK_NAME, CASCADE) in enumerate(product(benchmarks, cascades)):
    print(f"Generating PR data for {CASCADE.upper()} on {BENCHMARK_NAME.upper()} ({iter_idx+1}/{total_iter})")
    chain_name = "qwen_oai_chain" if CASCADE in ['openai', 'qwen'] else "llama_chain"

    all_setup_data = setup_data(NAME=BENCHMARK_NAME, CHAIN_NAME=chain_name)
    results_train = all_setup_data['raw_results']['train']
    results_test = all_setup_data['raw_results']['test']

    transformed_conf_train = all_setup_data['transformed_conf']['train']
    transformed_conf_test = all_setup_data['transformed_conf']['test']
    corr_train = all_setup_data['corr']['train']
    corr_test = all_setup_data['corr']['test']

    # Process data into numpy array
    tf_conf_train_arr = np.array(transformed_conf_train).transpose()
    tf_conf_test_arr = np.array(transformed_conf_test).transpose()

    finite_maxima_train = np.nanmax(np.where(np.isfinite(tf_conf_train_arr), tf_conf_train_arr, np.nan), axis=0)
    finite_maxima_test = np.nanmax(np.where(np.isfinite(tf_conf_test_arr), tf_conf_test_arr, np.nan), axis=0)

    tf_conf_train_arr = np.where(np.isfinite(tf_conf_train_arr), tf_conf_train_arr, finite_maxima_train)
    tf_conf_test_arr = np.where(np.isfinite(tf_conf_test_arr), tf_conf_test_arr, finite_maxima_test)

    corr_train_arr = np.array(corr_train).transpose()
    corr_test_arr = np.array(corr_test).transpose()


    ### STEP 1: train logistic regression model to calibrate last model's confidence

    if CASCADE=="qwen":
        final_model_idx = 2
        X_train = tf_conf_train_arr[:,[final_model_idx]]
        X_test = tf_conf_test_arr[:,[final_model_idx]]
        y_train = corr_train_arr[:,final_model_idx]
    elif (CASCADE=="openai") or (CASCADE=="llama"):
        # final model is last
        X_train = tf_conf_train_arr[:,[-1]]
        X_test = tf_conf_test_arr[:,[-1]]
        y_train = corr_train_arr[:,-1]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_predprob = model.predict_proba(X_train)[:,1]
    y_pred = model.predict(X_train)

    # Examine last model's correctness prediction performance
    final_model_corr_pred_perf = classification_report(y_train, y_pred, target_names=["Incorrect", "Correct"], output_dict=True)

    # Obtain final model's calibrated confidence on test data
    y_predprob_test = model.predict_proba(X_test)[:,1]


    ### STEP 2: predict last model's abstention decision using earlier models' confidences

    for abs_rate_bottom in [0.2, 0.3]:
        conf_threshold = np.quantile(y_predprob, q=abs_rate_bottom)
        y_last_model_abstains = (y_predprob < conf_threshold)

        if (CASCADE=="llama" or CASCADE=="openai"):
            X_last_model_abstains = tf_conf_train_arr[:,:-1]
            X_last_model_abstains_test = tf_conf_test_arr[:,:-1]
        elif (CASCADE=="qwen"):
            # 1 is the index for the small Qwen model (Qwen 32B Coder)
            X_last_model_abstains = tf_conf_train_arr[:,[1]]
            X_last_model_abstains_test = tf_conf_test_arr[:,[1]]

        model_last_model_abstains = LogisticRegression(max_iter=1000).fit(
            X=X_last_model_abstains, y=y_last_model_abstains
        )

        # Apply last model's abstention policy
        y_last_model_abstains_test = (y_predprob_test <= conf_threshold)
        y_last_model_abstains_pred = model_last_model_abstains.predict(X=X_last_model_abstains_test)
        precision, recall, thresholds = plot_precision_recall_curve(model_last_model_abstains, X_last_model_abstains_test, y_last_model_abstains_test)

        # Gather precision-recall data
        pr_data = {
            "cascade": CASCADE,
            "benchmark": BENCHMARK_NAME,
            "benchmark_pretty_name": PRETTY_NAMES[BENCHMARK_NAME],
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "abs_rate_bottom": abs_rate_bottom,
            "abs_conf_threshold": conf_threshold,
            "final_model_corr_pred_perf": final_model_corr_pred_perf
        }

        ALL_PR_DATA.append(pr_data)

# Write precision-recall data to file
if SAVE_DATA_TO_FILE:
    with open("./precision_recall_data/pr-data.pkl", "wb") as file:
        pickle.dump(ALL_PR_DATA, file)