from dotenv import load_dotenv

load_dotenv()

import niagara
from early_abs_setup import setup_data, compute_threshold_grids
import numpy as np
import os
import pickle
from niagara.probabilistic_modeling.optimize_cascade import get_expected_uncumulated_costs, \
    optimize_cascade_thresholds_w_abstention, compute_metrics, \
    get_TS_from_X, loss_fn, make_full_data, score_cascade, train_probability_model
from tqdm import tqdm

CASCADES = [
    ("llama_chain", [0,1]),
    ("llama_chain", [0,2]),
    ("llama_chain", [0,3]),
    ("llama_chain", [0,4]),
    ("llama_chain", [1,2]),
    ("llama_chain", [1,3]),
    ("llama_chain", [1,4]),
    ("llama_chain", [2,3]),
    ("llama_chain", [2,4]),
    ("llama_chain", [3,4]),
    ("qwen_oai_chain", [0,1]),
    ("qwen_oai_chain", [0,2]),
    ("qwen_oai_chain", [0,3]),
    ("qwen_oai_chain", [1,2]),
    ("qwen_oai_chain", [1,3]),
    ("qwen_oai_chain", [2,3])
]

BENCHMARKS = [
    "mmlu",
    "medmcqa",
    "triviaqa",
    "gsm8k",
    "xsum",
    "truthfulqa"
]

ERROR_TYPE = 'joint'

for benchmark_name in tqdm(BENCHMARKS):
    print(f"Starting on {benchmark_name}...")
    for cascade in CASCADES:
        chain_name, model_indices = cascade

        model_idx_str = "".join([str(x) for x in model_indices])
        print(f"Running cascade {chain_name}-{model_idx_str}...")

        all_setup_data = setup_data(NAME=benchmark_name, CHAIN_NAME=chain_name)

        # Raw costs per million tokens
        raw_model_costs = { 
            model_name: all_setup_data['chain'].models[i].cpm_tokens 
                for i, model_name in enumerate(all_setup_data['chain'].model_names) 
        }
        
        results_train = all_setup_data['raw_results']['train']
        results_test = all_setup_data['raw_results']['test']
        expected_uncumulated_costs_train = get_expected_uncumulated_costs(raw_model_costs, results_train)
        expected_uncumulated_costs_test = get_expected_uncumulated_costs(raw_model_costs, results_test)

        ### Get confidence and correctness data

        calibrated_conf_train = all_setup_data['calibrated_conf']['train']
        calibrated_conf_test = all_setup_data['calibrated_conf']['test']
        corr_train = all_setup_data['corr']['train']
        corr_test = all_setup_data['corr']['test']

        # Set grid of user preferences
        lambda_abs_grid = np.linspace(1e-3,1,41)
        lambda_cost_grid = np.linspace(1e-7,3e-4,41)

        ### Train probabilistic models ###

        PROB_MODEL_RESULTS_FILENAME = f"./optimal_thresholds_data/{benchmark_name}/prob_model_results_{chain_name}.pkl"

        if os.path.exists(PROB_MODEL_RESULTS_FILENAME):
            with open(PROB_MODEL_RESULTS_FILENAME, "rb") as file:
                prob_model_results = pickle.load(file)
        else:
            full_data = np.array(calibrated_conf_train).transpose()
            prob_model_results = train_probability_model(full_data)

            with open(PROB_MODEL_RESULTS_FILENAME, "wb") as file:
                pickle.dump(prob_model_results, file)
    
        ### Compute optimal thresholds ###

        CASCADE_OPTIMAL_THOLDS_RESULTS_FILENAME = f"./optimal_thresholds_data/{benchmark_name}/optimal_tholds_{chain_name}_{model_idx_str}.pkl"

        if not os.path.exists(CASCADE_OPTIMAL_THOLDS_RESULTS_FILENAME):
            print("Computing optimal thresholds w/ early abstention...")

            T_2d, S_2d = compute_threshold_grids(
                model_indices,
                expected_uncumulated_costs_train,
                prob_model_results,
                lambda_abs_grid,
                lambda_cost_grid,
                only_allow_abstention_at_last_model=False,
                error_type=ERROR_TYPE
            )

            print("Computing optimal thresholds w/ final-model abstention...")

            T_2d_no, S_2d_no = compute_threshold_grids(
                model_indices,
                expected_uncumulated_costs_train,
                prob_model_results,
                lambda_abs_grid,
                lambda_cost_grid,
                only_allow_abstention_at_last_model=True,
                error_type=ERROR_TYPE
            )

            optimal_tholds = {
                "early_abs": {
                    "T": T_2d,
                    "S": S_2d
                },
                "final_model_abs": {
                    "T": T_2d_no,
                    "S": S_2d_no
                },
                "error_type": ERROR_TYPE,
                "lambda_abs_grid": lambda_abs_grid,
                "lambda_cost_grid": lambda_cost_grid
            }
            
            with open(CASCADE_OPTIMAL_THOLDS_RESULTS_FILENAME, "wb") as file:
                pickle.dump(optimal_tholds, file)