from dotenv import load_dotenv
import os

load_dotenv()

os.environ['FIREWORKS_API_KEY'] = 'NO_NEED_TO_PROVIDE_REAL_API_KEY_BUT_LEAVE_THIS_LINE'
os.environ['OPENAI_API_KEY'] = 'NO_NEED_TO_PROVIDE_REAL_API_KEY_BUT_LEAVE_THIS_LINE'

import niagara
from niagara import Chain, Model, ModelIntrinsicLogProb, NullTransformation, LogisticRegressionCalibrator
from niagara import OpenAIClient, FireworksClient

from niagara.probabilistic_modeling.optimize_cascade import make_full_data, score_cascade, \
    get_TS_from_X, optimize_cascade_thresholds_w_abstention
import numpy as np

import pickle
from niagara.confidence_transformations import OneSidedAsymptoticLog, TwoSidedAsymptoticLog

llama_chain = Chain(
    models = [
        Model(
            model_name=name, 
            thresholds={"reject": -10000, "accept": 0.0},
            conf_signal=ModelIntrinsicLogProb(),
            conf_signal_transform=NullTransformation(),
            conf_signal_calibrator=LogisticRegressionCalibrator(),
            client=FireworksClient()
        )
        for name in ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b"]
    ]
)

qwen_oai_chain = Chain(
    models = [
        Model(
            model_name=name, 
            thresholds={"reject": -10000, "accept": 0.0},
            conf_signal=ModelIntrinsicLogProb(),
            conf_signal_transform=NullTransformation(),
            conf_signal_calibrator=LogisticRegressionCalibrator(),
            client=client,
        )
        for name, client in [("gpt-4o-mini", OpenAIClient()), ("qwen2.5-32b-coder-instruct", FireworksClient()), ("qwen2.5-72b-instruct", FireworksClient()), ("gpt-4o", OpenAIClient())]
    ]
)

# Pretty names to use in plots etc.
PRETTY_NAMES = {
    "mmlu": "MMLU",
    "medmcqa": "MedMCQA",
    "triviaqa": "TriviaQA",
    "xsum": "XSum",
    "gsm8k": "GSM8K",
    "truthfulqa": "TruthfulQA",
}

def setup_data(NAME="mmlu", CHAIN_NAME="qwen_oai_chain"):
    if NAME in ['mmlu', 'medmcqa']:
        TRANSFORM = OneSidedAsymptoticLog()
    elif NAME in ['triviaqa', 'xsum', 'gsm8k', 'truthfulqa']:
        TRANSFORM = TwoSidedAsymptoticLog()
    if CHAIN_NAME == "qwen_oai_chain":
        CHAIN = qwen_oai_chain
    elif CHAIN_NAME == "llama_chain":
        CHAIN = llama_chain

    # Update the transformation for the chain
    for model in CHAIN.models:
        model.conf_signal_transform = TRANSFORM

    with open(f'benchmarks/data/{NAME}/chain_results/{NAME}_full_{CHAIN_NAME}_results_train.pkl', 'rb') as f:
        results_train = pickle.load(f)
    with open(f'benchmarks/data/{NAME}/chain_results/{NAME}_full_{CHAIN_NAME}_results_test.pkl', 'rb') as f:
        results_test = pickle.load(f)

    ### Compute calibrated confidence values

    process_scores = lambda scores: sum(scores.values()) >= 20

    if NAME=="xsum":
        raw_corr_train = { k: [process_scores(x) for x in v] for k,v in results_train['model_correctness'].items() }
    else:
        raw_corr_train= results_train['model_correctness']

    raw_conf_train = results_train['raw_confidences']

    corr_train = [
        raw_corr_train[model_name] for model_name in CHAIN.model_names
    ]

    transformed_conf_train = [ 
        list(TRANSFORM.transform_confidence_signal(raw_conf_train[model_name]))
            for model_name in CHAIN.model_names
    ]

    calibration_data = [
        {"correctness": corr, "transformed_confidence": conf} 
            for (corr, conf, model_name) 
                in zip(corr_train, transformed_conf_train, CHAIN.model_names)
    ]

    CHAIN.calibrate(calibration_data)

    ### Get p values of log reg for all models

    pvalues_list = []
    llr_pvalues = []

    for i in range(len(CHAIN.models)):
        pvalues = CHAIN.models[i].conf_signal_calibrator.logreg.pvalues
        llr_pvalue = CHAIN.models[i].conf_signal_calibrator.logreg.llr_pvalue
        pvalues_list.append(pvalues)
        llr_pvalues.append(llr_pvalue)

    fit_stats = {'pvalues_list': pvalues_list, 'llr_pvalues': llr_pvalues}

    ###

    calibrated_conf_train = [
        list(
            CHAIN.models[model_idx].conf_signal_calibrator.calibrate_confidence_signal(
                transformed_conf_train[model_idx]
            )
        )
        for model_idx in range(len(CHAIN.model_names))
    ]

    ### Compute test data

    if NAME=="xsum":
        raw_corr_test = { k: [process_scores(x) for x in v] for k,v in results_test['model_correctness'].items() }
    else:
        raw_corr_test= results_test['model_correctness']

    raw_conf_test = results_test['raw_confidences']

    corr_test = [
        raw_corr_test[model_name] for model_name in CHAIN.model_names
    ]

    transformed_conf_test = [ 
        list(TRANSFORM.transform_confidence_signal(raw_conf_test[model_name]))
            for model_name in CHAIN.model_names
    ]

    calibrated_conf_test = [
        list(
            CHAIN.models[model_idx].conf_signal_calibrator.calibrate_confidence_signal(
                transformed_conf_test[model_idx]
            )
        )
        for model_idx in range(len(CHAIN.model_names))
    ]

    # return the required data:
    results_train, results_test
    calibrated_conf_test, calibrated_conf_train
    corr_train, corr_test
    return {
        "chain": CHAIN, 
        "raw_results": {"train": results_train, "test": results_test},
        "transformed_conf": {
            "train": transformed_conf_train,
            "test": transformed_conf_test,
        },
        "calibrated_conf": {
                "train": calibrated_conf_train,
                "test": calibrated_conf_test
        },
        "corr": {
            "train": corr_train,
            "test": corr_test
        },
        "logreg_fit_stats": fit_stats
    }


def compute_threshold_grids(
    model_indices,
    expected_uncumulated_costs,
    prob_model_results,
    lambda_abs_grid,
    lambda_cost_grid,
    eps=1e-6,
    only_allow_abstention_at_last_model=False,
    error_type='conditional'
):
    """
    Compute 2D grids of optimal thresholds for all combinations of lambda_abs and lambda_cost.
    
    Args:
        model_indices: List of model indices in the cascade
        expected_uncumulated_costs: List of costs for each model
        prob_model_results: Probability predictions from each model
        lambda_abs_grid: List of lambda_abs values
        lambda_cost_grid: List of lambda_cost values
        eps: Epsilon for optimization
        
    Returns:
        tuple: (T_2d, S_2d) where each is a 2D list of optimal thresholds.
        T_2d[i][j] corresponds to lambda_abs_grid[i], lambda_cost_grid[j]
    """
    n_abs = len(lambda_abs_grid)
    n_cost = len(lambda_cost_grid)
    
    # Initialize 2D lists
    T_2d = [[None for _ in range(n_cost)] for _ in range(n_abs)]
    S_2d = [[None for _ in range(n_cost)] for _ in range(n_abs)]
    
    # Double loop over lambda values
    for i, lambda_abs in enumerate(lambda_abs_grid):
        for j, lambda_cost in enumerate(lambda_cost_grid):
            # Optimize with current parameters
            optim_result = optimize_cascade_thresholds_w_abstention(
                model_indices=model_indices,
                cost_sensitivity=lambda_cost,
                abstention_sensitivity=lambda_abs,
                expected_uncumulated_costs=expected_uncumulated_costs,
                prob_models=prob_model_results,
                eps=eps,
                only_allow_abstention_at_last_model=only_allow_abstention_at_last_model,
                error_type=error_type
            )
            
            # Extract optimal thresholds
            T, S = get_TS_from_X(optim_result.x, only_allow_abstention_at_last_model=only_allow_abstention_at_last_model)
            
            # Store in 2D grids
            T_2d[i][j] = T
            S_2d[i][j] = S
    
    return T_2d, S_2d