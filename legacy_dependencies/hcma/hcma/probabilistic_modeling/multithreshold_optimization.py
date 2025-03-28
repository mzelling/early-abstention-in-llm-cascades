from probabilistic_modeling.confidence_model import MarkovConfidenceModel
from scipy.optimize import minimize
from probabilistic_modeling.confidence_model import unflatten_thresholds, predict_metrics
import numpy as np

def make_objective(joint_model, costs, penalties=[1,0,0]):
    return lambda flattened_thresholds: np.sum( 
        np.array(penalties) 
        * np.array(predict_metrics(joint_model, unflatten_thresholds(flattened_thresholds), costs=costs))
    )

def make_constraints(n_models):
    constraint_funs = [ 
        lambda params: params[(2*i)+1]-params[(2*i)] for i in range(n_models-1) 
    ]
    return [
        {
            'type': 'ineq',
            'fun': fun
        } for fun in constraint_funs
    ]

def get_initial_thresholds_and_bounds(joint_model):
    bound_eps = 0.0
    unique_bounds = (
        [[joint_model.base_marginal.p_min - bound_eps, joint_model.base_marginal.p_max + bound_eps]] 
        + [ [bim.joint_distribution.marginals[1].p_min - bound_eps, bim.joint_distribution.marginals[1].p_max + bound_eps] 
            for bim in joint_model.bivariate_models ]
    )
    bounds = [ unique_bounds[doubled_idx // 2] for doubled_idx in range(2*len(unique_bounds) - 1) ]
    x0 = [ np.mean(bd) for bd in bounds ]
    # set initial rejection rates to be zero
    # for i in range(len(x0)//2):
    #     x0[2*i] = 0.0
    return (x0, bounds)


def get_expected_costs(model_costs, model_tokens, ref_costs=False):
    """ Get effective cost per million tokens, for each model. """
    if ref_costs:
        total_costs = { v: sum([ c for c in model_costs[v] ]) for v in model_costs.keys() }
        total_tokens = { v: sum([ t['in'] + t['out'] for t in model_tokens[v] ]) for v in model_tokens.keys() }
        return { v: (1e+6)*total_costs[v]/total_tokens[v] for v in model_costs.keys() }
    else:
        # just output the actual expected cost per model, in $
        return { 
            v: sum(model_costs[v])/len(model_costs[v]) for v in model_costs.keys() 
        }


def optimize_thresholds(
        chain, calibration_data_train, model_costs, model_tokens,
        lambda_c_grid = np.linspace(0,1,10), lambda_r_grid = np.linspace(0,1,10)
    ):
    calibrated_confidence = chain.compute_calibrated_confidence(calibration_data_train)
    joint_model = MarkovConfidenceModel(chain_length=len(chain.models))
    joint_model.fit(calibrated_confidence)

    constraints = make_constraints(len(chain.models))
    x0, bounds = get_initial_thresholds_and_bounds(joint_model)

    costs_per_model = get_expected_costs(model_costs, model_tokens)
    costs = [ costs_per_model[v] for v in chain.model_names ]

    rows = []
    for row_idx, rej_penalty in enumerate(lambda_r_grid):
        rows.append([])
        print(f"Starting row {row_idx}")
        for col_idx, cost_penalty in enumerate(lambda_c_grid):
            penalties = [1, rej_penalty, cost_penalty]

            result = minimize(
                make_objective(joint_model, costs=costs, penalties=penalties),
                x0=x0,
                bounds=bounds,
                constraints=constraints
            )
            rows[row_idx].append(result)
            print(f"Completed {row_idx+1},{col_idx+1}")

    return rows