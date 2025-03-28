# Set up the chain

import numpy as np
from hcma import Chain, Model, FireworksClient, OpenAIClient, ModelIntrinsicLogProb, LogisticRegressionCalibrator, OneSidedAsymptoticLog
import pickle

with open("../four_model_chain_results.pkl", "rb") as file:
    four_chain_results = pickle.load(file)

mychain = Chain(
    models=[
        Model(
            model_name="gpt-4o-mini",
            thresholds={"accept": 0.2, "reject": 0.8},
            conf_signal = ModelIntrinsicLogProb(),
            conf_signal_transform = OneSidedAsymptoticLog(),
            conf_signal_calibrator = LogisticRegressionCalibrator(),
            client=OpenAIClient()
        ),
        Model(
            model_name="llama3.1-70b",
            thresholds={"accept": 0.2, "reject": 0.8},
            conf_signal = ModelIntrinsicLogProb(),
            conf_signal_transform = OneSidedAsymptoticLog(),
            conf_signal_calibrator = LogisticRegressionCalibrator(),
            client=FireworksClient()
        ),
        Model(
            model_name="llama3.1-405b",
            thresholds={"accept": 0.2, "reject": 0.8},
            conf_signal = ModelIntrinsicLogProb(),
            conf_signal_transform = OneSidedAsymptoticLog(),
            conf_signal_calibrator = LogisticRegressionCalibrator(),
            client=FireworksClient()
        ),
        Model(
            model_name="gpt-4o",
            thresholds={"accept": 0.2, "reject": 0.8},
            conf_signal = ModelIntrinsicLogProb(),
            conf_signal_transform = OneSidedAsymptoticLog(),
            conf_signal_calibrator = LogisticRegressionCalibrator(),
            client=OpenAIClient()
        )
    ]
)

mychain.calibrate(four_chain_results['results'])
confvals = mychain.compute_calibrated_confidence(four_chain_results['results'])


# Fit the probabilistic models

from probabilistic_modeling.confidence_model import MarkovConfidenceModel

mm = MarkovConfidenceModel(chain_length=4)
mm.fit(confvals)

mm_sm = MarkovConfidenceModel(chain_length=3)
mm_sm.fit([confvals[0], confvals[1], confvals[2]])


# Tests for partial expectation

from scipy.stats import beta

assert np.abs( partial_expectation(beta(a=1, b=3), integration_interval=[0,1]) - (1/(1+3)) ) < 0.01
assert np.abs( partial_expectation(beta(a=10, b=1), integration_interval=[0,1]) - (10/(10+1)) ) < 0.01
assert np.abs( partial_expectation(beta(a=100, b=10), integration_interval=[0,1]) - (100/(100+10)) ) < 0.01
assert np.abs( partial_expectation(beta(a=1, b=100), integration_interval=[0,1]) - (1/(100)) ) < 0.001


### Test cases for expectation of lumpy betamix

from scipy.stats import beta
from probabilistic_modeling.marginals import beta_couple

lumpy_marginal = mm.base_marginal

betamix_expectation = (
    lumpy_marginal.p_min + 
    (lumpy_marginal.p_max - lumpy_marginal.p_min) *
    (
        lumpy_marginal.pi * (lumpy_marginal.alpha1 / (lumpy_marginal.alpha1 + lumpy_marginal.beta1))
        + (1-lumpy_marginal.pi) * (lumpy_marginal.alpha2 / (lumpy_marginal.alpha2 + lumpy_marginal.beta2))
    )
)

discrete_high = lumpy_marginal.w_max * lumpy_marginal.p_max
discrete_low = lumpy_marginal.w_min * lumpy_marginal.p_min


expectation_of_lumpy_betamix = (1 - lumpy_marginal.w_max - lumpy_marginal.w_min) * betamix_expectation + discrete_high + discrete_low

assert np.abs(betamix_expectation - beta_couple.compute_expectation(
    pi=lumpy_marginal.pi,
    alpha1=lumpy_marginal.alpha1,
    beta1=lumpy_marginal.beta1,
    alpha2=lumpy_marginal.alpha2,
    beta2=lumpy_marginal.beta2,
    p_min=lumpy_marginal.p_min,
    p_max=lumpy_marginal.p_max
)) < 0.001
assert np.abs(lumpy_marginal.compute_expectation() - expectation_of_lumpy_betamix) < 0.001



# Integrated test cases

thresholds = [[0.3,0.9], [0.,0.5]]
fudge_factor = 1e-6

print("\n--- PROBABILITIES ---\n")

# compute probability that model 0 delegates
r0,a0 = thresholds[0]
prob_model0_delegates = mm_sm.base_marginal.cdf(a0 + fudge_factor) - mm_sm.base_marginal.cdf(r0 + fudge_factor)

# compute probability that model 0 accepts
prob_model0_accepts = mm_sm.base_marginal.cdf(1 + fudge_factor) - mm_sm.base_marginal.cdf(a0 + fudge_factor)

# print probabilities for model 0
print(f"Prob(Model 0 Accepts) = {prob_model0_accepts}")
print(f"Prob(Model 0 Delegates) = {prob_model0_delegates}")
print(f"Prob(Model 0 Does Not Reject) = {prob_model0_accepts + prob_model0_delegates}")

# compute probability that model 1 accepts given that model 0 delegates
r1,a1 = thresholds[1]
prob_model1_accepts_given_that_model0_delegates = mm_sm.bivariate_models[0].get_cond_prob(
    prev_conf=[r0,a0], curr_conf=[a1,1+fudge_factor]
)
print(f"Prob(Model 1 Accepts Given That Model 0 Delegates) = {prob_model1_accepts_given_that_model0_delegates}")

# compute partial expectation of 1-conf0 against acceptance of model0
print("\n--- EXPECTED ERROR ---")
print("")
expected_error_where_model0_accepts = partial_expectation(mm_sm.base_marginal, integration_interval=[a0,1+fudge_factor], func=lambda x: 1-x)
expected_correctness_where_model0_accepts = partial_expectation(mm_sm.base_marginal, integration_interval=[a0,1+fudge_factor], func=lambda x: x)
print(f"Expected Error Restricted To Outcomes Where Model 0 Accepts = {expected_error_where_model0_accepts}")
print(f"Conditional Expectation of Error Given That Model 0 Accepts = {expected_error_where_model0_accepts / prob_model0_accepts}")
print(f"Expected Correctness Restricted To Outcomes Where Model 0 Accepts = {expected_correctness_where_model0_accepts}")
print(f"Conditional Expectation of Correctness Given That Model 0 Accepts = {expected_correctness_where_model0_accepts / prob_model0_accepts}")
print(f"-> CHECK: (Expected Correctness of Model 0) + (Expected Error of Model 0) - Prob(Model 0 Accepts) = {expected_correctness_where_model0_accepts + expected_error_where_model0_accepts - prob_model0_accepts}")

assert np.abs(expected_correctness_where_model0_accepts + expected_error_where_model0_accepts - prob_model0_accepts) < 1e-3

# compute partial expectation of 1-conf1 of model1
expected_error_where_model1_accepts = conditioned_partial_expectation(
    mm_sm.bivariate_models[0], prev_conf=[r0,a0], func=lambda x: 1-x, integration_interval=[a1, 1.0 + fudge_factor]
)
expected_correctness_where_model1_accepts = conditioned_partial_expectation(
    mm_sm.bivariate_models[0], prev_conf=[r0,a0], func=lambda x: x, integration_interval=[a1, 1.0 + fudge_factor]
)
print("")
print(f"Expected Error Restricted To Outcomes Where Model 1 Accepts = {expected_error_where_model1_accepts}")
print(f"Conditional Expectation of Error Given That Model 1 Accepts = {expected_error_where_model1_accepts / prob_model1_accepts_given_that_model0_delegates}")

print(f"Expected Correctness Restricted To Outcomes Where Model 1 Accepts = {expected_correctness_where_model1_accepts}")
print(f"Conditional Expectation of Correctness Given That Model 1 Accepts = {expected_correctness_where_model1_accepts / prob_model1_accepts_given_that_model0_delegates}")

print(f"-> CHECK: (Expected Correctness of Model 1) + (Expected Error of Model 1) - Prob(Model 1 Accepts Given That Model 0 Delegates) = {expected_correctness_where_model1_accepts + expected_error_where_model1_accepts - prob_model1_accepts_given_that_model0_delegates}")
assert np.abs(expected_correctness_where_model1_accepts + expected_error_where_model1_accepts - prob_model1_accepts_given_that_model0_delegates) < 1e-3

### Compute Abstention Rate

print("\n--- ABSTENTION RATE ---\n")
prob_model0_abstains = 1 - (prob_model0_accepts + prob_model0_delegates)
prob_model1_abstains = prob_model0_delegates * (1 - prob_model1_accepts_given_that_model0_delegates)
abstention_rate = prob_model0_abstains + prob_model1_abstains
print(f"Prob(Model0 Abstains) = {prob_model0_abstains}")
print(f"Prob(Model1 Abstains) = {prob_model1_abstains}")
print(f"Abstention Rate = {abstention_rate}")

print(f"-> CHECK: (Expected Correctness+Error Where Model 0 Accepts) + Prob(Model 0 Delegates) + Prob(Model 0 Abstains) - 1 = {expected_correctness_where_model0_accepts + expected_error_where_model0_accepts + prob_model0_delegates + prob_model0_abstains - 1}")
print(f"-> CHECK: ( 1 - (Expected Correctness+Error Where Model 1 Accepts) * Prob(Model 0 Delegates) ) - Prob(Model 1 Abstains) = {((1 - (expected_correctness_where_model1_accepts + expected_error_where_model1_accepts)) * prob_model0_delegates) - prob_model1_abstains}")


### Tests for computing performance metrics

from probabilistic_modeling.confidence_model import partial_expectation, predict_performance_probabilities

### Have the first model take every query without rejection
metrics_no_delegation = predict_performance_probabilities(mm_sm, thresholds=[[0.,0.],[0.,0.], [0.,0.]])

assert metrics_no_delegation["E"][-2] == 0.0
assert metrics_no_delegation["E"][-1] == 0.0
assert np.all(np.array(metrics_no_delegation['R']) == 0.0)

### Have the first model take every query but reject some of the time
metrics_only_first_model_rejects = predict_performance_probabilities(mm_sm, thresholds=[[0.5,0.5],[0.,0.], [0.,0.]])

assert metrics_only_first_model_rejects["A"][-2] == 0.0
assert metrics_only_first_model_rejects["A"][-1] == 0.0
assert metrics_only_first_model_rejects["E"][-2] == 0.0
assert metrics_only_first_model_rejects["E"][-1] == 0.0
assert metrics_only_first_model_rejects ["R"][-2] == 0.0
assert metrics_only_first_model_rejects ["R"][-1] == 0.0
assert metrics_only_first_model_rejects ["R"][0] > 0.0

all_response_probs_for_first_model = [ metrics_only_first_model_rejects[metric] for metric in ['A','R'] ]
assert np.abs(np.sum(all_response_probs_for_first_model) - 1.0) < 1e-6

### Have the first model delegate to the second model
metrics_with_one_delegation = predict_performance_probabilities(mm_sm, thresholds=[[0.5,0.8],[0.,0.], [0.,0.]])
all_response_probs = [ metrics_with_one_delegation[metric] for metric in ['A','R'] ]
assert np.abs(np.array(all_response_probs).sum() - 1.0) < 1e-6

### Have the first model delegate to the second model, second model can reject
metrics_for_two_model_chain = predict_performance_probabilities(mm_sm, thresholds=[[0.5,0.8],[0.5,0.5], [0.,0.]])
all_response_probs = [ metrics_for_two_model_chain[metric] for metric in ['A','R'] ]
assert np.abs(np.array(all_response_probs).sum() - 1.0) < 1e-6

### Delegate to all models, avoid rejection in the last model
metrics_for_three_model_delegation = predict_performance_probabilities(mm_sm, thresholds=[[0.5,0.8],[0.5,0.8], [0.0,0.0]])
all_response_probs = [ metrics_for_three_model_delegation[metric] for metric in ['A','R'] ]
assert np.abs(np.array(all_response_probs).sum() - 1.0) < 1e-6

### Delegate to all models, allow rejection at any level
metrics_for_three_model_delegation = predict_performance_probabilities(mm_sm, thresholds=[[0.5,0.8],[0.5,0.8], [0.5,0.5]])
all_response_probs = [ metrics_for_three_model_delegation[metric] for metric in ['A','R'] ]
assert np.abs(np.array(all_response_probs).sum() - 1.0) < 1e-6