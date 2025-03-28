import numpy as np
from .copula import CopulaMarkovModel
from .marginals import lumpy_betamix

class MarkovConfidenceModel:
    """ Model joint distribution of calibrated confidence for models in a chain. """
    
    def __init__(self, chain_length=3, base_marginal=None, bivariate_models=None):
        if chain_length < 2:
            raise ValueError("chain_length must be >= 2 to fit a MarkovConfidenceModel")
        if bivariate_models is None:
            self.bivariate_models = [ CopulaMarkovModel() for j in range(chain_length-1) ]
        else:
            self.bivariate_models = bivariate_models

        if base_marginal is None:
            self.base_marginal = lumpy_betamix
        else:
            self.base_marginal = base_marginal

        self.chain_length = chain_length
        self.fitted = False

    def fit(self, conf_data: list):
        # fit base marginal
        base_marginal_params = self.base_marginal.fit(conf_data[0])
        self.base_marginal = self.base_marginal(*base_marginal_params)
        # next, fit the bivariate Markov transition distributions
        for i, bi_model in enumerate(self.bivariate_models):
            bi_model.fit([conf_data[i], conf_data[i+1]])
        self.fitted = True

    def compute_prob(self, intervals=[[0.,1.], [0.,1.], [0.,1.]]):
        """ 
        Compute the joint probability that the confidences of the first
        k models lie in the k given intervals.
        """
        if not self.fitted:
            raise ValueError(
                f"this {self.__class__.__name__} has not been fitted; run {self.__class__.__name__}.fit"
            )
        else:
            a, b = intervals[0]
            base_marginal_prob = self.base_marginal.cdf(b) - self.base_marginal.cdf(a)

            if len(intervals) == 1:
                return base_marginal_prob
            elif len(intervals) > 1:

                transition_probs = [ 
                    self.bivariate_models[j-1].get_cond_prob(
                        [intervals[j-1][0], intervals[j-1][1]], 
                        [intervals[j][0], intervals[j][1]]
                    ) for j in range(1, len(intervals))
                ]
            
                return np.prod([base_marginal_prob, *transition_probs])


def partial_expectation(
        marginal, integration_interval=[0,1], func=lambda x: x, dx=1e-3, fudge_factor=0.0
    ):
    """ 
    Integrate the marginal distribution against the provided function, over the
    given integration interval.
    """
    a, b = integration_interval
    pdf = lambda x: (marginal.cdf(x+dx-fudge_factor) - marginal.cdf(x-fudge_factor))/dx
    xvals = np.arange(a, b, dx)
    integrand_vals = np.array([ pdf(x) * func(x) for x in xvals ])
    integral_result = np.sum((integrand_vals[1:] + integrand_vals[:-1])/2) * dx

    return integral_result


def conditioned_partial_expectation(
        bivariate_model, prev_conf, func = lambda x: x, integration_interval=[0,1], dx=1e-3
    ):
    """ 
    Integrate the Markov conditional distribution against the provided function, over the
    given integration interval.
    """
    a,b = prev_conf
    c,d = integration_interval
    
    # define approximated conditional density function
    cond_pdf = lambda x: bivariate_model.get_cond_prob(prev_conf=[a,b], curr_conf=[x, x+dx])/dx
    # define x values corresponding to the density function approximation grid
    xvals = np.arange(c, d, dx)
    # prepare values of the integrand: approximate density * func
    integrand_vals = np.array([ cond_pdf(x) * func(x) for x in xvals ])
    # use trapezoidal rule to integrate
    integral_result = np.sum((integrand_vals[1:] + integrand_vals[:-1])/2) * dx

    return integral_result


### Functions for computing probability of error

def compute_IE(i, joint_model: MarkovConfidenceModel, thresholds, fudge_factor=1e-10):
    if i == 0:
        r, a = thresholds[0]
        return partial_expectation(joint_model.base_marginal, [a - fudge_factor, 1.0 + fudge_factor], func=lambda x: 1-x)
    elif i > 0:
        r, a = thresholds[i]
        r_prev, a_prev = thresholds[i-1]
        return conditioned_partial_expectation(
            bivariate_model=joint_model.bivariate_models[i-1], 
            prev_conf=[r_prev, a_prev],
            func = lambda x: 1-x,
            integration_interval=[a - fudge_factor, 1.0 + fudge_factor]
        )

def compute_IR(i, joint_model: MarkovConfidenceModel, thresholds, fudge_factor=0.0):
    if i == 0:
        r, a = thresholds[0]
        return joint_model.base_marginal.cdf(r - fudge_factor)
    elif i > 0:
        r, a = thresholds[i]
        r_prev, a_prev = thresholds[i-1]
        return joint_model.bivariate_models[i-1].get_cond_prob(
            prev_conf=[r_prev, a_prev], curr_conf=[-np.inf, r]
        )


def compute_IA(i, joint_model: MarkovConfidenceModel, thresholds, fudge_factor=0.0):
    if i == 0:
        r, a = thresholds[0]
        return 1 - joint_model.base_marginal.cdf(a - fudge_factor)
    elif i > 0:
        r, a = thresholds[i]
        r_prev, a_prev = thresholds[i-1]
        return joint_model.bivariate_models[i-1].get_cond_prob(
                prev_conf=[r_prev, a_prev], curr_conf=[a, np.inf]
            )


def compute_P(i, joint_model: MarkovConfidenceModel, thresholds, fudge_factor=0.0):
    if i == 0:
        r, a = thresholds[0]
        return joint_model.base_marginal.cdf(a - fudge_factor) - joint_model.base_marginal.cdf(r - fudge_factor)
    elif i > 0:
        r, a = thresholds[i]
        r_prev, a_prev = thresholds[i-1]
        return joint_model.bivariate_models[i-1].get_cond_prob([r_prev, a_prev], [r,a])


def predict_performance_probabilities(joint_model: MarkovConfidenceModel, thresholds, fudge_factor=0.0):
    E = []; R = []; A = [];
    P_cum = 1

    for j in range(joint_model.chain_length):

        IE_j = compute_IE(j, joint_model, thresholds, fudge_factor=fudge_factor)
        IA_j = compute_IA(j, joint_model, thresholds, fudge_factor=fudge_factor)
        IR_j = compute_IR(j, joint_model, thresholds, fudge_factor=fudge_factor)

        if j>=1:
            P = compute_P(j-1, joint_model, thresholds, fudge_factor=fudge_factor)
            P_cum *= P

        E_j = P_cum * IE_j
        A_j = P_cum * IA_j
        R_j = P_cum * IR_j

        E.append(E_j)
        A.append(A_j)
        R.append(R_j)

    return {"E": E, "A": A, "R": R}


def predict_metrics(joint_model: MarkovConfidenceModel, thresholds, costs = [0.2, 0.8, 3.0]):
    """ Predict the performance metrics. """
    metrics = predict_performance_probabilities(joint_model, thresholds=thresholds)
    cond_error = np.sum(metrics['E'])/np.sum(metrics['A'])
    rej_rate = np.sum(metrics['R'])
    expected_cost = np.sum( np.cumsum(costs) * (np.array(metrics['R']) + np.array(metrics['A'])) )

    return cond_error, rej_rate, expected_cost


def unflatten_thresholds(flattened_thresholds):
    return [ *([ [flattened_thresholds[2*i], flattened_thresholds[2*i+1]] for i in range(len(flattened_thresholds)//2) ]), [flattened_thresholds[-1], flattened_thresholds[-1]]]


def flatten_thresholds(thresholds):
    return [ x for x_list in thresholds for x in x_list ][:-1]


flattened_thresholds = (0.5,0.8,0.2,0.9,0.)
assert np.all(np.array(flatten_thresholds(unflatten_thresholds(flattened_thresholds))) == np.array(flattened_thresholds))


def compute_performance_derivatives(joint_model: MarkovConfidenceModel, flattened_thresholds, costs=[0.2, 0.8, 3.0, 10.0], eps=1e-3):
    thresholds = unflatten_thresholds(flattened_thresholds)
    perf_orig = np.array([ list(predict_metrics(joint_model, thresholds=thresholds, costs=costs)) ])

    perturbed_thresholds = np.array([flattened_thresholds]) + eps*np.eye(2*joint_model.chain_length - 1)

    perturbed_columns = []
    for j in range(2*joint_model.chain_length - 1):
        perf_perturbed = list(predict_metrics(
            joint_model, 
            thresholds=unflatten_thresholds(list(perturbed_thresholds[j,:])),
            costs=costs
        ))
        perturbed_columns.append(perf_perturbed)

    jacobian = np.transpose((perf_orig - perturbed_columns)/eps)
    return jacobian