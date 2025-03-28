import numpy as np
from scipy.stats import gamma
from statsmodels.distributions.copula.api import (
    CopulaDistribution, GumbelCopula
)
from .marginals import lumpy_betamix

class CopulaMarkovModel:
    def __init__(self, copula=GumbelCopula, marginals=[lumpy_betamix, lumpy_betamix]):
        self.copula = copula
        self.marginals = marginals
        self.joint_distribution = None
        self.fitted = False

    def fit(self, conf_data: list):
        """ 
        Fit this CopulaMarkovModel. 

        Parameters:
        - conf_data: list
            Contains lists of confidences for both the previous and current model.
        
        """
        assert len(conf_data) == 2, f"{self.__class__.__name__} expects confidence data for exactly two models"
        previous_model_data = conf_data[0]
        curr_model_data = conf_data[1]

        marginal_params_prev = self.marginals[0].fit(conf_data[0])
        marginal_params_curr = self.marginals[1].fit(conf_data[1])
        fitted_marginals = [ 
            self.marginals[0](*marginal_params_prev), 
            self.marginals[1](*marginal_params_curr)
        ]

        copula_sample = np.concatenate(
            [ np.array(z)[:, np.newaxis] for z in conf_data ],
            axis=1
        )
        copula_params = self.copula().fit_corr_param(copula_sample)
        fitted_copula = self.copula(copula_params)

        self.joint_distribution = CopulaDistribution(
            copula=fitted_copula, marginals=fitted_marginals
        )
        self.fitted = True

    def copularize_conf(cdf, conf_array):
            return cdf(conf_array)

    def get_cond_prob(self, prev_conf, curr_conf, fudge_factor=1e-10):
        """ Get the conditional probability. 

        Return 0 if the conditioned event has zero probability.
        
        Parameters:
        prev_conf: [a,b]
        curr_conf: [c,d]
        fudge_factor: float
            Purpose is to approximate probabilities over intervals [a,b), rather
            than (a,b] (the probabilities we get with fudge_factor=0). Observe
            that (a-eps, b-eps] is roughly equivalent to [a, b).
        """
        if not self.fitted:
             raise ValueError("this CopulaMarkovModel is not fitted!")
        
        u1_lower, u1_upper = CopulaMarkovModel.copularize_conf(
            self.joint_distribution.marginals[0].cdf,
            [ x - fudge_factor for x in prev_conf ]
        )

        if u1_upper - u1_lower <= 0.0:
            return 0.0

        u2_lower, u2_upper = CopulaMarkovModel.copularize_conf(
            self.joint_distribution.marginals[1].cdf,
            [ x - fudge_factor for x in curr_conf ]
        )

        main_term = self.joint_distribution.copula.cdf(u=(u1_upper, u2_upper))
        subtracted_terms = self.joint_distribution.copula.cdf(u=(u1_lower, u2_upper)) + self.joint_distribution.copula.cdf(u=(u1_upper, u2_lower))
        addback_term = self.joint_distribution.copula.cdf(u=(u1_lower, u2_lower))

        numerator_prob = main_term - subtracted_terms + addback_term
        denominator_prob = self.joint_distribution.copula.cdf(u=(u1_upper,1)) - self.joint_distribution.copula.cdf(u=(u1_lower,1))

        cond_prob = numerator_prob / denominator_prob
        return cond_prob