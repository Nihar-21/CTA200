from autograd import numpy as np
from lifelines.fitters import ParametricUnivariateFitter

class CureFitter_K(ParametricUnivariateFitter):

    _fitted_parameter_names = ["p_", "lambda_"]

    _bounds = ((0, 1), (0, None))

    def _cumulative_hazard(self, params, T):
        p, lambda_ = params
        sf = np.exp(-(T / lambda_))
        return -np.log(p + (1-p) * sf)

class CureFitter0_K(ParametricUnivariateFitter):

    _fitted_parameter_names = ["p_"]

    _bounds = [(0, 1)]
    def _cumulative_hazard(self, params, T):
        lambda_ = 100
        p = params
        sf = np.exp(-(T / lambda_))
        return -np.log(p + (1-p) * sf)
    
    