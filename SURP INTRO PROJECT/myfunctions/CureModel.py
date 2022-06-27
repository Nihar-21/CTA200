from autograd import numpy as np
from lifelines.fitters import ParametricUnivariateFitter

class CureFitter(ParametricUnivariateFitter):

    _fitted_parameter_names = ["p_", "lambda_"]

    _bounds = ((0, 1), (0, None))

    def _cumulative_hazard(self, params, T):
        p, lambda_ = params
        sf = np.exp(-(T / lambda_))
        return -np.log(p + (1-p) * sf)