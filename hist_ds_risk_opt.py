########################################################################################################################
##################################### Constraining Historical Downside Risk ############################################
########################################################################################################################

import numpy as np
import pandas as pd
import cvxpy as cvx

def maximize_alpha_constrain_downside(alphas, returns, percentile, max_loss):
    w = cvx.Variable(returns.shape[1])

    # Number of worst-case return periods to sample.
    nsamples = round(returns.shape[0] * percentile)

    portfolio_rets = returns * w
    avg_worst_day = cvx.sum_smallest(portfolio_rets, nsamples) / nsamples

    objective = cvx.Maximize(w.T * alphas)
    constraints = [avg_worst_day >= max_loss]

    problem = cvx.Problem(objective, constraints)
    problem.solve()

    return w.value.round(4).ravel()

