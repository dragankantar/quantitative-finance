import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cvx

def markowitz_MPT(mu, cov, gamma)
    w = cvx.Variable(len(ret))
    gamma = cvx.Parameter(nonneg=True)
    returns = mu * w.T 
    risk = cvx.quad_form(w, cov)

    objective = cvx.Maximize(returns - gamma*risk)
    constraints =  [cvx.sum(w) == 1, w >= 0]
    problem = cvx.Problem(objective, constraints)

    problem.solve()

    return np.array(w.value.flat).round(3)

tickers = 'SPY, DJI, STI, HSI'
start = '2010-1-1'
end = '2020-1-1'

data = yf.download(tickers, start, end)
data['Close']

r = data['Close'].pct_change()
covmtx = np.cov(r.T)

markowitz_MPT(r, covmtx, 0.2)








