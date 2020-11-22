#useful for mixed asset portfolio

#PMPT
"""Assumptions
Asymetrical risk

downside risk is measured by target semideviation ('downside deviation')

uses sortion ratio instead of sharpe ratio as a measure of risk adjuster return

volatility skewness as measure of the ratio of a destributions's percentage of total 
variance from returns above mean to retuns below mean was added
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import cvxpy as cvx

def pmpt(mu, returns, percentile, max_loss):
    w = cvx.Variable(returns.shape[1])

    # Number of worst-case return periods to sample.
    nsamples = round(returns.shape[0] * percentile)

    portfolio_rets = returns @ w
    avg_worst_day = cvx.sum_smallest(portfolio_rets, nsamples) / nsamples

    objective = cvx.Maximize(w.T @ mu)
    constraints = [avg_worst_day >= max_loss]

    problem = cvx.Problem(objective, constraints)
    problem.solve()

    return w.value.round(4).ravel()


###
tickers = 'SPY, DJI, STI'
start = '2010-1-1'
end = '2018-1-1'

data = yf.download(tickers, start, end)

r = data['Close'].pct_change().iloc[1:] # returns of assets
mu = np.mean(r).to_numpy() # expected returns of assets
covmtx = np.cov(r, rowvar = False)
percentile = 0.05 # defines worst days
max_loss = -0.05 # maximum acceptable loss


result = pmpt(mu, r, percentile, max_loss)
print("Portfolio:", result)

portfolio_rets = rets.dot(result)
worst_days = portfolio_rets[portfolio_rets <= np.percentile(portfolio_rets, 5)]
print("Average Bad Day:", worst_days.mean())

for i in range(r.shape[1]):
    plt.subplot(r.shape[1], 1, i+1)
    plt.plot(r.iloc[0:, i])
    plt.xlabel('Time') 
    plt.ylabel('%s Daily Returns' % r.columns.values[i])
    
plt.tight_layout()
plt.show() 

