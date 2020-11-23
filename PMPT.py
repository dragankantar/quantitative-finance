import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import cvxpy as cvx

def pmpt(mu, returns, percentile, max_loss):
    w = cvx.Variable(returns.shape[1])

    nsamples = round(returns.shape[0] * percentile)

    portfolio_rets = returns @ w
    avg_worst_day = cvx.sum_smallest(portfolio_rets, nsamples) / nsamples

    objective = cvx.Maximize(w.T @ mu)
    constraints = [avg_worst_day >= max_loss]

    problem = cvx.Problem(objective, constraints)
    problem.solve()

    return w.value.round(4).ravel()

tickers = 'SPY, DJI, STI'
start = '2010-1-1'
end = '2018-1-1'

""" optional constraints (add in constraints square brackets)
cvx.sum(w) == 1, ### no leverage, fully invested
cvx.sum(w) <= 1, ### no leverage
w >= 0] ### long only
"""
data = yf.download(tickers, start, end)

r = data['Close'].pct_change().iloc[1:].to_numpy() # returns of assets
mu = np.mean(r).to_numpy() # expected returns of assets
percentile = 0.05 # defines worst days
max_loss = -0.05 # maximum acceptable loss

result = pmpt(mu, r, percentile, max_loss)
print("Portfolio weights:", result)

portfolio_returns = r.dot(result)
worst_days = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]
print("Average Bad Day:", worst_days.mean())

# plotting daily returns
r_forplot = data['Close'].pct_change().iloc[1:] # redefining data as pandas df, np ndarray
for i in range(r_forplot.shape[1]):
    plt.subplot(r_forplot.shape[1], 1, i+1)
    plt.plot(r_forplot.iloc[0:, i])
    plt.xlabel('Time') 
    plt.ylabel('%s Daily Returns' % r_forplot.columns.values[i])
    
plt.tight_layout()
plt.show() 
