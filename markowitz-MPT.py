import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import cvxpy as cvx

def markowitz_MPT(mean, matrix, gamma):
    w = cvx.Variable(len(mean))
    gamma = cvx.Parameter(nonneg=True, value=gamma)
    returns = mean @ w.T
    risk = cvx.quad_form(w, matrix)

    objective = cvx.Maximize(returns - gamma*risk)
    constraints = [cvx.sum(w) == 1, w >= 0]
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    return np.array(w.value.flat).round(3)

tickers = 'MSFT, SPY'
begin = '2010-1-1'
end = '2018-1-1'
gamma = 0.2 #risk aversion parameter

data = yf.download(tickers, begin, end)

returns = data['Adj Close'].pct_change()
mean_returns = returns.mean()
covariance_matrix = returns.cov()

markowitz_MPT(mean_returns, covariance_matrix, gamma)

for i in range(returns.shape[1]):
    plt.subplot(returns.shape[1], 1, i+1)
    plt.plot(returns.iloc[0:, i])
    plt.xlabel('Time')
    plt.ylabel('%s Daily Returns' % returns.columns.values[i])

plt.tight_layout()
plt.show()
