import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cvx

def markowitz_MPT(mu, covmtx, gamma):
    w = cvx.Variable(len(mu))
    gamma = cvx.Parameter(nonneg=True)
    returns = mu @ w.T 
    risk = cvx.quad_form(w, covmtx)

    objective = cvx.Maximize(returns - gamma*risk)
    constraints =  [cvx.sum(w) == 1, w >= 0]
    problem = cvx.Problem(objective, constraints)

    problem.solve()

    return np.array(w.value.flat).round(3)

tickers = 'SPY, DJI, STI'
start = '2010-1-1'
end = '2020-1-1'

data = yf.download(tickers, start, end)

r = data['Close'].pct_change()
r = r.iloc[1:]
mu = np.mean(r)
covmtx = np.cov(r, rowvar = False)

markowitz_MPT(mu, covmtx, 0.2)

for i in range(r.shape[1]):
    plt.subplot(r.shape[1], 1, i+1)
    plt.plot(r.iloc[0:, i])
    plt.xlabel('Time') 
    plt.ylabel('%s Daily Returns' % r.columns.values[i])
    
plt.tight_layout()
plt.show() 

