import cvxpy as cp

w = cp.Variable(n)
gamma = cp.Parameter(nonneg=True)
returns = mu.T*w 
risk = cp.quad_form(w, Sigma)
objective = cp.Maximize(ret - gamma*risk)
constraints =  [cp.sum(w) == 1, w >= 0]
cp.Problem(objective, constraints)

