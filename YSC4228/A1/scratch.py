#local_linear.py

#y = f(x) + e using a local linear kernel estimator and k-fold cross-validation to select the bandwidth
# gaussian kernel



#predicting new values


#plot of estimated regression function; scatterplot w abline

#python local_linear.py --x xin --y yin --output output –-num_folds 10

"""
NOTES:


"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.nonparametric.kernel_regression import KernelReg

import sklearn.model_selection as sk_ms

import skfda
import skfda.preprocessing.smoothing.kernel_smoothers as ks
import skfda.preprocessing.smoothing.validation as val

"""
parser = argparse.ArgumentParser()
parser.add_argument("") 	# naming it
args = parser.parse_args()	# returns data from the options specified (echo)


args = parser.parse_args()

print("Argument 1", str(sys.argv[1]))
print("Argument 2", str(sys.argv[2]))
print("Argument 3", str(sys.argv[3]))

parser = argparse.ArgumentParser()
parser.add_argument("--x", type=str, default="echo", help="echo this thing") 	# naming it "echo"
parser.add_argument("--y")
args = parser.parse_args()	# returns data from the options specified (echo)
print(args.echo)
print(args.optional)
"""
os.chdir('C:/Users/draga/OneDrive - National University of Singapore/Electives/QuantFin/Assignmnet 1')

parser = argparse.ArgumentParser()
parser.add_argument("--x")
parser.add_argument("--y")
parser.add_argument("--output")
parser.add_argument("--num_folds")
parser.add_argument("--plot")
parser.add_argument("--xout")
args = parser.parse_args()

xin=pd.read_csv('xin.csv', names='x')
xin_list=[]
for row in range(xin.shape[0]):
    xin_list.append(float(xin.iloc[row]))

yin=pd.read_csv('yin.csv', names=['y'])
yin_list=[]
for row in range(yin.shape[0]):
    yin_list.append(float(yin.iloc[row]))



df=pd.concat([yin,xin], axis=1)

# Using statsmodels
kde = KernelReg(x, y, var_type='c', reg_type='ll', bw=[3.2])

estimator = kde.fit(y)
estimator = np.reshape(estimator[0], df.shape[0])

plt.scatter(x, y)
plt.scatter(x, estimator, c='r')
plt.show()

# Using SKFDA

df_grid=skfda.FDataGrid(df)

bandwidth = np.arange(0.1, 5, 0.2)

llr = val.SmoothingParameterSearch(
    ks.LocalLinearRegressionSmoother(),
    bandwidth)
fit = llr.fit(df_grid)
llr_df = llr.transform(df_grid)

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x, llr_df, c='r')
plt.show()

# just K-fold cross-validation
"""
rn = range(1,26)
kf3 = sk_ms.KFold(3, shuffle=False)
for train_index, test_index in kf3.split(rn):
    train_set = train_index
    test_set = test_index
"""
fit=ks.LocalLinearRegressionSmoother.fit(df_grid)

sk_ms.cross_validate(fit, xin, yin, cv=3)

# Outputs
fmt='%.8f'
np.savetxt(args.yout, yout, fmt=fmt, delimiter=',', comments='')
np.savetxt(args.xout, xout, fmt=fmt, delimiter=',', comments='')




# Custom Kernel Estimator #

from scipy.stats import norm
import math

class GKR:

    def __init__(self, x, y, b):
        self.x = x
        self.y = y
        self.b = b

    def gaussian_kernel(self, z):
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*z**2)

    def predict(self, X):
        kernels = [self.gaussian_kernel((xi-X)/self.b) for xi in self.x]
        weights = [len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels]
        return np.dot(weights, self.y)/len(self.x)

"""
import csv
xin=open('xin.csv','r')
reader=csv.reader(xin)
xin_list=[]
for row in xin:
    xin_list.append(row)
"""

llke = GKR(xin_list, yin_list, 0.3)

y_hat=pd.DataFrame(index=range(xin.shape[0]), columns=['y_hat'])
for row in range(xin.shape[0]):
    y_hat.iloc[row] = llke.predict(float(xin.iloc[row]))
y_hat


sk_ms.cross_val_score(llke, xin, yin, cv=3)




# plotting
plt.scatter(xin, yin)
plt.scatter(xin, y_hat, c='r')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

"""
y_hat=[]
for row in range(xin.shape[0]):
    y_hat.append(llke.predict(float(xin.iloc[row])))
y_hat
"""

x_train, x_test, y_train, y_test = train_test_split(xin, yin, test_size=, random_state=0)





import numpy as np
from sklearn.cross_validation import cross_val_score

class GKR_sk:
    def __init__(self, x, y, b):
        self.x = x
        self.y = y
        self.b = b

    def gaussian_kernel(self, z):
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*z**2)

    def combine(self, inputs):
        return sum([i*w for (i,w) in zip([1] + inputs, self.weights)])

    def predict(self, X):
        return [self.combine(x) for x in X]

    def classify(self, inputs):
        return sign(self.predict(inputs))

    def fit(self, X):
        kernels = [self.gaussian_kernel((xi-X)/self.b) for xi in self.x]
        weights = [len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels]
        return np.dot(weights, self.y)/len(self.x)

    def get_params(self, deep = False):
        return {'b':self.b, 'x':self.x, 'y':self.y}

X = np.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.matrix([0, 1, 1, 0]).transpose()

sk_ms.cross_val_score(GKR_sk(xin_list, yin_list, 0.1),
                      xin_list,
                      yin_list, 
                      fit_params={'b':0.1},
                      scoring = 'neg_mean_squared_error',
                      cv=5)

###############################################################################

xin=pd.read_csv('xin.csv', names=['x'])
xin_list=[]
for row in range(xin.shape[0]):
    xin_list.append(float(xin.iloc[row]))

yin=pd.read_csv('yin.csv', names=['y'])
yin_list=[]
for row in range(yin.shape[0]):
    yin_list.append(float(yin.iloc[row]))

df=pd.concat([yin,xin], axis=1)

class GKR:
    def __init__(self, x, y, b):
        self.x = x
        self.y = y
        self.b = b

    def gaussian_kernel(self, z):
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*z**2)

    def predict(self, X):
        kernels = [self.gaussian_kernel((xi-X)/self.b) for xi in self.x]
        weights = [len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels]
        return np.dot(weights, self.y)/len(self.x)

llke = GKR(xin_list, yin_list, 0.15)

xout = np.random.uniform(-3.0, 4.0, 1000)
xout = pd.DataFrame(data=xout.flatten(), columns=['xout'])
y_hat=pd.DataFrame(index=range(xin.shape[0]), columns=['y_hat'])
for row in range(xin.shape[0]):
    y_hat.iloc[row] = llke.predict(float(xin.iloc[row]))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(xin, yin, color='red', ls="-", label="data")
ax.scatter(xin, y_hat, color='green', ls="--", label="predictions")
ax.set_ylabel('y', loc='top')
ax.set_xlabel('x', loc='right')
ax.legend()
fig.show()

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(xin_list, yin_list)
plt.show()
xin_list.sort(reverse=False)
yin_list.sort(reverse=False)
print(xin_list)
# Outputs
np.savetxt(args.yout, y_hat, fmt='%.8f', delimiter=',', comments='')
no.savetxt(args.xout, xout, fmt='%.8f', delimiter=',', comments='')
