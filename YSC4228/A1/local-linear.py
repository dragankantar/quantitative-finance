import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import operator

## Assignment 1- Alaukik, Dragan, Martynas
## Example command to run: "python local-linear.py --x xin --y yin –-output output --num_folds 10 --xout lin --plot"


## PART 1- reading the arguments from the user for the script
arg_list = sys.argv
x_file = None
y_file = None
num_folds = None
plot_file = False
is_xout = False
x_out_file = None

i = 1
while (i < len(arg_list)):
    if arg_list[i] == '--x':
        i += 1
        x_file = arg_list[i]
    elif arg_list[i] == '--y':
        i += 1
        y_file = arg_list[i]
    elif arg_list[i] == '--num_folds':
        i += 1
        num_folds = int(arg_list[i])
    elif arg_list[i] == '--plot':
        plot_file = True
    elif arg_list[i] == '--xout':
        i += 1
        is_xout = True
        x_out_file = arg_list[i]
    i+= 1

def read_coordinates(filename):
    with open(filename) as f:
        content = f.readlines()
        content = [float(x.strip()) for x in content] 
        content = np.array(content) 
    return content

xin_list = read_coordinates(x_file)
yin_list = read_coordinates(y_file)
xin = np.array(xin_list)
yin = np.array(yin_list)


## PART 2- Defining a class GKR that defines the estimator

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
    
    def calculate_error(self, y, y_hat):
        leng = len(y)
        sum_error = 0
        for i in range(leng):
            sum_error += (y[i] - y_hat[i]) **2
        error = sum_error/leng
        return error


# PART 3- finding the best bandwidth through manual cross validation from a list of bandwidths
def find_bandwidth(xin_list,yin_list, num_folds, bandwidths):
    print("Finding the required bandwidth ... \n")
    error_avg = {}
    leng = len(xin_list)
    for bandwidth in bandwidths:
        error_sum = 0
        for i in range(num_folds):
            begin = int((i/num_folds) *  leng)
            end = int(((i + 1)/num_folds) * leng)
            llke = GKR(xin_list[begin:end], yin_list[begin:end], bandwidth)
            x_out = xin_list[begin:end]
            y_hat_test = []
            for row in range(len(x_out)):
                y_hat_test.append(llke.predict(float(x_out[row])))
            error = llke.calculate_error(yin_list[begin:end], y_hat_test)
            error_sum += error
        avg = error_sum/num_folds
        error_avg[bandwidth] = avg
    return min(error_avg, key=error_avg.get) 

bandwidths = [0.1, 0.2, 0.3]
bandwidth = find_bandwidth(xin_list,yin_list, num_folds, bandwidths)

print("Found te bandwidth. The bandwidth we use is " + str(bandwidth) + " \n")
llke = GKR(xin_list, yin_list, bandwidth)


## PART 4: Running the function on x_out if x_out is defined
print("Making Predictions ... \n")
if (is_xout):
    xin_list = read_coordinates(x_out_file)
    xin = np.array(xin_list)

y_hat=pd.DataFrame(index=range(xin.shape[0]), columns=['y_hat'])
for row in range(xin.shape[0]):
    y_hat.iloc[row] = llke.predict(float(xin[row]))
    
print("Made Predictions ... \n")


   
## PART 5: Saving the file
np.savetxt("output-for-xin-xout", y_hat, fmt='%.8f', delimiter=',', comments='')

## PART 6: Plotting the required values
if (plot_file):
    print("Now plotting. One you see the plot, pls close the plot to terminate. \n")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xin, yin, color='red', ls="-", label="data", alpha=0.5)
    ax.scatter(xin, y_hat, color='green', ls="--", label="predictions", alpha=0.3)
    ax.set_ylabel('y', loc='top')
    ax.set_xlabel('x', loc='right')
    ax.legend()
    plt.show()

print("Done!!!")
