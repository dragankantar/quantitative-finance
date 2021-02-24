"""
local linear kernel estimator and k-fold cross-validation to select the bandwidth
NOTES:


"""

import sys
import argparse

import matplotlib.pylab as plt
import numpy as np

import skfda
import skfda.preprocessing.smoothing.kernel_smoothers as ks
import skfda.preprocessing.smoothing.validation as val


parser = argparse.ArgumentParser()
parser.add_argument("--x")
parser.add_argument("--y")
parser.add_argument("--output")
parser.add_argument("--num_folds")
parser.add_argument("--plot")
parser.add_argument("--xout")
args = parser.parse_args()




fmt='%.8f'
savetext(args.yout, yout, fmt=fmt, delimiter=',', comments='')
savetxt(args.xout, xout, fmt=fmt, delimiter=',', comments='')
