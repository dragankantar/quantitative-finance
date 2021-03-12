"""
YSC4228: Data Science in Quantitative Finance
Midterm Assignment
Dragan Kantar

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yfinance as yf
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", nargs=1, type=str)
parser.add_argument("--b", nargs=1, type=str)
parser.add_argument("--e", nargs=1 , type=str)
parser.add_argument("--initial_aum", nargs=1, type=int)
parser.add_argument("--plot", action='store_true')
args = parser.parse_args()









print("Begin Date:", args.b)
print("End Date:", args.e)
print("Number of Days:" )

print("Total Stock Return (adj for dividends):", "percent")
print("Total Return (of the AUM invested):")
print("Annualized rate of return (of the AUM invested):", "percent")

print("Initial AUM:")
print("Final AUM:")
print("Average AUM:")
print("Maximum AUM:")

print("PnL (of the AUM invested):")
print("Avg Daily Return of the Portfolio:", "percent")
print("Daily Standard Deviation of the Portfolio Return:")
print("Daily Sharpe Ratio of the portfolio (assuming risk-free rate of 0.01pct):")
