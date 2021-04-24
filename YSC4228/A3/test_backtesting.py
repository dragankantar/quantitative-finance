'''
Pytests to test the metrics using wo possible methods to optimize portfolio.

usage: Run "pytest" in a shell in the directory containing this file.

'''

import pytest
from backtester import BackTester
import datetime
from dateutil.relativedelta import relativedelta

aum = 10000
optimizer = "mvo"
tickers = "GOOG,AAPL,FB"
tickers = tickers.split(",")
backtest_months = 12
test_months = 1
TO_PLOT = True

bck_tstr = BackTester(tickers)
end_date = datetime.date(2021, 4, 25)
start_date = end_date - relativedelta(months=test_months)
back_test_date = end_date - relativedelta(months=backtest_months)
start_date, end_date, back_test_date = bck_tstr.get_dates(test_months, backtest_months)
train, test, data =  bck_tstr.get_data(back_test_date, start_date, end_date)

## Walk Forward Method
weights, performance, realised_annual_return, realised_annual_volatility, realised_sharpe_ratio = bck_tstr.wf(train, optimizer, test, test_months) 
expected_annual_return,annual_volatility, sharpe_ratio = performance
number_of_shares = bck_tstr.shares_needed(aum, weights, test)

## CV Method
weights_, performance_, realised_annual_return_, realised_annual_volatility_, realised_sharpe_ratio_ = bck_tstr.cv(backtest_months, data, optimizer, test_months, annual_risk_free_rate=0.02)
expected_annual_return_,annual_volatility_, sharpe_ratio_ = performance
number_of_shares_ = bck_tstr.shares_needed(aum, weights_, test)

## Testing metics using Walk Forward

def test_back_tstr():
    assert bck_tstr.tickers == tickers

def test_realized_return_wf():
    assert realised_annual_return == 3.1538305356916423

def test_realized_volatiliti_wf():
    assert realised_annual_volatility == 0.20941677017932647

def test_realized_sharpe_wf():
    assert realised_sharpe_ratio == 14.964563406302656

def test_expected_return_wf():
    assert expected_annual_return == 0.7079969560659486

def test_expected_volatility_wf():
    assert annual_volatility == 0.2809038282773268

def test_expected_sharpe_wf():
    assert sharpe_ratio == 2.44922598700475

def test_num_shares_wf():
    assert number_of_shares.to_dict() == {"AAPL":27, "FB":1, "GOOG":3}

## Testing metrics using Using CV

def test_realized_return_cv():
    assert realised_annual_return_ == 1.5533927403085224

def test_realized_volatiliti_cv():
    assert realised_annual_volatility_ == 0.3089849090137056

def test_realized_sharpe_cv():
    assert realised_sharpe_ratio_ == 10.476494531619243

def test_expected_return_cv():
    assert expected_annual_return_ == 0.7079969560659486

def test_expected_volatility_cv():
    assert annual_volatility_ == 0.2809038282773268

def test_expected_sharpe_cv():
    assert sharpe_ratio_ == 2.44922598700475

def test_num_shares_cv():
    assert number_of_shares_.to_dict() == {"AAPL":28, "FB":12, "GOOG":2}
