from calculator import returns_calc
from datetime import date

stock_2 = returns_calc("MSFT", None, "20200101", 100)

def test_no_begin():
    begin_real = stock_2.begin_real
    assert(print(begin_real)=="1986-03-13")

stock_3 = returns_calc("MSFT", "20100101", None, 100)

def test_no_end():
    end_real = stock_2.end_real
    assert(print(end_real)==print(date.today()))

stock_4 = returns_calc("MSFT", None, None, 100)

def test_no_begin_no_end():
    begin_real = stock_2.begin_real
    assert(print(begin_real)=="1986-03-13")
    end_real = stock_2.end_real
    assert(print(end_real)==print(date.today()))

stock_5 = returns_calc(None, "20100101", "20200101", 100)

def test_no_ticker():
    assert(stock_5.ticker==None)
