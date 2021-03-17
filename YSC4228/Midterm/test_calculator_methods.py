from calculator import returns_calc
from datetime import date
import unittest
import matplotlib.pyplot as plt

stock = returns_calc("MSFT", "20100101", "20200101", 100)

class test_methods(unittest.TestCase):
    def test_get_name(self):
        name = stock.get_name()
        self.assertAlmostEqual(name, "Microsoft Corporation")

    def test_calendar_days(self):
        cal_days = stock.cal_days()
        self.assertAlmostEqual(cal_days, 3652)

    def test_total_stock_return(self):
        total_ret = stock.total_stock_return()
        self.assertAlmostEqual(total_ret, 557.34, places=3)

    def test_total_return_aum(self):
        total_ret_aum = stock.total_return_aum()
        self.assertAlmostEqual(total_ret_aum, 557.34, places=3)

    def test_annualized_ror(self):
        annualized_ror = stock.annualized_ror_calc()
        self.assertAlmostEqual(annualized_ror, 0.20, places=3)

    def test_final_aum_calc(self):
        final_aum = stock.final_aum_calc()
        self.assertAlmostEqual(final_aum, 55833.56, places=3)

    def test_avg_aum_calc(self):
        avg_aum = stock.avg_aum_calc()
        self.assertAlmostEqual(avg_aum, 218.72, places=3)

    def test_max_aum_calc(self):
        max_aum = stock.max_aum_calc()
        self.assertAlmostEqual(max_aum, 662.59, places=3)

    def test_pnl_calc(self):
        pnl = stock.pnl_calc()
        self.assertAlmostEqual(pnl, 55733.56, places=3)

    def test_avg_daily_ret(self):
        avg_ret_d = stock.avg_daily_ret()
        self.assertAlmostEqual(avg_ret_d, 0.000851, places=3)

    def test_sd_portfolio_d(self):
        sd_rets_d = stock.sd_portfolio_d()
        self.assertAlmostEqual(sd_rets_d, 0.0143, places=3)

    def test_sharpe_calc(self):
        sharpe = stock.sharpe_calc()
        self.assertAlmostEqual(sharpe, 5.25, places=3)

    def test_plot(self):
        fig = plt.figure()
        assert fig.get_axes() == []
        stock.plot()
        assert fig.get_axes() != []
