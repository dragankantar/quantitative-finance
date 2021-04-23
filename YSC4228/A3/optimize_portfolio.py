import datetime
import argparse
import cleaner
from backtester import BackTester

##python3 optimize_portfolio.py --tickers GOOG,AAPL,FB --optimizer mvo --aum 10000 --backtest_months 12 --test_months 1 --backtest_type wf


def main():

    def enable_parsing(parser):
        parser.add_argument("--optimizer", choices=['msr', 'mvo', 'hrp'], help="Optimizers: msr, mvo, hrp", required=True)
        parser.add_argument("--backtest_type", choices=['wf', 'sv'], help="Backtest type options: wf, sv", required=True)
        parser.add_argument("--tickers", help="Tickers to be used seperated by comma", required=True)
        parser.add_argument("--aum", type=int, help="Assets Under Management", required=True)
        parser.add_argument("--backtest_months", type=int, help="Backtest months to be used", required=True)
        parser.add_argument("--test_months", type=int, help="Test months to be used", required=True)
        parser.add_argument('--plot_weights', help="To plot",
                                action='store_true')
        return parser

    parser = argparse.ArgumentParser()
    parser = enable_parsing(parser)
    args = parser.parse_args()

    optimizer = args.optimizer
    backtest_type = args.backtest_type
    tickers = args.tickers
    tickers = tickers.split(",")
    backtest_months = args.backtest_months
    test_months = args.test_months
    TO_PLOT = args.plot_weights

    if ((backtest_months % test_months) != 0):
        print("Backtest month should be a multiple of test months.")
        return

   
    # print("Arguments are these: \n")
    # print(optimizer)
    # print(backtest_type)
    # print(tickers)
    # print(backtest_months)
    # print(test_months)
    # print(TO_PLOT)
    # print("\n")
    

    bck_tstr = BackTester(tickers)

    start_date, end_date,  backtest_start_date = bck_tstr.get_dates(test_months, backtest_months)
    df = bck_tstr.get_data_frame(backtest_start_date, start_date)

    #if optimizer == "msr" or optimizer == "mvo":
    optimized_portfolio, performance = bck_tstr.get_portfolio(df, optimizer)
    # expected statistics
    expected_annual_return,annual_volatility, sharpe_ratio = performance
  
        


    #print("Print back-test start date is " +  backtest_start_date.strftime('%Y-%m-%d'))
    print("Start date is " + start_date.strftime('%Y-%m-%d'))
    print("End date is " + end_date.strftime('%Y-%m-%d'))
    print("\nExpected Statistics\n")
    print("Annual Return: " + str(expected_annual_return))
    print("Annual Volatility: " + str(annual_volatility))
    print("Annual Sharpe Ratio: " + str(sharpe_ratio))

    
    print("Head of data frame between this start date and end date looks like:")
    print(df.head())
    print(optimized_portfolio)
    
    return
    

if __name__ == "__main__":
    #try:
    main()
    # except:
    #    print("Please provide new command. Did not accept current command.")


