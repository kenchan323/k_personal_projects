import datetime as dt
import yfinance
import pandas as pd
from pandas.tseries.offsets import BDay

'''
This script reads the output csv from the other script (script_scrape_senator_trades.py) which had parsed the historical
senator submitted trades. This script then uses the yfinance module (Yahoo Finance open source API) to fetch price
data from Yahoo Finance and backtest the return from going Long (Short) whenever a senator has submitted a Buy (Sell) 
trade. The backtesting is done over several horizons (e.g. subsequent 1M/3M/6M/1Y) and should be compared against the 
return of S&P 500 to determine if positive excess return persists.

@kenchan323
2020-12-26
'''
PATH_TRADE_DATA = r"C:\Users\User\Documents\python_projects\senator_trade_backtest.csv"
_excluded_types = ["Corporate Bond"]
_stock_exchanges = ["NASDAQ", "NYSE"]
_excluded_transaction_types = ["Exchange"]

_forward_eval_horizon_d = {"1M": 22,
                           "3M": 66,
                           "6M": 126,
                           "1Y": 252}
ticker_spx = "^GSPC"


def _get_daily_return_series(ticker, dt_start, dt_end):
    '''
    Use the yfinance module to return a pandas Series of daily return in % (e.g. 0.01 == 1%)
    :param ticker: str - ticker that is recognised by Yahoo Finance (e.g. "^GSPC" for SPX)
    :param dt_start: datetime - start date of time series
    :param dt_end: datetime - end date of time series
    :return: pandas Series - daily percentage return of ticker
    '''
    yfin_ticker = yfinance.Ticker(ticker)
    # Get some historical time-series daily data for SPX
    df_price_hist_spx = yfin_ticker.history(start=dt_start.strftime(format="%Y-%m-%d"),
                                            end=dt_end.strftime(format="%Y-%m-%d"))
    # Calculate the 1D percentage return for SPX
    df_price_hist_spx[ticker] = (df_price_hist_spx["Close"] / df_price_hist_spx["Close"].shift(1)) - 1
    return df_price_hist_spx[ticker]


'''
Data clean what's been parsed by the other script
'''
df_trades = pd.read_csv(PATH_TRADE_DATA, parse_dates=["Transaction Date"], dayfirst=False)
df_trades["Asset Type"] = df_trades["Asset Type"].apply(str)
# Clean label for LONG/SHORT per trade
df_trades["DIRECTION"] = df_trades["Type"].apply(lambda x: "LONG" if x == "Purchase" else "SHORT")
# Drop the excluded "Asset Type"
df_trades = df_trades[~df_trades["Asset Type"].isin(_excluded_types)]
# Drop certain transaction types
df_trades = df_trades[~df_trades["Type"].isin(_excluded_transaction_types)]
# For reason some rows don't have an "Asset Type" even though the "Asset Name" states the stock exchange and ticker is
# valid. We re-label those as Stock
df_trades["Asset Type"] = df_trades.apply(lambda x: "Stock" if (x["Asset Type"] == "nan")
                                                              and (any(exch in x["Asset Name"] for exch in _stock_exchanges))
                                                           else x["Asset Type"], axis=1)
# Drop the rows that are not exchange traded stocks
df_trades = df_trades[df_trades["Asset Type"] != "nan"]
# Drops rows with no tickers
df_trades = df_trades[df_trades["Ticker"] != "--"]

# Try "BIIB" as that has appeared the most in this segment of data("KORS" doesn't work, not on Yahoo Finance)
earliest_date = df_trades["Transaction Date"].min()
earliest_date_minus_1 = earliest_date - BDay(1)
df_trades_biib = df_trades[df_trades["Ticker"] == "BIIB"]
# Use yfinance to get some price return data for "BIIB" (Biogen Inc)
df_biib_ret = _get_daily_return_series("BIIB",
                                       dt_start=earliest_date_minus_1,
                                       dt_end=dt.datetime.today()).dropna()
# Let's get S&P 500 price return too
df_spx_ret = _get_daily_return_series(ticker_spx,
                                      dt_start=earliest_date_minus_1,
                                      dt_end=dt.datetime.today()).dropna()