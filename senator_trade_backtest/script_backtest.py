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
# Some default performance evaluation horizons
_forward_eval_horizon_d = {"2D": 2,
                           "3D": 3,
                           "1W": 5,
                           "1M": 22,
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

def _backtest_post_trigger_rets(df_stock_ret, dt_trigger, trg_direction, df_bmk_ret=None, **kwargs):
    '''
    Backtest a trade's return on a stock given a trigger date and the trade direction (1 for Long/-1 for Short). Trade
    performance evaluation is done over the default set of horizons (defined at the top of the script)
    :param df_stock_ret: pandas Series - the time-series return of a stock (datetime indexed)
    :param dt_trigger: datetime - trigger start date
    :param trg_direction: str - direction of trade ("LONG" or "SHORT")
    :param df_bmk_ret: pandas Series - the time-series return of benchmark (datetime indexed). If not supplied,
    benchmark return is 0
    :param kwargs: any additional info the function caller wishes to supply to have included in the output dictionary
    e.g. "Senator"="Pelosi"
    :return: dict - nested dictionary object. Each dict object to contain trade performance summary over a specific
    horizon
    '''
    dict_out = {}
    bmk_ticker = "NONE"
    int_trade_direction = 1 if trg_direction == "LONG" else -1 if trg_direction == "SHORT" else 0
    # Let's create a nested dict of results then convert into a pandas DataFrame
    i = 0 # counter to index entry to output DataFrame
    for str_horizon, int_days in _forward_eval_horizon_d.items():
        start_iloc = df_stock_ret.index.get_loc(dt_trigger)
        end_iloc = start_iloc + int_days
        if end_iloc >= len(df_stock_ret.index):
            # Too far into the future, we don't have these price data
            continue
        end_dt = df_stock_ret.index[end_iloc]
        df_stock_ret_tmp = df_stock_ret.iloc[start_iloc: end_iloc + 1] + 1
        stock_ret = (df_stock_ret_tmp.prod() - 1) * int_trade_direction
        if isinstance(df_bmk_ret, pd.Series):
            bmk_ticker = df_bmk_ret.name
            start_iloc_bmk = df_bmk_ret.index.get_loc(dt_trigger)
            #end_iloc_bmk = start_iloc_bmk + int_days
            end_iloc_bmk = df_bmk_ret.index.get_loc(end_dt)
            df_bmk_ret_tmp = df_bmk_ret.iloc[start_iloc_bmk: end_iloc_bmk + 1] + 1
            #bmk_ret = (df_bmk_ret.prod() - 1) * int_trg_direction
            bmk_ret = (df_bmk_ret_tmp.prod() - 1)
        else:
            bmk_ret = 0
        dict_out[i] = {"Ticker": df_stock_ret.name,
                       "Start_Date": dt_trigger.strftime("%Y-%m-%d"),
                       "End_Date": end_dt.strftime("%Y-%m-%d"),
                       "Trade": trg_direction,
                       "Horizon": str_horizon,
                       "Trade_Return": stock_ret,
                       "Bmk_Ticker": bmk_ticker,
                       "Bmk_Return": bmk_ret
                       }
        # Plus any additional information supplied
        for k, v in kwargs.items():
            dict_out[i][k] = v
        i += 1
    return dict_out

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

import functools
list_dict_res = [_backtest_post_trigger_rets(df_stock_ret=df_biib_ret,
                                             dt_trigger=row["Transaction Date"],
                                             trg_direction=row["DIRECTION"],
                                             df_bmk_ret=df_spx_ret,
                                             Senator="Sheldon,Whitehouse") for idx, row in df_trades_biib.iterrows()]
df_blank = pd.DataFrame()
# We add a blank pandas DataFrame to the first element so we can use reduce to append all the dict's together into a
# DataFrame
list_dict_res.insert(0, df_blank)
# Append recursively all the trade summary into a big pandas DataFrame
df_output_summary = functools.reduce(lambda df, dict_res:
                                     df.append(pd.DataFrame.from_dict(dict_res, orient="index"), ignore_index=True),
                                     list_dict_res)