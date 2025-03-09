import yfinance

import datetime as dt
import time
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
PATH_TRADE_DATA = r"C:\Users\User\Documents\python_projects\senator_trade_submitted_trades_raw.csv"
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
# Bucket the trade sizes
_trade_amount_buckets = {"$1,001 - $15,000": 0,
                         "$15,001 - $50,000": 1,
                         "$50,001 - $100,000": 2,
                         "$100,001 - $250,000": 3,
                         "$250,001 - $500,000": 4,
                         "$500,001 - $1,000,000": 5,
                         "$1,000,001 - $5,000,000": 6,
                         "$5,000,001 - $25,000,000": 7}
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
        try:
            start_iloc = df_stock_ret.index.get_loc(dt_trigger)
        except KeyError:
            # We don't have stock return data for this ticker on the date the senator trade was submitted
            if abs(dt_trigger - df_stock_ret.index.min()).days <= 2:
                # If the earliest available date in the stock return series is close enough to the trade submission date
                # then let's call it that
                start_iloc = 0
            else:
                return {}

        # We minus one because we want to capture return from t0 to tn-1 (so n days)
        end_iloc = start_iloc + int_days - 1
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
# We bucket each trade into buckets based on the amount
df_trades["Trade_Size_Bucket"] = df_trades["Amount"].map(_trade_amount_buckets)
# We will focus on bucket 3 or above (> 100k)
df_trades = df_trades[df_trades["Trade_Size_Bucket"] >= 3]

df_trades = df_trades.reset_index(drop=True)

# The earliest date across the full table of senator trades
earliest_date_minus_1_all = df_trades["Transaction Date"].min() - BDay(1)

# Let's get S&P 500 price return too so we can calculate excess return
df_spx_ret = _get_daily_return_series(ticker_spx,
                                      dt_start=earliest_date_minus_1_all,
                                      dt_end=dt.datetime.today()).dropna()

dict_stock_ret = {}
for ticker_i, df_ticker in df_trades.groupby("Ticker"): # focusing on all submitted trades of one stock
    earliest_date_minus_1 = df_ticker["Transaction Date"].min() - BDay(1)
    print(f"Getting price return for {ticker_i} since {earliest_date_minus_1.strftime('%Y-%m-%d')}...")
    # Use yfinance to get price return data for this particular ticker
    df_stock_ret = _get_daily_return_series(ticker_i,
                                            dt_start=earliest_date_minus_1,
                                            dt_end=dt.datetime.today()).dropna()
    time.sleep(2) # pace out the requests?
    if len(df_stock_ret) == 0:
        # Maybe this stock has been de-listed
        continue
    # Will store this return series in a dict
    dict_stock_ret[ticker_i] = df_stock_ret

# Ok now we loop over each senator trade and back-testing based on the stock return series we retrieved

# Only do back-testing IF we have stock return for a ticker
list_dict_res = [_backtest_post_trigger_rets(df_stock_ret=dict_stock_ret[row["Ticker"]],
                                             dt_trigger=row["Transaction Date"],
                                             trg_direction=row["DIRECTION"],
                                             df_bmk_ret=df_spx_ret,
                                             Senator=row["Senator"]) for idx, row in df_trades.iterrows()
                 if row["Ticker"] in dict_stock_ret.keys()]

df_blank = pd.DataFrame()
# We add a blank pandas DataFrame to the first element so we can use reduce to append all the dict's together into a
# DataFrame
list_dict_res.insert(0, df_blank)
# Append recursively all the trade summary into a big pandas DataFrame
import functools
df_output_summary = functools.reduce(lambda df, dict_res:
                                     df.append(pd.DataFrame.from_dict(dict_res, orient="index"), ignore_index=True),
                                     list_dict_res)
# Now we have a trade summary of all the backtest driven trades
df_output_summary.to_csv("senator_backtest_summary.csv", index=False)

'''
Some analysis of the strategy
'''
import seaborn as sns
import matplotlib.pyplot as plt
df_output_summary["ER"] = df_output_summary["Trade_Return"] - df_output_summary["Bmk_Return"]
# Only focus on the subsequent 1 month trades
df_output_summary_1m = df_output_summary[df_output_summary["Horizon"] == "1M"]
# Only focus on the subsequent 1 week trades
df_output_summary_1w = df_output_summary[df_output_summary["Horizon"] == "1W"]
sns.distplot(df_output_summary_1m["ER"])
plt.suptitle("1m Excess Return (v long SPX) Distribution of L/S trades from following Senator trades (>100k size)")
# See which Senator provides best l/s signals over a 1 week horizon
df_senator_1w = df_output_summary_1w.groupby("Senator")["ER"].describe()
# Same but over 1 month horizon
df_senator_1m = df_output_summary_1m.groupby("Senator")["ER"].describe()
