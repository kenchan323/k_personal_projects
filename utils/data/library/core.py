import pandas as pd

from utils.data.etl.core import YahooPricesETL, DB_CONFIG_MAP

import arcticdb as adb


def get_prices(tickers, field:str = 'Close',
               start: pd.Timestamp = None, end: pd.Timestamp = None, source: str='YahooFinance'):
    """
    Return a wide format prices timeseries DataFrame from a specified, internally onboarded data source
    e.g.
       get_prices(['^SPX', '^FTSE'], start=pd.Timestamp(2013,1,1), end=pd.Timestamp(2025,1,1))
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if source == 'YahooFinance':
        adb_lib = YahooPricesETL(**DB_CONFIG_MAP['YahooFinance']).get_adb_lib()
        q = adb.QueryBuilder()
        q = q[(q['FIELD'] == field) & q['TICKER'].isin(tickers)]
        df = adb_lib.read('prices', query_builder=q).data
        return df.pivot(index='DATE', columns='TICKER', values='VALUE').loc[start: end]
    else:
        raise ValueError(f'{source=} not recognised for get_prices() !')


