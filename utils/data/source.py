"""
Definition of the DataSource abstract class and its various implementation under different data sources
"""

import pandas as pd
import typing
import cachetools
import operator

from pandas.tseries.offsets import BDay

from utils.data.etl.core import YahooPricesETL, DB_CONFIG_MAP

import yfinance as yf
import arcticdb as adb

YF_HISTORY_FLDS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']


class DataSource:
    """
    Abstract class
    """
    # max entries in cache under a least-recently-used method
    LRU_CACHE_MAX_ENTRIES = 50

    def __init__(self, enable_cache=True):
        self.cache = cachetools.LRUCache(maxsize=DataSource.LRU_CACHE_MAX_ENTRIES) if enable_cache else None

    def load_timeseries(self, ids, fld, start=None, end=None, **kwargs) -> pd.DataFrame:
        pass

    def load_meta(self, ids, fld, **kwargs):
        pass

    def load_info(self, ids, **kwargs):
        pass


class YahooFinanceAPI(DataSource):
    """
    Wrapper around the yfinance library
    """

    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def load_timeseries(self, ids: typing.Union[tuple, str], fld: typing.Union[str, None] = 'Close',
                        start: pd.Timestamp = None, end: pd.Timestamp = pd.Timestamp.now(),
                        drop_time=True, **kwargs) -> pd.DataFrame:
        """
        fld can be any from the list of HISTORY_FLDS. Or if None then we return all of the fields.
        :rtype: pd.DataFrame
            - if fld is specified, columns are tickers and index are dates
            - if fld is None, then columns are Multi-Index keys
                    e.g. [(Ticker1, Field1), (Ticker1, Field2), ..., (TickerN, FieldK)]
        e.g.
            yf_api = YahooFinance()
            df = yf_api.load_meta(tuple(['NVDA', 'AAPL']), fld='Close',
                                   start=pd.Timestamp(2025, 1, 30),
                                   end=pd.Timestamp(2025, 2, 28))
        """
        if fld is not None:
            assert fld in YF_HISTORY_FLDS, f'load_timeseries only accepts the follow fields : {YF_HISTORY_FLDS}'

        if isinstance(ids, str):
            yf_ticker = yf.Ticker(ids)
            data = yf_ticker.history(start=start.strftime('%Y-%m-%d') if start else None,
                                     end=(end + BDay(1)).strftime('%Y-%m-%d') if end else None,
                                     **kwargs)
            data.index.name = 'DATE'
            if drop_time:
                # we drop the time (e.g. hours minutes) component in the index
                data.index = data.index.map(lambda x: pd.Timestamp(x.date()))

            if fld is not None:
                data = data[fld]
            return data.loc[:end]
        elif isinstance(ids, tuple):
            data_out = {}
            for _id in ids:
                data_out[_id] = self.load_timeseries(_id, fld, start=start, end=end, **kwargs)
            if fld is None:
                # return a dict if user wants all the timeseries fields
                # multi-index column [(Ticker1, Field1), (Ticker1, Field2), ..., (TickerN, FieldK)]
                return pd.concat({k: pd.DataFrame(v) for k, v in data_out.items()}, names=['TICKER', 'FIELD'], axis=1)
            else:
                return pd.concat(data_out, axis=1).loc[:end]
        else:
            raise ValueError('ids must be either str or list of str')

    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def load_meta(self, ids: typing.Union[tuple, str], fld: str, **kwargs) -> typing.Dict[str, pd.DataFrame]:
        """
        :rtype:  Dict[str, pd.DataFrame] - keys are tickers
        e.g.
            yf_api = YahooFinanceAPI()
            df = yf_api.load_meta(tuple(['NVDA', 'AAPL']), 'dividends')
        """
        assert hasattr(yf.Ticker, fld), 'fld needs to be an accessible property of yfinance.Ticker'
        if isinstance(ids, str):
            yf_ticker = yf.Ticker(ids)
            return {ids: getattr(yf_ticker, fld)}
        elif isinstance(ids, tuple):
            data_out = {}
            for _id in ids:
                data_out.update(self.load_meta(_id, fld, **kwargs))
            return data_out

    def load_batch_history(self, ids: typing.Union[tuple, str],
                            start: pd.Timestamp = None, end: pd.Timestamp = None,
                        drop_time=True, **kwargs) -> pd.DataFrame:
        pass

    def load_info(self, ids, **kwargs):
        """
         :rtype:  pd.DataFrame - columns are tickers, id are fields
            yf_api = YahooFinanceAPI()
            df = yf_api.load_info(tuple(['NVDA', 'AAPL']))
        """
        return pd.DataFrame(self.load_meta(ids, fld='info'))


class YahooFinanceDB(DataSource):
    """
    Loading data from the internally onboarded YahooFinance data
    """
    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def load_timeseries(self, ids: typing.Union[tuple, str], fld: str = 'Close',
                        start: pd.Timestamp = None, end: pd.Timestamp = pd.Timestamp.now(),
                        drop_time=True, **kwargs) -> pd.DataFrame:
        """
        Return a wide format prices timeseries DataFrame from am internally onboarded YahooFinance data source
        e.g.
        yf_db = YahooFinanceDB()
        df = yf_db.load_timeseries(tuple(['^SPX', '^FTSE']), fld='Close',
                                   start=pd.Timestamp(2013,1,1), end=pd.Timestamp(2025,1,1))
        """
        if fld is not None:
            assert fld in YF_HISTORY_FLDS, f'load_timeseries only accepts the follow fields : {YF_HISTORY_FLDS}'

        if isinstance(ids, str):
            ids = [ids]

        adb_lib = YahooPricesETL(**DB_CONFIG_MAP['YahooFinance']).get_adb_lib()
        q = adb.QueryBuilder()
        q = q[(q['FIELD'] == fld) & q['TICKER'].isin(ids)]
        df = adb_lib.read('prices', query_builder=q).data

        return df.pivot(index='DATE', columns='TICKER', values='VALUE').loc[start: end]

    def load_info(self, ids, **kwargs):
        """
        Return a wide format ticker info DataFrame from am internally onboarded YahooFinance data source
        e.g.
        yf_db = YahooFinanceDB()
        df = yf_db.load_info(ids=['^SPX', '^VIX'])
        """
        if isinstance(ids, str):
            ids = [ids]

        adb_lib = YahooPricesETL(**DB_CONFIG_MAP['YahooFinance']).get_adb_lib()
        q = adb.QueryBuilder()
        q = q[q['TICKER'].isin(ids)]
        long_df = adb_lib.read('meta', query_builder=q).data
        return long_df.set_index(['FIELD', 'TICKER'])['VALUE'].unstack()

