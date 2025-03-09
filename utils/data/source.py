"""
Definition of the DataSource abstract class and its various implementation under different data sources
"""

import pandas as pd
import typing
import cachetools
import operator

import yfinance as yf


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


class YahooFinance(DataSource):
    """
    Wrapper around the yfinance library
    """
    HISTORY_FLDS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def load_timeseries(self, ids: typing.Union[tuple, str], fld: str,
                        start: pd.Timestamp = None, end: pd.Timestamp = None, **kwargs) -> pd.DataFrame:
        """
        :rtype: pd.DataFrame - keys are tickers and index are dates
        e.g.
            yd_data = YahooFinance()
            df = yd_data.load_meta(tuple(['NVDA', 'AAPL']), 'Close',
                                   start=pd.Timestamp(2025, 1, 30),
                                   end=pd.Timestamp(2025, 2, 28))
        """
        assert fld in YahooFinance.HISTORY_FLDS, (f'load_timeseries only accepts '
                                                  f'the follow fields : {YahooFinance.HISTORY_FLDS}')
        if isinstance(ids, str):
            yf_ticker = yf.Ticker(ids)
            return yf_ticker.history(start=start.strftime('%Y-%m-%d') if start else None,
                                     end=end.strftime('%Y-%m-%d') if end else None,
                                     **kwargs)[fld]
        elif isinstance(ids, tuple):
            data_out = {}
            for _id in ids:
                data_out[_id] = self.load_timeseries(_id, fld, start=start, end=end, **kwargs)
            return pd.concat(data_out, axis=1)
        else:
            raise ValueError('ids must be either str or list of str')

    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def load_meta(self, ids: typing.Union[tuple, str], fld: str, **kwargs) -> typing.Dict[str, pd.DataFrame]:
        """
        :rtype:  Dict[str, pd.DataFrame] - keys are tickers
        e.g.
            yd_data = YahooFinance()
            df = yd_data.load_meta(tuple(['NVDA', 'AAPL']), 'dividends')
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
