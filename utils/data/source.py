"""
Definition of the DataSource abstract class and its various implementation under different data sources
"""

import pandas as pd
import typing
import cachetools
import operator

from pandas.tseries.offsets import BDay

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
            yd_data = YahooFinance()
            df = yd_data.load_meta(tuple(['NVDA', 'AAPL']), fld='Close',
                                   start=pd.Timestamp(2025, 1, 30),
                                   end=pd.Timestamp(2025, 2, 28))
        """
        if fld is not None:
            assert fld in YahooFinance.HISTORY_FLDS, (f'load_timeseries only accepts '
                                                      f'the follow fields : {YahooFinance.HISTORY_FLDS}')

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

    def load_batch_history(self, ids: typing.Union[tuple, str],
                            start: pd.Timestamp = None, end: pd.Timestamp = None,
                        drop_time=True, **kwargs) -> pd.DataFrame:
        pass

