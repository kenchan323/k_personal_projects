"""
Some utility classes to run a backtest for a given set of asset weight timeseries and a set of given asset price
timeseries

Including in the main method is a toy example of running a backtest
"""
import pandas as pd
import numpy as np
import typing
from typing import Union

import utils.mapping.constants as cst


class Portfolio:
    """
    Portfolio class where various portfolio property calculations are functionalised
    """
    def __init__(self, h_ts: pd.DataFrame):
        """
        :param h_ts: asset weight target timeseries
        """
        self._h_ts = h_ts

    def check_asset_coverage(self, prices: pd.DataFrame):
        return set(self._h_ts.keys()).issubset(set(prices.keys()))

    def drifted_wgts(self, prices: pd.DataFrame):
        """
        Calculate the drifted weights on each date for which we have asset returns data for. On rebalance dates we
        show the rebalance target weights
        :rtype pd.DataFrame
        """
        _h = self._h_ts
        assert self.check_asset_coverage(prices), 'positions must be covered by assets available in price data'
        assert _h.index.min() >= prices.index.min(), ('first date in positions must be within time range covered '
                                                     'by prices data')

        r_ts = prices.pct_change()

        _h_temp = _h.reindex(r_ts.loc[_h.index.min():].index)
        # we want the rebalance dates in the index too
        _h = _h.reindex(_h.index.union(_h_temp.index))

        for _idx, _r_d in enumerate(_h.index):
            if _h.loc[_r_d].isna().all():
                # on dates where only drifts happen
                drifted = _h.iloc[_idx - 1] * (1 + r_ts.loc[_r_d])
                drifted = drifted / drifted.sum()
                _h.loc[_r_d] = drifted
        return _h

    def pre_rebalance_wgts(self, prices: pd.DataFrame):
        """
        Return the timeseries of pre-rebalance asset weights. E.g. the starting weights of the assets on rebalance
        dates, just prior the rebalancing, having just experienced the market drifts of the day.
        :rtype pd.DataFrame with the same dimension as self.h_ts
        """
        r_ts = prices.pct_change()
        dw = self.drifted_wgts(prices)

        h_pre_rebal = {}
        for idx, rebal_dt in enumerate(self._h_ts.index):
            if idx == 0:
                # the first rebalance date
                continue
            prev_drifted = dw.loc[dw.index < rebal_dt].iloc[-1]  # the drifted weight of t-1
            new_drifted = prev_drifted * (r_ts.loc[rebal_dt] + 1)  # apply today's drifts
            new_drifted = new_drifted / new_drifted.sum()
            h_pre_rebal[rebal_dt] = new_drifted
        return pd.DataFrame.from_dict(h_pre_rebal, orient='index')

    @property
    def h_ts(self):
        """
        the rebalance target weights
        """
        return self._h_ts

    def gross_exp(self):
        """
        gross exposures on each date
        """
        return self.h_ts.abs().sum(axis=1)

    def name_count(self):
        """
        name count timeseries
        """
        return self.h_ts.count(axis=1)

    def turnover(self, prices_ts=None, agg=True):
        """
        Portfolio 2-way turnover timeseries. If prices are supplied then turnover calculation will account impact of drifts
        """
        if prices_ts is not None:
            # post-rebalance weight minus pre-rebalance drifted weights = turnover
            turns = self.h_ts.sub(self.pre_rebalance_wgts(prices_ts)).abs()
            if agg:
                return turns.sum(axis=1)
            else:
                return turns
        else:
            turns = self.h_ts.diff().abs()
            if agg:
                return turns.sum(axis=1)
            else:
                return turns


class TCost:
    """
    Transaction cost class to provide calculation of PnL hit from turning over portfolio
    """
    def __init__(self, tcost: Union[float, pd.DataFrame, typing.Dict[str, float]] = 0):
        """
        :param tcost: either a scaler (treating t-cost to the same across all assets and all time-periods),
                      a DataFrame (treating t-cost to potentially vary across asset and time dimension),
                      a dict (t-cost to potentially vary across assets but constant across time dimension
        """
        self.tcost = tcost

    def tcost_hit(self, portfolio: Portfolio, prices_ts: pd.DataFrame):
        """
        Calculate the pnl hit e.g. the performance detraction by assets by time periods incurred from turnover
        :param portfolio: Portfolio object to apply the t-cost hit on
        :param prices_ts: asset prices on each date
        :rtype pd.DataFrame
        """
        _h_ts = portfolio.h_ts
        if isinstance(self.tcost, float):
            # apply same tcost across assets across time
            tc_ts = (_h_ts / _h_ts) * self.tcost
        elif isinstance(self.tcost, dict):
            assert set(self.tcost.keys()).issubset(set(_h_ts.keys())), \
                't-cost dict keys must cover all assets in holdings'
            tc_ts = _h_ts  * np.nan
            tc_ts.iloc[0, :] = pd.Series(self.tcost)
            tc_ts = tc_ts.ffill()
        elif isinstance(self.tcost, pd.DataFrame):
            assert (_h_ts/_h_ts).equals(self.tcost/self.tcost), ('For a tcost timeseries DataFrame, index and columns '
                                                               'must be identical to the index/columns of holdings timeseries')
            tc_ts = self.tcost
        else:
            raise ValueError('only types float/dict/pd.DataFrame are accepted for tcost')

        tcost_hit = portfolio.turnover(agg=False, prices_ts=prices_ts) * tc_ts.values
        return tcost_hit.fillna(0)


class Performance:
    """
    Performance class where various performance metrics calculations are functionalised
    """

    @staticmethod
    def from_price_ts(prices_ts: pd.DataFrame, forced_freq: str=None):
        assert isinstance(prices_ts, pd.DataFrame), 'prices must be a DataFrame with 1 column of assert prices'
        _pf = Portfolio(prices_ts/prices_ts) # just holdings of one's
        if forced_freq:
            prices_ts = prices_ts.resample(forced_freq).last()
        return Performance(pnl_asset_ts=prices_ts.pct_change(), portfolio=_pf, prices_ts=prices_ts)

    def __init__(self, pnl_asset_ts: pd.DataFrame,
                 portfolio: Portfolio,
                 prices_ts: pd.DataFrame,
                 tc_hit_ts: pd.DataFrame = TCost(0),
                 benchmark: typing.Optional['Performance'] = None):
        """
        :param pnl_asset_ts: PnL of each asset during each time-period
        :param tc_hit_ts: PnL hit incurred by turnover in each asset during each time-period
        :param benchmark: Performance benchmark
        """
        # https://peps.python.org/pep-0484/#forward-references
        self.pnl_asset_ts = pnl_asset_ts
        self.portfolio = portfolio
        self.prices_ts = prices_ts
        self.tc_hit_ts = tc_hit_ts
        self.benchmark = benchmark
        self.periods = len(pnl_asset_ts)
        self.ret_freqstr = self.pnl_asset_ts.index.freqstr

    def agg_ts(self, post_tc: bool = False):
        """
        Aggregated performance on during each time-period
        :param post_tc: apply transaction cost
        """
        agg_pnl_ts = self.pnl_asset_ts.sub(self.tc_hit_ts).sum(axis=1) if post_tc else self.pnl_asset_ts.sum(axis=1)
        if self.benchmark:
            assert self.ret_freqstr == self.benchmark.ret_freqstr, \
                (f'Performance return freq ({self.ret_freq}) does not match with that of '
                 f'benchmark performance ({self.benchmark.ret_freq})')
            #TODO expose post_tc flag for self.benchmark
            agg_pnl_ts = agg_pnl_ts.sub(self.benchmark.agg_ts().reindex(agg_pnl_ts.index))

        return agg_pnl_ts

    def cumulative_agg_ts(self, post_tc: bool = False):
        """
        Cumulative performance on upto each time-period
        :param post_tc: apply transaction cost
        """
        return (self.agg_ts(post_tc) + 1).cumprod() - 1

    def underwater_ts(self,  post_tc: bool = False):
        cum_ts = self.cumulative_agg_ts(post_tc) + 1
        return (cum_ts / cum_ts.expanding().max()) - 1

    def annualised_return(self, post_tc:bool = False, ret_type='geometric'):
        annu_factor = cst.annualisation_factor(self.ret_freqstr)

        if ret_type == 'geometric':
            # e.g. 1 for 100%
            _cumul = self.cumulative_agg_ts(post_tc)
            n_periods = self.periods / annu_factor  # e.g. if 18 BM in returns, this becomes 1.5
            return np.power(1+_cumul.iloc[-1], 1 / n_periods) -1
        elif ret_type == 'arithmetic':
            rets = self.agg_ts(post_tc)
            return rets.mean() * annu_factor
        else:
            raise ValueError(f'{ret_type=} not recognised')

    def annualised_vol(self, post_tc:bool = False):
        annu_factor = cst.annualisation_factor(self.ret_freqstr)
        tr = self.agg_ts(post_tc) # e.g. 1 for 100%
        return tr.std() * np.sqrt(annu_factor)

    def ir(self, post_tc:bool = False, ret_type='geometric'):
        return self.annualised_return(post_tc, ret_type) / self.annualised_vol(post_tc)

    def max_dd(self, post_tc: bool = False):
        return abs(self.underwater_ts(post_tc).min())

    def current_dd(self, post_tc: bool = False):
        return abs(self.underwater_ts(post_tc).iloc[-1])

    def stat_summary(self, post_tc:bool = False):
        ir = self.ir(post_tc)
        ann_r = self.annualised_return(post_tc)
        ann_vol = self.annualised_vol(post_tc)

        annu_factor = cst.annualisation_factor(self.ret_freqstr)
        turnover = self.portfolio.turnover(prices_ts=self.prices_ts).sum()
        ann_turn = turnover/self.periods * annu_factor
        ann_tc = 0 if not post_tc else ((self.tc_hit_ts.sum().sum() / self.periods) * annu_factor)

        max_dd = self.max_dd(post_tc)
        current_dd = self.current_dd(post_tc)
        return pd.Series({'IR'                : ir,
                          'Return (ann.)'     : ann_r,
                          'TE (ann.)'         : ann_vol,
                          'Two-way Turnover (ann.)'   : ann_turn,
                          'T-Cost Hit (ann.)' : ann_tc,
                          'Max Drawdown'      : max_dd,
                          'Current Drawdown'  :current_dd,
                          'Type'              : 'Relative' if self.benchmark is not None else 'Absolute'})


class Backtest:
    """
    Backtest class to return a Performance object for a given portfolio, asset prices and tcost assumptions
    """
    def __init__(self, portfolio: Portfolio, prices: pd.DataFrame, tcost: typing.Optional[TCost] = None):
        """
        :param portfolio: Portfolio object containing the asset holdings on each date
        :param prices: asset prices on each date
        :param tcost: TCost to be applied
        """
        assert portfolio.h_ts.index.max() <= prices.index.max(), 'latest position date must be covered by prices date index'
        assert portfolio.h_ts.index.min() >= prices.index.min(), 'earliest position date must be covered by prices date index'
        self.portfolio = portfolio
        self.prices_ts = prices.ffill()
        self.tcost = tcost if tcost is not None else TCost(0)

    def run(self, benchmark=None):
        """
        Run the backtest
        :rtype Performance
        """
        r_ts = self.prices_ts.reindex(self.portfolio.h_ts.index).pct_change()
        h_ts_shifted = self.portfolio.h_ts.shift(1)
        r_ts_h_indexed = r_ts.ffill().reindex(h_ts_shifted.index, method='ffill')
        pnl_asset_ts = h_ts_shifted * r_ts_h_indexed.values
        tc_hit_ts = self.tcost.tcost_hit(self.portfolio, self.prices_ts)
        return Performance(pnl_asset_ts,
                           portfolio=self.portfolio,
                           prices_ts=self.prices_ts,
                           tc_hit_ts=tc_hit_ts,
                           benchmark=benchmark)


#if __name__ == 'main':
# tc = 0.004
#
# h_ts = pd.DataFrame(index=[pd.date_range(start=pd.Timestamp(2024, 11, 29),
#                                          end=pd.Timestamp(2025, 2, 28),
#                                          freq='BM')],
#                     columns=['AAPL', 'MSFT', 'NVDA'],
#                     data=[[0.25, 0.25, 0.5],
#                           [0.3, 0.6, 0.1],
#                           [0.23, 0.3, 0.47],
#                           [0.4, 0.2, 0.4]])
#
# p_ts = pd.DataFrame(columns=['AAPL', 'MSFT', 'NVDA'],
#                     index=pd.date_range(start=pd.Timestamp(2024, 11, 29),
#                                         end=pd.Timestamp(2025, 2, 28), freq='BM'),
#                     data=[[239., 393., 112.],
#                           [246.88142583, 398.38228482, 120.40866141],
#                           [261.80621796, 408.21266778, 122.76499626],
#                           [289.62756761, 438.6417914, 114.35634255]])

# pf = Portfolio(h_ts)
#
# tcost = TCost(0.01)
#
# bkt = Backtest(pf, p_ts, tcost)
#
# perf = bkt.run()
#
#
# p_hf_ts = pd.DataFrame(columns=['AAPL', 'MSFT', 'NVDA'],
#                     index=pd.date_range(start=pd.Timestamp(2024, 11, 29),
#                                         end=pd.Timestamp(2025, 2, 28), freq='BM'),
#                     data=[[239., 393., 112.],
#                           [246.88142583, 398.38228482, 120.40866141],
#                           [261.80621796, 408.21266778, 122.76499626],
#                           [289.62756761, 438.6417914, 114.35634255]])
#
#
# cum_pnl = perf.cumulative_agg_ts(True)

# import numpy as np
# import pandas as pd
# from utils.data.etl.core import YahooInfoETL, YahooPricesETL,  DB_CONFIG_MAP
# etl_yprices_obj = YahooPricesETL(**DB_CONFIG_MAP['YahooFinance'])
# #etl_yprices_obj.run_etl(wipe_existing=True)
# raw_prices = etl_yprices_obj.read_data()
#
# df = raw_prices.query('(FIELD == "Close") & ((TICKER == "^SPX") | (TICKER == "^STOXX50E"))').copy()
# df['VALUE'] = df['VALUE'].replace('NaN', np.nan)
# index_prices_ts = df.pivot(index='DATE', columns='TICKER',values='VALUE')
# _date_range = pd.date_range(pd.Timestamp(2010, 12, 31), pd.Timestamp.now(), freq='BM')
# bm_index_prices_ts = index_prices_ts.ffill(limit=1).reindex(_date_range, method='ffill')
# # 50% 50%
# h_df = bm_index_prices_ts/bm_index_prices_ts * 0.5
#
# pf = Portfolio(h_df)
#
# #pf.drifted_wgts(index_prices_ts)
#
# tcost = TCost(0.001)
#
# bkt = Backtest(pf, index_prices_ts, tcost)
#
# perf = bkt.run()