"""
Functions used to implementing an ETL pipeline based on data extracted from data source (e.g. YahooFinance).
Some basic data cleaning is applied to the dat (Transform).
And then loaded into a local disk storage database (Load) such as ArcticDB
"""
import yaml
from pathlib import Path
import datetime
import time
import typing

import pandas as pd
import numpy as np
import arcticdb as adb

import utils
import imp


def _add_fixed_val_col(df: pd.DataFrame, col_val: typing.Dict[str, float] = {}):
    """
    To add columns to a dataframe with a constant value
    :param df: data to modify
    :param col_val: key new column name(s) with values being constant value to be applied
    """
    df = df.copy()
    for _c, _v in col_val.items():
        df[_c] = _v
    return df


class ETLFromSource:
    """
    Abstract class to define how an implementation of a ETLFromSource should look
    """
    def __init__(self, db_name: str, db_config: str, load_config: str):
        """
        :param db_name: database name
        :param db_config: str path to database config yaml
        :param load_config: str path to data load config yaml
        """
        self.db_name = db_name
        self.db_config = yaml.safe_load(Path(db_config).read_text())
        self.load_config = yaml.safe_load(Path(load_config).read_text())

    def run_etl(self, wipe_existing: bool = False):
        """
        Run the full ETL pipeline
        :param wipe_existing: whether or not to wipe existing data prior to loading fresh data
        """
        self._load(self._transform(self._extract()), wipe_existing=wipe_existing)

    def read_data(self):
        pass

    def _extract(self):
        pass

    def _transform(self, data):
        pass

    def _load(self, data, wipe_existing: bool = False):
        pass

    def incremental_extract(self):
        #TODO - to define how to read existing data and work out the incremental downloads requirement for appending
        pass


class YahooPricesETL(ETLFromSource):
    """
    Defining the ETL pipeline of loading YahooFinance API security prices data are extracted, transformed and
    pushed to a local ArcticDB database instance
    """
    _TABLE = 'prices'
    _LIBRARY = 'asset'

    def _extract(self):
        tickers = self.load_config[self.db_name]['universe']
        table_config = self.load_config[self.db_name][self._LIBRARY]['symbols'][self._TABLE]
        yf = utils.YahooFinance(enable_cache=True)

        # if no chunk_size defined, assuming chunk_size == 1 (e.g. do all tickers in one attempt)
        chunk_size = table_config.get('chunk_size', 1)
        sleep_sec = table_config.get('sleep_sec', 0)

        chunks = np.array_split(tickers, chunk_size)
        _out = []
        kwargs = table_config.get('kwargs', {})

        for _idx, _chunk in enumerate(chunks):
            print(f'Loading data from source......chunk {_idx + 1} out of {len(chunks)}')
            time.sleep(sleep_sec)
            _out.append(yf.load_timeseries(tuple(_chunk),
                                           fld=None,
                                           **kwargs))
        return pd.concat(_out, axis=1)

    def _transform(self, data):
        data_long = data.stack().stack()
        data_long.name = 'VALUE'
        data_long = data_long.reset_index()
        data_long['TICKER'] = data_long['TICKER'].map(str)

        fields = self.db_config[self.db_name][self._LIBRARY][self._TABLE]['fields']['yahoo_finance']

        prices = data_long.query('FIELD in @fields')

        return _add_fixed_val_col(prices, {'SOURCE': 'YahooFinance',
                                                  'TIMESTAMP': datetime.datetime.now().replace(microsecond=0)})

    def _load(self, data, wipe_existing=False):
        arctic_db = adb.Arctic(self.db_config[self.db_name]['loc'])

        # asset (LIBRARY) ---> prices (DF)
        asset_library = arctic_db.get_library(self._LIBRARY, create_if_missing=True)

        table_path = f"{self.db_name}//{self._LIBRARY}//{self._TABLE}"

        if wipe_existing:
            asset_library.delete(self._TABLE)
            print(f'Wiping existing data in {table_path}...')

        print(f'Pushing data ({len(data)} row) to {table_path}...')
        asset_library.write(self._TABLE, data)
        print(f'Pushed data ({len(data)} row) to {table_path}')

    def read_data(self):
        arctic_db = adb.Arctic(self.db_config[self.db_name]['loc'])
        return arctic_db.get_library(self._LIBRARY, create_if_missing=True).read(self._TABLE).data


class YahooInfoETL(ETLFromSource):

    def _extract(self):
        tickers = self.load_config[self.db_name]['universe']
        table_config = self.load_config[self.db_name][self._LIBRARY]['symbols'][self._TABLE]
        yf = utils.YahooFinance(enable_cache=True)

        # if no chunk_size defined, assuming chunk_size == 1 (e.g. do all tickers in one attempt)
        chunk_size = table_config.get('chunk_size', 1)
        sleep_sec = table_config.get('sleep_sec', 0)

        chunks = np.array_split(tickers, chunk_size)
        _out = []
        kwargs = table_config.get('kwargs', {})

        for _idx, _chunk in enumerate(chunks):
            print(f'Loading data from source......chunk {_idx + 1} out of {len(chunks)}')
            time.sleep(sleep_sec)
            _out.append(yf.load_meta(tuple(_chunk),
                                           fld='info',
                                           **kwargs))
        return pd.concat(_out, axis=1)

if __name__ == 'main':

    etl_obj = YahooPricesETL(db_name='finance',
                             db_config=r"C:\dev\k_personal_projects\utils\data\config\db.yaml",
                             load_config=r"C:\dev\k_personal_projects\utils\data\config\load_universe.yaml")
    etl_obj.run_etl(wipe_existing=True)


    """
    db_cfg = yaml.safe_load(Path(r"C:\dev\k_personal_projects\utils\data\config\db.yaml").read_text())

    arctic_db = adb.Arctic(db_cfg['db']['finance']['loc'])

    # asset (LIBRARY) ---> prices (DF)
    asset_library = arctic_db.get_library('asset', create_if_missing=True)

    # EXTRACT
    yf = utils.YahooFinance(enable_cache=True)
    df = yf.load_timeseries(tuple(['^SPX', '^IXIC', '0001.HK', '^VIX', '^MOVE', '^990100-USD-STRD']),
                            fld=None,
                            start=pd.Timestamp(1980,12,31),
                            end=pd.Timestamp(2025, 3, 12))
    # TRANSFORM
    df_long = df.stack().stack()
    df_long.name = 'VALUE'
    df_long = df_long.reset_index()

    FIELDS = db_cfg['db']['finance']['libraries']['fields']['yahoo_finance']

    prices = df_long.query('FIELD in @FIELDS')

    prices = _add_fixed_val_col(prices, {'SOURCE':'YahooFinance',
                                         'TIMESTAMP': datetime.datetime.now().replace(microsecond=0)})

    # LOAD
    asset_library.write('prices', prices)


    # if ac.has_library('asset'):
    #     ac.delete_library('asset')
    #
    # if asset_library.has_symbol('prices'):
    #     asset_library.delete('prices')

    #_prices = asset_library.read('prices').data
    
    """
