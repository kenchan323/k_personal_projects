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

DB_CONFIG_MAP = {'YahooFinance':
                  {'db_name': 'finance',
                   'db_config': r"C:\dev\k_personal_projects\utils\data\etl\config\db.yaml",
                   'load_config': r"C:\dev\k_personal_projects\utils\data\etl\config\load_universe.yaml"}
              }


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
    def __init__(self, db_name: str, db_config: str, load_config: str=None):
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

    def _chunking(self, chunk_size, elements):
        if chunk_size < len(elements):
            return np.array_split(elements, chunk_size)
        else:
            print('chunk larger than size of tickers to iterate over. Overriding chunk with the counts of tickers')
            return np.array_split(elements, len(elements))


class ArticDbETL(ETLFromSource):
    LIBRARY = ''
    TABLE = ''

    def get_adb_lib(self):
        return adb.Arctic(self.db_config[self.db_name]['loc']).get_library(self.LIBRARY, create_if_missing=False)

    def read_data(self):
        adb_library = self.get_adb_lib()
        return adb_library.read(self.TABLE).data

    def _load(self, data, wipe_existing=False):
        arctic_db = adb.Arctic(self.db_config[self.db_name]['loc'])

        # asset (LIBRARY) ---> prices (DF)
        asset_library = arctic_db.get_library(self.LIBRARY, create_if_missing=True)

        table_path = f"{self.db_name}//{self.LIBRARY}//{self.TABLE}"

        if wipe_existing:
            asset_library.delete(self.TABLE)
            print(f'Wiping existing data in {table_path}...')

        print(f'Pushing data ({len(data)} row) to {table_path}...')
        asset_library.write(self.TABLE, data)
        print(f'Pushed data ({len(data)} row) to {table_path}')


class YahooPricesETL(ArticDbETL):
    """
    Defining the ETL pipeline of loading YahooFinance API security prices data are extracted, transformed and
    pushed to a local ArcticDB database instance
    """
    LIBRARY = 'asset'
    TABLE = 'prices'

    def _extract(self):
        assert self.load_config is not None, 'load_config must be specified to perform data extraction!'
        tickers = self.load_config[self.db_name]['universe']
        table_config = self.load_config[self.db_name][self.LIBRARY]['symbols'][self.TABLE]
        yf = utils.YahooFinanceAPI(enable_cache=True)

        # if no chunk_size defined, assuming chunk_size == 1 (e.g. do all tickers in one attempt)
        chunk_size = table_config.get('chunk_size', 1)
        sleep_sec = table_config.get('sleep_sec', 0)

        chunks = self._chunking(chunk_size, tickers)

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

        fields = self.db_config[self.db_name][self.LIBRARY][self.TABLE]['fields']['yahoo_finance']

        prices = data_long.query('FIELD in @fields')

        return _add_fixed_val_col(prices, {'SOURCE': 'YahooFinance',
                                                  'TIMESTAMP': datetime.datetime.now().replace(microsecond=0)})


class YahooInfoETL(ArticDbETL):
    LIBRARY = 'asset'
    TABLE = 'meta'

    def _extract(self):
        """
        :rtype pd.DataFrame : K by N
        """
        assert self.load_config is not None, 'load_config must be specified to perform data extraction!'
        tickers = self.load_config[self.db_name]['universe']
        table_config = self.load_config[self.db_name][self.LIBRARY]['symbols'][self.TABLE]
        yf = utils.YahooFinanceAPI(enable_cache=True)

        # if no chunk_size defined, assuming chunk_size == 1 (e.g. do all tickers in one attempt)
        chunk_size = table_config.get('chunk_size', 1)
        sleep_sec = table_config.get('sleep_sec', 0)

        chunks = self._chunking(chunk_size, tickers)
        _out = {}
        kwargs = table_config.get('kwargs', {})
        for _idx, _chunk in enumerate(chunks):
            print(f'Loading data from source......chunk {_idx + 1} out of {len(chunks)}')
            time.sleep(sleep_sec)
            _out.update(yf.load_meta(tuple(_chunk),
                                     fld='info',
                                     **kwargs))
        return pd.DataFrame(_out)

    def _transform(self, data):
        fields = self.db_config[self.db_name][self.LIBRARY][self.TABLE]['fields']['yahoo_finance']
        data = data.loc[fields]

        data_long = data.T.reset_index().melt(id_vars='index',
                                              var_name='FIELD', value_name='VALUE').rename({'index': 'TICKER'}, axis=1)
        data_long['TICKER'] = data_long['TICKER'].map(str)

        return _add_fixed_val_col(data_long, {'SOURCE': 'YahooFinance',
                                              'TIMESTAMP': datetime.datetime.now().replace(microsecond=0)})


