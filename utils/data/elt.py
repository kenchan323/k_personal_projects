"""
Functions used to implementing an ETL pipeline based on data extracted from data source (e.g. YahooFinance).
Some basic data cleaning is applied to the dat (Transform).
And then loaded into a local disk storage database (Load) such as ArcticDB
"""
import yaml
from pathlib import Path
import datetime

import pandas as pd
import arcticdb as adb

import utils
import imp


def _add_fixed_val_col(df, col_val={}):
    """
    To add columns to a dataframe with a constant value
    :param df: pd.DataFrame
    :param col_val: dict - key new column name(s) with values being constant value to be applied
    """
    df = df.copy()
    for _c, _v in col_val.items():
        df[_c] = _v
    return df


if __name__ == 'main':
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