"""
This script executes ETL (Extract, Transform, Load) processes for Yahoo Finance data.

It utilizes the `YahooPricesETL` and `YahooInfoETL` classes to extract financial data,
transform it, and load it into a database. The configuration for the database connection
is provided via the `DB_CONFIG_MAP` dictionary.

Key Features:
- Option to wipe existing data before loading new data (`WIPE_EXISTING_DATA`).
- Executes ETL processes for both price and information data from Yahoo Finance.

Usage:
- Ensure the `DB_CONFIG_MAP` is correctly configured for the target database.
- Set `WIPE_EXISTING_DATA` to `True` for a full reload or `False` to append data.

Dependencies:
- `YahooPricesETL` and `YahooInfoETL` classes from `utils.data.etl.core`.
"""

from utils.data.etl.core import YahooPricesETL, YahooInfoETL, DB_CONFIG_MAP

WIPE_EXISTING_DATA = True  # e.g. full reloading

print(DB_CONFIG_MAP)

etl_yprices_obj = YahooPricesETL(**DB_CONFIG_MAP['YahooFinance'])
print(f'Running YahooPricesETL {WIPE_EXISTING_DATA=}....')
etl_yprices_obj.run_etl(wipe_existing=WIPE_EXISTING_DATA)
print(f'Running YahooPricesETL {WIPE_EXISTING_DATA=}....Completed!')

print(f'Running YahooInfoETL {WIPE_EXISTING_DATA=}....')
etl_yinfo_obj = YahooInfoETL(**DB_CONFIG_MAP['YahooFinance'])
etl_yinfo_obj.run_etl(wipe_existing=WIPE_EXISTING_DATA)
print(f'Running YahooInfoETL {WIPE_EXISTING_DATA=}....Completed!')
