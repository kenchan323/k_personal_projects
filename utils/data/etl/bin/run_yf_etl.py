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
