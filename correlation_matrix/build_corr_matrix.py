import pandas as pd
import os


dir_project = os.path.dirname(__file__)
dir_csv_folders = os.path.join(dir_project, "dj_mem_ret_yahoo_finance")

def parse_csv_returns(dir_returns_files=dir_csv_folders, rho_matrix=False, start_date=None, end_date=None, use_name=False):
    for root, dirs, files in os.walk(dir_csv_folders):
        csv_files = files # list of csv file names
        break
    df_output = pd.DataFrame()

    if use_name:
        # Load ticker to stock name mapping
        series_ticker_map = pd.read_csv(os.path.join(dir_returns_files, "mapping", "ticker_name.csv"), index_col=0)[
            "Name"]
        dict_ticker_name_map = series_ticker_map.to_dict()

    for csv_file in csv_files:
        if csv_file == "ticker_name.csv":
            continue
        ticker = csv_file.split(".")[0]
        # Set index col to be date
        df_stock = pd.read_csv(os.path.join(dir_returns_files, csv_file), index_col=0, parse_dates=True)

        if use_name:
            # We get the stock name instead of using its ticker
            df_output[dict_ticker_name_map[ticker]] = df_stock["Close"]
        else:
            df_output[ticker] = df_stock["Close"]

    if start_date != None:
        df_output = df_output.loc[start_date:]

    if end_date != None:
        df_output = df_output.loc[:end_date]

    max_date = df_output.index.max()
    min_date = df_output.index.min()
    print("Based on time period {} to {}".format(min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")))

    # Turn daily close prices to daily percentage return
    df_output = df_output.pct_change()
    if rho_matrix:
        return df_output.corr()
    return df_output