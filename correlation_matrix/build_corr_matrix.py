import pandas as pd
import os


dir_project = os.path.dirname(__file__)
dir_csv_folders = os.path.join(dir_project, "dj_mem_ret_yahoo_finance")

def parse_csv_returns(dir_returns_files=dir_csv_folders, rho_matrix=False):
    for root, dirs, files in os.walk(dir_csv_folders):
        csv_files = files # list of csv file names
    df_output = pd.DataFrame()

    for csv_file in csv_files:
        ticker = csv_file.split(".")[0]
        # Set index col to be date
        df_stock = pd.read_csv(os.path.join(dir_returns_files, csv_file), index_col=0)
        df_output[ticker] = df_stock["Close"]

    # Turn daily close prices to daily percentage return
    df_output = df_output.pct_change()
    if rho_matrix:
        return df_output.corr()
    return df_output

df_returns = parse_csv_returns()
df_corr = df_returns.corr()