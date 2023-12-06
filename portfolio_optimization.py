import numpy as np
import pandas as pd
# from pandas_datareader import data

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
# from copy import deepcopy

import pandas_datareader as pdr


class PortfolioOptimization:
    def __init__(self, arr_stocks: np.array, benchmark='IMOEX'):
        self.arr_assets = arr_stocks
        self.benchmark = benchmark

    def make_stock_table(self, start_date, end_date):
        df = pd.DataFrame()
        for stock in self.arr_assets:
            cur_series = pdr.get_data_moex(stock, start=start_date, end=end_date)['CLOSE']
            df[stock] = cur_series
        cur_series = pdr.get_data_moex(self.benchmark, start=start_date, end=end_date)['CLOSE']
        df_benchmark = cur_series.to_frame()
        del cur_series

        df_benchmark.rename(columns={'CLOSE': 'IMOEX'}, inplace=True)

        self.df_assets = df
        self.df_benchmark = df_benchmark

    def plot_stock_performance(self):
        plt.figure(figsize=(12, 5))
        for col in self.df_assets.columns.values:
            sns.lineplot(data=self.df_assets[col], label=col)
        sns.lineplot(data=self.df_benchmark[self.benchmark], label=self.benchmark, linestyle='--')
        plt.title('Price of the Stocks', fontsize=18)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price in RUB', fontsize=12)
        plt.legend(
            np.append(self.df_assets.columns.values, self.benchmark), loc='upper left',
            prop={'size': 6}
        )
        plt.show();

    def plot_stock_return_performance(self):
        stock_returns = self.df_assets.pct_change()[1:].cumsum()
        benchmark_returns = self.df_benchmark.pct_change()[1:].cumsum()

        plt.figure(figsize=(12, 5))
        for col in stock_returns.columns.values:
            sns.lineplot(data=stock_returns[col], label=col)
        sns.lineplot(data=benchmark_returns[self.benchmark], label=self.benchmark, linestyle='--')
        plt.title('Returns of the Stocks', fontsize=18)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('% Change', fontsize=12)
        plt.legend(
            np.append(self.df_assets.columns.values, self.benchmark), loc='upper left',
            prop={'size': 6}
        )
        plt.show();

    def calc_stock_returns(self):
        self.df_assets_performance = self.df_assets.pct_change()[1:]
        self.df_benchmark_performance = self.df_benchmark.pct_change()[1:]
    
    def calc_betas(self):
        dict_assets_covariance_imoex = {}
        for col in self.df_assets.columns.values:
            dict_assets_covariance_imoex[col] = df1[col].cov(df2[df2_col])
        
        mkt_disp = df2[df2_col].std(ddof=0) ** 2
        dict_asset_betas = {
            ticker: cov / mkt_disp for ticker, cov in dict_assets_covariance_imoex.items()
        }
        return dict_asset_betas
