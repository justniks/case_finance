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
from copy import deepcopy

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

        for col in self.df_assets_performance.columns.values:
            dict_assets_covariance_imoex[col] = (
                self.df_assets_performance[col]
                .cov(self.df_benchmark_performance[self.benchmark])
            )
        
        mkt_disp = self.df_benchmark_performance[self.benchmark].std(ddof=0) ** 2
        dict_asset_betas = {
            ticker: cov / mkt_disp for ticker, cov in dict_assets_covariance_imoex.items()
        }
        return dict_asset_betas
    

    def check_beta_sustainability(self):
        df_11 = self.df_assets_performance.copy(); df_21 = self.df_benchmark_performance.copy()
        daily_betas = PortfolioOptimization.calc_betas(df_11, df_21)

        df_sust = pd.DataFrame.from_dict(
            daily_betas, orient='index', columns=['daily_beta']
        )

        monthly_1_betas = PortfolioOptimization.calc_betas(
            df_11.drop_duplicates(subset='Year month', keep = 'first'), 
            df_21.drop_duplicates(subset='Year month', keep = 'first')
        )
        series_tmp = pd.Series(monthly_1_betas)
        df_sust['monthly_1_beta'] = series_tmp
        del series_tmp

        monthly_last_betas = PortfolioOptimization.calc_betas(
            df_11.drop_duplicates(subset='Year month', keep = 'last'), 
            df_21.drop_duplicates(subset='Year month', keep = 'last')
        )
        series_tmp = pd.Series(monthly_last_betas)
        df_sust['monthly_last_beta'] = series_tmp
        del series_tmp

        monthly_avg_betas = PortfolioOptimization.calc_betas(
            df_11.resample('M').mean(), df_21.resample('M').mean()
        )
        series_tmp = pd.Series(monthly_avg_betas)
        df_sust['monthly_avg_beta'] = series_tmp
        del series_tmp

        monthly_cumulative_betas = PortfolioOptimization.calc_betas(
            df_11.resample('M').sum(), df_21.resample('M').sum()
        )
        series_tmp = pd.Series(monthly_cumulative_betas)
        df_sust['monthly_cumulative_beta'] = series_tmp
        del series_tmp

        self.df_all_betas = df_sust.round(decimals=4)
    

    def calc_expected_returns(self, rfr=(11.92 / 100), erp=0.10):
        self.rfr = rfr; self.erp = erp
        exp_returns = self.rfr + self.df_all_betas.monthly_avg_beta * self.erp
        self.exp_returns = exp_returns.to_dict()
        return exp_returns * 100


    def make_corr_mat(self):
        corr_mat = self.df_assets_performance.corr()
        plt.figure(figsize=(12, 5))
        sns.heatmap(corr_mat, annot = True)
        plt.title('Correlation Matrix', fontsize=18)
        plt.xlabel('Assets', fontsize=12)
        plt.ylabel('Assets', fontsize=12)
        plt.show();


    def calc_cov_matrix(self):
        self.assets_cov = self.df_assets_performance.cov() * 252
    

    def calc_opt_stock_weights(self):
        exp_ret = self.exp_returns.copy()
        stock_cov = self.assets_cov.copy()
        ef = EfficientFrontier(
            exp_ret, stock_cov, weight_bounds=(0,1)
        )
        opt_stock_weights = ef.max_sharpe(risk_free_rate=self.rfr)
        self.opt_stock_weights = dict(ef.clean_weights())
    

    def make_pie_graph_weights(self):
        dict_cleaned_stock_weights = {
            key: self.opt_stock_weights[key] 
            for key in self.opt_stock_weights 
            if self.opt_stock_weights[key] != 0
        }

        self.cleaned_opt_stock_weights = dict_cleaned_stock_weights

        colors = sns.color_palette('pastel')
        plt.pie(
            self.cleaned_opt_stock_weights.values(), 
            labels=self.cleaned_opt_stock_weights.keys(), 
            colors=colors, autopct=''
        )
        plt.show();
    

    def calc_stock_portfolio_performance(self):
        r_i = self.exp_returns.copy()
        cov_ij = self.assets_cov.copy()
        ef = EfficientFrontier(
            r_i, cov_ij, weight_bounds=(0,1)
        )

        weights = ef.max_sharpe(risk_free_rate=self.rfr)
        cleaned_weights = ef.clean_weights()
        
        ef.portfolio_performance(verbose=True, risk_free_rate=self.rfr)
        # volatility = std(portfolio)

    
    def plot_efficient_frontier(self):
        r_i = self.exp_returns.copy()
        cov_ij = self.assets_cov.copy()
        ef = EfficientFrontier(
            r_i, cov_ij, weight_bounds=(0,1)
        )
        
        fig, ax = plt.subplots()
        ef_max_sharpe = ef.deepcopy()
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, show_tickers=True)

        # Find the tangency portfolio
        ef_max_sharpe.max_sharpe(risk_free_rate=self.rfr)
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
        ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

        # Generate random portfolios
        n_samples = 10_000
        w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(
            np.diag(w @ ef.cov_matrix @ w.T)
        )
        sharpes = rets / stds
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

        # Output
        ax.set_title("Efficient Frontier with random portfolios")
        ax.legend()
        plt.tight_layout()
        plt.show();