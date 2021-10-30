"""
Author: Luo Xiangyi
Create on: 2021-10-24

This module plot some figures for analysis.
For both data exploration and result analysis.

"""
import matplotlib.pyplot as plt
import trading_strategy.data_crawler as crawler
import pandas as pd
import seaborn as sns
import numpy as np


def show_correlation_heatmap(df, stock_name):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(20, 16))

    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                square=True,
                linewidths=.5,
                linecolor='black',
                cmap='coolwarm',
                annot=True,
                fmt=".2f",
                annot_kws={"fontsize": 6}
                )

    ax.set_title(f'Correlation Heatmap of {stock_name}', fontsize=20)
    plt.savefig(f'../trading_strategy_figure/corr_heatmap/{stock_name}_corr_heatmap.jpeg', bbox_inches='tight')


if __name__ == "__main__":

    # ########## correlation heatmap ##########

    for stock in crawler.MIX_PORTFOLIO:
        df = pd.read_csv(f'../trading_strategy_data/portfolio_data/momentum/{stock}_combined_data.csv')
        show_correlation_heatmap(df, stock)
        print(f'===== {stock} correlation heatmap saved ! =====')

    #
    # for stock in crawler.MOMENTUM_PORTFOLIO:
    #     df = pd.read_csv(f'../trading_strategy_data/portfolio_data/momentum/{stock}_combined_data.csv')
    #     show_correlation_heatmap(df, stock)
    #     print(f'===== {stock} correlation heatmap saved ! =====')
    #
    # print('==========')
    #
    # for stock in crawler.CONTRARIAN_PORTFOLIO:
    #     df = pd.read_csv(f'../trading_strategy_data/portfolio_data/contrarian/{stock}_combined_data.csv')
    #     show_correlation_heatmap(df, stock)
    #     print(f'===== {stock} correlation heatmap saved ! =====')
    #
    # print('==========')
    #
    # for stock in crawler.RANDOM_STOCKS:
    #     df = pd.read_csv(f'../trading_strategy_data/random_stocks_data/{stock}_combined_data.csv')
    #     show_correlation_heatmap(df, stock)
    #     print(f'===== {stock} correlation heatmap saved ! =====')

