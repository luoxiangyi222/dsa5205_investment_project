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

    fig, ax = plt.subplots(figsize=(30, 24))

    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                square=True,
                linewidths=.5,
                linecolor='black',
                cmap='coolwarm',
                annot=True,
                fmt=".2f",
                annot_kws={"fontsize": 8}
                )

    ax.set_title(f'Correlation Heatmap of {stock_name}', fontsize=24)

    plt.savefig(f'../trading_strategy_figure/{stock_name}_corr_heatmap.jpeg')


def show_pair_plot(df, stock_name):

    fig, ax = plt.subplots(figsize=(30, 24))
    sns.pairplot(df, kind='reg')
    ax.set_title(f'Correlation Heatmap of {stock_name}', fontsize=24)
    plt.show()
    # plt.savefig(f'../trading_strategy_figure/{stock_name}_corr_heatmap.jpeg')


for stock in crawler.RANDOM_SELECTED_STOCKS:
    df = pd.read_csv(f'../trading_strategy_data/combined_data/{stock}_combined_data.csv')
    # show_correlation_heatmap(df, stock)
    show_pair_plot(df, stock)