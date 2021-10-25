"""
Author: Luo Xiangyi
Create on: 2021-10-24

This module does data preprocessing and feature selection.
"""
# RFE blog
# https://machinelearningmastery.com/rfe-feature-selection-in-python/

import pandas as pd
import trading_strategy.data_crawler as crawler


def RFE(stock):
    rfe_df = pd.read_csv(f'../trading_strategy_data/combined_data/{stock}_rfe_data.csv')


def RFE_voting(random_stock_list):
    # TODO: run RFE for each stock and voting
    pass


def PCA(df):
    pass


if __name__ == "__main__":

    # RFE for feature selection
    for stock in crawler.RANDOM_SELECTED_STOCKS:
        RFE(stock)


