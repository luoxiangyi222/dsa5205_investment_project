"""
Author: Luo Xiangyi
Create on: 2021-10-24

This module does data preprocessing and feature selection.
"""
# RFE blog
# https://machinelearningmastery.com/rfe-feature-selection-in-python/

import pandas as pd
import trading_strategy.data_crawler as crawler

# TODO: price prediction or classification ?


def labelling_trends(df, lag: int):
    """
    Label the price trend data, for classification task.
    """
    df[f'trend_lag_{lag}'] = 0


if __name__ == "__main__":

    for stock in crawler.SELECTED_STOCKS:

        combined_df = pd.read_csv(f'../trading_strategy_data/combined_data/{stock}_combined_data.csv')
        for n in range(10):
            labelled_df = labelling_trends(combined_df, n)