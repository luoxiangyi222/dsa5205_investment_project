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
    df[f'trend_{lag}_days_later'] = 0
    df.loc[(df['Close'].shift(periods=-lag) - df['Close'] > 0), f'trend_{lag}_days_later'] = 1
    return df


def labelling_prices(df, lag: int):
    df[f'price_{lag}_days_later'] = df['Close'].shift(periods=-lag)
    return df


def add_label_for_RFE():
    for stock in crawler.RANDOM_SELECTED_STOCKS:

        combined_df = pd.read_csv(f'../trading_strategy_data/combined_data/{stock}_combined_data.csv')

        labelled_df_rfe = labelling_prices(combined_df, 1)

        labelled_df_rfe.to_csv(f'../trading_strategy_data/combined_data/{stock}_labelled_data.csv', index=False)




if __name__ == "__main__":
    add_label_for_RFE()


