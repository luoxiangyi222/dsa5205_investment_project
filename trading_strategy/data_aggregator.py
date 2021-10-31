
"""
Author: Luo Xiangyi
Create on: 2021-10-23

This module combines data from different sources into one dataframe.
"""

import pandas as pd
import numpy as np
import trading_strategy.data_crawler as crawler
from functools import reduce
from datetime import date

pd.set_option('display.max_columns', None)  # enable to see all columns
pd.set_option('display.max_rows', None)  # enable to see all columns


def get_pandemic_data():

    # Covid 19 data
    us_case_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_daily_case_trends.csv')
    us_case_df = us_case_df.drop(columns=['State'])
    us_death_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_daily_death_trends.csv')
    us_death_df = us_death_df.drop(columns=['State'])

    # Change date format
    us_case_df['Date'] = pd.to_datetime(us_case_df.Date)
    us_death_df['Date'] = pd.to_datetime(us_death_df.Date)

    return [us_case_df, us_death_df]


def aggregate_data(stock_ticker, folder_name):

    # ###### load daily price data ######
    price_df = pd.read_csv(f'../trading_strategy_data/{folder_name}/{stock_ticker}_daily_price_data.csv')
    print(price_df.head())
    price_df['Date'] = pd.to_datetime(price_df['Date'])

    # add price change and change percentage features
    price_df['Close_Change'] = price_df['Close'] - price_df['Close'].shift(periods=1)
    price_df['Close_Change_pctg'] = price_df['Close_Change'] / price_df['Close'].shift(periods=1)

    # min_max scaling for volume
    price_df = crawler.add_min_max_scaling(price_df, 'Volume')

    # ###### load daily TA data ######
    TA_df = pd.read_csv(f'../trading_strategy_data/{folder_name}/{stock_ticker}_TA_indicators.csv')
    TA_df['Date'] = pd.to_datetime(TA_df['Date'])

    # ###### load daily Google Trend data ######
    trend_df = pd.read_csv(
        f'../trading_strategy_data/{folder_name}/{stock_ticker}_daily_trend.csv')
    trend_df['Date'] = pd.to_datetime(trend_df['Date'])

    # merge all data together based on time
    df_list = [TA_df, price_df, trend_df]
    df_list.extend(get_pandemic_data())
    combined_df = reduce(lambda left, right: pd.merge(left, right, how='left', on='Date', validate='one_to_many'), df_list)

    # insert BBANDS extension based on Upper/Lower Bands and High/Low price
    new_col = np.zeros((combined_df.shape[0], 1))
    new_col[(combined_df['High'] > combined_df['BBANDS_Upper_Band'])] = -1
    new_col[(combined_df['Low'] < combined_df['BBANDS_Lower_Band'])] = 1
    combined_df.insert(37, "BBANDS_compare_high_low_price", new_col)

    # round to 8 dp
    combined_df = combined_df.round(8)

    # truncate data having no covid-19 data
    covid_start_date = '2020-01-24'
    validation_period_stop_date = '2021-10-01'

    # combined_df = combined_df[combined_df.Date > covid_start_date]
    combined_df = combined_df[~((combined_df['Date'] < covid_start_date) |
                                (combined_df['Date'] > validation_period_stop_date))]

    combined_df = combined_df.replace({np.inf: 1, -np.inf: -1})
    combined_df = combined_df.fillna(0)

    combined_df.to_csv(f'../trading_strategy_data/{folder_name}/{stock_ticker}_combined_data.csv', index=False)

    print(f'Shape of combined dataframe: {combined_df.shape}')
    print(f'========== {stock_ticker} combined data saved! ==========')
    return combined_df


if __name__ == "__main__":

    for stock in crawler.MIX_PORTFOLIO:
        aggregate_data(stock, folder_name='portfolio_data/mix')
    print()
    print('====== MIX aggregation done ! =====')


    # for stock in crawler.MOMENTUM_PORTFOLIO:
    #     aggregate_data(stock, folder_name='portfolio_data/momentum')
    # print('====== Momentum aggregation done ! =====')



    #
    # for stock in crawler.CONTRARIAN_PORTFOLIO:
    #     aggregate_data(stock, folder_name='portfolio_data/contrarian')
    # print('====== Contrarian aggregation done ! =====')
    #


    # for stock in crawler.RANDOM_STOCKS:
    #     aggregate_data(stock, folder_name='random_stocks_data')
    # print('====== Random stock aggregation done ! =====')
