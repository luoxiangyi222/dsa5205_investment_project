
"""
Author: Luo Xiangyi
Create on: 2021-10-23

This module combines data from different sources into one dataframe.
"""

import pandas as pd
import datetime
import trading_strategy.data_crawler as crawler
from functools import reduce

pd.set_option('display.max_columns', None)  # enable to see all columns
pd.set_option('display.max_rows', None)  # enable to see all columns


def get_pandemic_data():

    # Covid 19 data
    us_case_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_daily_case_trends.csv')
    us_death_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_daily_death_trends.csv')
    # us_vaccination_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_vaccinations_trends.csv')

    # Change date format
    us_case_df['Date'] = pd.to_datetime(us_case_df.Date)
    us_death_df['Date'] = pd.to_datetime(us_death_df.Date)
    # us_vaccination_df['Date'] = pd.to_datetime(us_vaccination_df.Date)

    return [us_case_df, us_death_df]


def aggregate_data(stock_ticker):

    # ###### load daily price data ######
    price_df = pd.read_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_daily_price_data.csv')
    price_df['Date'] = pd.to_datetime(price_df['Date'])

    # add price change and change percentage features
    price_df['Close_Change'] = price_df['Close'] - price_df['Close'].shift(periods=1, fill_value=0)
    price_df['Close_Change_pctg'] = price_df['Close_Change'] / price_df['Close'].shift(periods=1, fill_value=price_df['Close'][0])

    # min_max scaling for volume
    price_df = crawler.add_min_max_scaling(price_df, 'Volume')

    # ###### load daily TA data ######
    TA_df = pd.read_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_TA_indicators.csv')
    TA_df['Date'] = pd.to_datetime(TA_df['Date'])

    # ###### load daily Google Trend data ######
    trend_df = pd.read_csv(
        f'../trading_strategy_data/google_trend_data/{stock_ticker}_daily_trend.csv')
    trend_df['Date'] = pd.to_datetime(trend_df['Date'])

    # merge all data together based on time
    df_list = [TA_df, price_df, trend_df]
    df_list.extend(get_pandemic_data())
    combined_df = reduce(lambda left, right: pd.merge(left, right, how='left', on='Date', validate='one_to_many'), df_list)

    combined_df.to_csv(f'../trading_strategy_data/combined_data/{stock_ticker}_combined_data.csv', index=False)

    print(f'========== {stock_ticker} combined data saved! ==========')
    return combined_df


if __name__ == "__main__":

    for stock in crawler.SELECTED_STOCKS:
        aggregate_df = aggregate_data(stock)


