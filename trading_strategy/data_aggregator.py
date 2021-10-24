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
    us_vaccination_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_vaccinations_trends.csv')

    # Change date format
    us_case_df['Date'] = pd.to_datetime(us_case_df.Date)
    us_death_df['Date'] = pd.to_datetime(us_death_df.Date)
    us_vaccination_df['Date'] = pd.to_datetime(us_vaccination_df.Date)

    return [us_case_df, us_death_df,us_vaccination_df]


def weekly_to_daily(weekly_df):

    weekly_df['Week'] = pd.to_datetime(weekly_df['Week'])
    # offset forward one day, make it from Sunday to Monday
    weekly_df['Week'] = weekly_df['Week'] + datetime.timedelta(days=1)

    weekly_df = weekly_df.rename(columns={'Week': 'Date'})

    weekly_df = weekly_df.set_index('Date')
    daily_df = weekly_df.resample('D').ffill().reset_index()

    return daily_df


def aggregate_data(stock_ticker):

    # ###### price ######
    price_df = pd.read_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_price_data.csv')
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    # add price change and change percentage
    price_df['Close_Change'] = price_df['Close'] - price_df['Close'].shift(periods=1, fill_value=0)
    price_df['Close_Change_pctg'] = price_df['Close_Change'] / price_df['Close'].shift(periods=1, fill_value=price_df['Close'][0])
    # min_max scaling for volume
    price_df = crawler.add_min_max_scaling(price_df, 'Volume')

    # ###### TA ######
    TA_df = pd.read_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_TA_indicators.csv')
    TA_df['Date'] = pd.to_datetime(TA_df['Date'])

    # ###### Investor Attention ######
    worldwide_trend_df = pd.read_csv(
        f'../trading_strategy_data/google_trend_data/{stock_ticker}_worldwide_google_trend.csv')
    # change weekly data to daily data
    worldwide_trend_df = weekly_to_daily(worldwide_trend_df)

    finance_trend_df = pd.read_csv(
        f'../trading_strategy_data/google_trend_data/{stock_ticker}_finance_google_trend.csv')
    finance_trend_df = weekly_to_daily(finance_trend_df)

    df_list = [TA_df, price_df, worldwide_trend_df, finance_trend_df]
    df_list.extend(get_pandemic_data())

    all_TA_df = reduce(lambda left, right: pd.merge(left, right, on='Date'), df_list)
    return all_TA_df


aggregate_df = aggregate_data('aapl')
print(aggregate_df.head())
