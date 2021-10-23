import pandas as pd
import datetime
import trading_strategy.data_crawler as crawler

# Covid 19 data
us_case_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_daily_case_trends.csv')
us_death_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_daily_death_trends.csv')
us_vaccination_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_vaccinations_trends.csv')

# Change date format
us_case_df['Date'] = pd.to_datetime(us_case_df.Date)
us_death_df['Date'] = pd.to_datetime(us_death_df.Date)
us_vaccination_df['Date'] = pd.to_datetime(us_vaccination_df.Date)


def weekly_to_daily(date_df, weekly_df):
    weekly_df['Week'] = pd.to_datetime(weekly_df['Week'])
    # offset forward one day, make it from Sunday to Monday
    weekly_df['Week'] = weekly_df['Week'] + datetime.timedelta(days=1)

    date_df['Date'] = pd.to_datetime(date_df['Date'])
    daily_df = pd.merge(date_df, weekly_df, how='left', left_on='Date', right_on='Week')

    # TODO: fill daily data according to last week data

    return daily_df


def aggregate_data(stock_ticker):

    # ###### price ######
    price_df = pd.read_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_price_data.csv')
    # add price change and change percentage
    price_df['Close_Change'] = price_df['Close'] - price_df['Close'].shift(periods=1, fill_value=0)
    price_df['Close_Change_pctg'] = price_df['Close_Change'] / price_df['Close'].shift(periods=1, fill_value=price_df['Close'][0])
    # min_max scaling for volume
    price_df = crawler.add_min_max_scaling(price_df, 'Volume')

    # ###### TA ######
    TA_df = pd.read_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_TA_indicators.csv')

    ticker_date_df = TA_df[['Ticker', 'Date']].copy()

    # ###### Investor Attention ######
    # change weekly data to daily data
    worldwide_trend_df = pd.read_csv(
        f'../trading_strategy_data/google_trend_data/{stock_ticker}_worldwide_google_trend.csv')

    worldwide_trend_df = weekly_to_daily(ticker_date_df, worldwide_trend_df)

    finance_trend_df = pd.read_csv(
        f'../trading_strategy_data/google_trend_data/{stock_ticker}_finance_google_trend.csv')

    return price_df


aggregate_df = aggregate_data('aapl')
print(aggregate_df.head())
