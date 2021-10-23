import pandas as pd

import trading_strategy.data_crawler as crawler

# Covid 19 data
us_case_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_daily_case_trends.csv')
us_death_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_daily_death_trends.csv')
us_vaccination_df = pd.read_csv('../trading_strategy_data/us_covid_19_data/us_vaccinations_trends.csv')

# change date time format


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

    # ###### Investor Attention ######
    worldwide_trend_df = pd.read_csv(
        f'../trading_strategy_data/google_trend_data/{stock_ticker}_worldwide_google_trend.csv')
    finance_trend_df = pd.read_csv(
        f'../trading_strategy_data/google_trend_data/{stock_ticker}_finance_google_trend.csv')



    return price_df


aggregate_df = aggregate_data('aapl')
print(aggregate_df.head())
