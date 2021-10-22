import pandas as pd

import trading_strategy.data_crawler as crawler


def aggregate_data(stock_ticker):
    price_df = pd.read_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_price_data.csv')
    TA_df = pd.read_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_TA_indicators.csv')


aggregate_data('aapl')