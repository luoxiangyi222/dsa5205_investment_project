import yfinance as yf
import pandas as pd
import requests


pd.set_option('display.max_columns', None)  # enable to see all columns
# pd.set_option('display.max_rows', None)  # enable to see all columns


SELECTED_STOCK_TICKS = ['aapl', 'cost', 'sbux', 'asml']


# Download function only contain Open, High, Low, Close, Adj Close, Volume
# All are associated with price.
ten_stock_data = yf.download(' '.join(SELECTED_STOCK_TICKS),
                             start="2020-01-01",
                             end="2021-09-30",
                             period="1d",
                             group_by='tickers')
ten_stock_data.to_csv('../stocks_data/ten_stocks_price_data.csv')


# Get fundamental data
# TODO Problem: all these fundamental data is recent, where can I get historical fundamental data
tickers_data = {}
for ticker in SELECTED_STOCK_TICKS:
    ticker_object = yf.Ticker(ticker)

    # convert info() output from dictionary to dataframe
    temp = pd.DataFrame.from_dict(ticker_object.info, orient="index")
    temp.reset_index(inplace=True)
    temp.columns = ["Attribute", "Recent"]

    # add (ticker, dataframe) to main dictionary
    tickers_data[ticker] = temp

for ticker, fund_data in tickers_data.items():
    fund_data.to_csv(f'../stocks_data/fund_data_{ticker}.csv')

# Get Technical Analysis indicators

ALPHA_VANTAGE_API_KEY = '6OFO07W5FJ2R943U'

