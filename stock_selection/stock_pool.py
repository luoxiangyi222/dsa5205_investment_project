import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from datetime import datetime
import yfinance as yf
import requests
from yahoo_fin import stock_info as si
import time
from functools import reduce
import math
from portfolio_backtest import Backtest
import pprint
import quandl

################################################
# Get all stocks data
###############################################

# get data from yahoo finance
# from 2020-1-1 to 2021-7-31
# tickers = []
# with open('./data/stock_list.txt') as file:
#     for line in file:
#         tickers.append(line.rstrip())
tickers = si.tickers_nasdaq()
# tickers = ["FB", "AMZN", "AAPL", "NFLX", "GOOG"]
# price_data = {ticker: si.get_data(ticker, start_date="2020-01-01", end_date="2021-07-31") for ticker in tickers}
i = 0
price_data = {}
temp = []
na = []
counter = 0
for ticker in tickers:
    try:
        print(ticker)
        dat = si.get_data(ticker, start_date="2020-01-01", end_date="2021-07-31")
        temp.append(ticker)
    except (AssertionError, KeyError, TypeError):
        i = i+1
        print(ticker+" fails " + str(i) +" out of "+ str(len(tickers)))
        na.append(ticker)
        continue
    price_data[ticker] = dat
    counter += 1
    if counter%100 == 0:
        print("-" * 80)
        print("{} out of {} is done\n".format(counter, len(tickers)))
        print("-" * 80)
ticker_lst = temp

combined = reduce(lambda x, y: x.append(y), price_data.values())
combined['intraday'] = (combined['close'] - combined['open']) / combined['open']

remove = []
counting = combined['ticker'].value_counts()
for t in combined['ticker'].unique():
    if counting[t] < 398:
        remove.append(t)
print(remove)
tickers = [x for x in ticker_lst if x not in remove]
print(tickers)

combined = combined[combined['ticker'].isin(tickers)]
# calculate stock returns
data_raw = combined[['ticker', 'adjclose']]

data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
# flatten columns multi-index, `date` will become the dataframe index
data.columns = [col[1] for col in data.columns.values]

print(data)
stock_data = combined[combined['ticker'].isin(tickers)]
print(stock_data)
stock_data.to_csv('./data/stock_pool_init.csv', index=True)
# or directly load the data from existing list
# combined = pd.read_csv('./data/stock_pool_init.csv', index_col= 0)
# print(combined)
# combined['time'] = pd.Index(pd.to_datetime(combined.index))
# combined = combined.set_index('time')
# tickers = combined['ticker'].unique()

data_raw = combined[['ticker', 'adjclose']]

data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
# flatten columns multi-index, `date` will become the dataframe index
data.columns = [col[1] for col in data.columns.values]
#############################################
# Calculate Maximum Daily Dropdown and Increase for each stocks
# Filter out Stocks that Have Maximum Drawdown > 20%
#############################################
# calculate stock returns
data_raw = combined[['ticker', 'intraday']]

intraday = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['intraday'])
# flatten columns multi-index, `date` will become the dataframe index
intraday.columns = [col[1] for col in intraday.columns.values]

print(intraday)

ind = {}
for c in intraday.columns:
    # how many days that have maximum drawdown > 20%
    temp = len(intraday[(intraday[c] > 0.2) | (intraday[c] < -0.2)])
    ind[c] = temp

tickers = [t for t in tickers if ind[t] == 0]
print(tickers)
# data = data[tickers]


#####################################################
# Calculate Maximum Drawdown for each stocks
# Filter out Stocks that Have Maximum Drawdown > 25%
#####################################################
# calculate daily returns
daily_stock_return = data.pct_change()

# cumulative stock returns
cum_returns = (daily_stock_return + 1).cumprod()
previous_peaks = cum_returns.cummax()
index = daily_stock_return[daily_stock_return.index > datetime(2020, 7, 30)].index
drawdown = (cum_returns - previous_peaks) / previous_peaks.loc[index]
print(drawdown)

# ind is a dictionary with the ticker
# and how many days that exceed the range
ind = {}
for c in drawdown.columns:
    # how many days that have maximum drawdown > 25%
    temp = len(drawdown[(drawdown[c] > 0.25) | (drawdown[c] < -0.25)])
    ind[c] = temp

tickers = [t for t in tickers if ind[t] == 0]
print(tickers)
data = data[tickers]

# save ticker for stocks that are in stock pool
with open('./data/stock_pool.txt', 'w') as f:
    for item in tickers:
        f.write("%s\n" % item)

# save the data for all these stocks
stock_data = combined[combined['ticker'].isin(tickers)]
print(stock_data)
stock_data.to_csv('./data/stock_pool_data.csv', index=True)
