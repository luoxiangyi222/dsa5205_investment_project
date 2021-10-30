
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from datetime import datetime
import requests
from yahoo_fin import stock_info as si
import time
from functools import reduce
import math
import csv
import itertools
import quandl


################################################
# Get all stocks data for stocks in stock pool
###############################################

# get data from yahoo finance
# from 2020-1-1 to 2021-7-31
# load stock pool list
tickers = []
with open('./data/stock_pool.txt') as file:
    for line in file:
        tickers.append(line.rstrip())
print(len(tickers))

# tickers = ["FB", "AMZN", "AAPL", "NFLX", "GOOG"]
# price_data = {ticker: si.get_data(ticker, start_date="2020-01-01", end_date="2021-07-31") for ticker in tickers}
# combined = reduce(lambda x, y: x.append(y), price_data.values())
# or directly load data
combined = pd.read_csv('./data/stock_pool_data.csv', index_col=0)
combined['time'] = pd.Index(pd.to_datetime(combined.index))

combined = combined.set_index('time')
combined = combined.interpolate(method='time')


# calculate stock returns
data_raw = combined[['ticker', 'adjclose']]

data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
# flatten columns multi-index, `date` will become the dataframe index
data.columns = [col[1] for col in data.columns.values]

#####################################################
# Calculate monthly returns
# Sort by monthly returns and 2-month returns
######################################################
# calculate monthly stock returns
monthly_stock_return = data.resample('M').ffill().pct_change().iloc[-1].sort_values(ascending=False).dropna()
# calculate stock returns over 2-month
index = data[data.index > datetime(2021, 5, 30)].index
data_2m = data.loc[index]
# two_month_return = ((data_2m.iloc[-1] - data_2m.iloc[0]) / data_2m.iloc[0]).sort_values(ascending=False)
two_month_return = data.resample('2M').ffill().pct_change().iloc[-1].sort_values(ascending=False).dropna()

# descriptive statistic of monthly stock return
stat_1m = monthly_stock_return.describe()
print(stat_1m)
# outlier detection and remove
IQR = stat_1m.loc['75%'] - stat_1m.loc['25%']
lower = stat_1m.loc['25%'] - 1.5 * IQR
upper = stat_1m.loc['75%'] + 1.5 * IQR
original_stocks = monthly_stock_return.size
monthly_stock_return = monthly_stock_return[(monthly_stock_return > lower) & (monthly_stock_return < upper)]
stocks_no_outliers = monthly_stock_return.size
print(print("Numbers of outliers in 1 month stock returns is {}".format(original_stocks - stocks_no_outliers)))

# descriptive statistic of 2-monthly stock return
stat_2m = two_month_return.describe()
print(stat_2m)
# outlier detection and remove
IQR = stat_2m.loc['75%'] - stat_2m.loc['25%']
lower = stat_2m.loc['25%'] - 1.5 * IQR
upper = stat_2m.loc['75%'] + 1.5 * IQR
original_stocks = two_month_return.size
two_month_return = two_month_return[(two_month_return > lower) & (two_month_return < upper)]
stocks_no_outliers = two_month_return.size
print("Numbers of outliers in 2 month stock returns is {}".format(original_stocks - stocks_no_outliers))

# get worst performing 20%
num = monthly_stock_return.size
contrarian_stocks_1m = monthly_stock_return[math.floor(num * 0.8):].index.tolist()
contrarian_stocks_2m = two_month_return[math.floor(num * 0.8):].index.tolist()
contrarian_stocks = contrarian_stocks_1m
contrarian_stocks.extend(x for x in contrarian_stocks_2m if x not in contrarian_stocks)
print(contrarian_stocks)
print(len(contrarian_stocks))
with open('./data/contrarian_stocks.txt', 'w') as f:
    for item in contrarian_stocks:
        f.write("%s\n" % item)


# get best performing 20%
momentum_stocks_1m = monthly_stock_return[0:math.ceil(num * 0.2)].index.tolist()
momentum_stocks_2m = two_month_return[0:math.ceil(num * 0.2)].index.tolist()
momentum_stocks = momentum_stocks_1m
momentum_stocks.extend(x for x in momentum_stocks_2m if x not in momentum_stocks)
print(momentum_stocks)
print(len(momentum_stocks))
with open('./data/momentum_stocks.txt', 'w') as f:
    for item in momentum_stocks:
        f.write("%s\n" % item)

# get a mix of both
mix_stocks = momentum_stocks
mix_stocks.extend(x for x in contrarian_stocks if x not in mix_stocks)
print(mix_stocks)
print(len(mix_stocks))
with open('./data/mix_stocks.txt', 'w') as f:
    for item in mix_stocks:
        f.write("%s\n" % item)
