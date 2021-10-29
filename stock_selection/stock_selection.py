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

#######################################################
# Stock Selection based on P/E and P/B
#######################################################


def stock_selection(ta_data):
    # Normalize all ta indicators using median & MAD
    ta_df = ta_data.copy()
    df = pd.DataFrame([])
    for col in ta_df.columns:
        df[col] = pd.to_numeric(ta_df[col])

    # calculate B/M and E/P
    df = df.applymap(lambda x: 1 / x).rename(columns={'P/B': 'Book to Market',
                                                      'P/E': 'Earnings to Price'})
    # median absolute deviation
    MAD = abs(df - df.mean()).median()
    normalized_df = (df - df.median()) / MAD
    normalized_df = normalized_df[(normalized_df['Book to Market'] < 3) & (normalized_df['Book to Market'] > -3) & \
                                  (normalized_df['Earnings to Price'] < 3) & (normalized_df['Earnings to Price'] > -3)]
    normalized_df['sum'] = normalized_df.sum(axis=1)
    normalized_df = normalized_df.sort_values(by=['sum'], ascending=False)
    return normalized_df.index[:10].tolist()


c = pd.read_csv('./data/contrarian_stock_data.csv', index_col=0).dropna()
momentum = pd.read_csv('./data/momentum_stock_data.csv', index_col=0).dropna()
# possible stocks from both momentum and contrarian
mix = momentum.append(c)
mix = mix[~mix.index.duplicated(keep='first')]
# save the portfolio following all three strategies
contrarian_portfolio = stock_selection(c)
momentum_portfolio = stock_selection(momentum)
mix_portfolio = stock_selection(mix)

with open('./data/contrarian_portfolio.txt', 'w') as f:
    for item in contrarian_portfolio:
        f.write("%s\n" % item)

with open('./data/momentum_portfolio.txt', 'w') as f:
    for item in momentum_portfolio:
        f.write("%s\n" % item)

#######################################################
# Diversification of Potential Portfolios
# definition of normalized portfolio covariance of
# equal-weighted portfolio is proposed as a measurement
# for diversification in the work:
# Kumar, Alok and Goetzmann, William N., Equity Portfolio Diversification.
# Available at SSRN: https://ssrn.com/abstract=627321 or http://dx.doi.org/10.2139/ssrn.627321
#######################################################
# calculate normalized portfolio covariance
# less is better
# or directly load data
combined = pd.read_csv('./data/stock_pool_data.csv', index_col= 0)
combined['time'] = pd.Index(pd.to_datetime(combined.index))
combined = combined.set_index('time')

# calculate stock returns
data_raw = combined[['ticker', 'adjclose']]

data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
# flatten columns multi-index, `date` will become the dataframe index
data.columns = [col[1] for col in data.columns.values]
daily_stock_return = data.pct_change()


def normalized_portfolio_covariance(portfolio):
    daily = daily_stock_return[daily_stock_return.index > datetime(2021, 1, 31)]
    data = daily[portfolio]
    correlation_matrix = data.corr()
    avg_corr = sum(sum(np.triu(correlation_matrix.values)))/sum(range(len(portfolio)+1))
    return 1/len(portfolio) + avg_corr * (len(portfolio)-1)/len(portfolio)


print(normalized_portfolio_covariance(contrarian_portfolio))
print(normalized_portfolio_covariance(momentum_portfolio))
print(normalized_portfolio_covariance(mix_portfolio))




