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
from dateutil.relativedelta import relativedelta
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

#######################################################
# Stock Selection based on skewness
#######################################################
# Normalize all ta indicators using median & MAD


def skewness_and_kurtosis(stock_list):
    mix_tickers = stock_list

    c = pd.read_csv('/Users/huaerli/Desktop/2021 AYS1/DSA5205/group-project-local/code/data/stock_pool_data.csv', index_col=0).dropna()

    tickerList = mix_tickers
    multiData = c[c['ticker'].isin(mix_tickers)]
    df = multiData.copy()
    data_raw = df[['ticker', 'adjclose']]

    data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
    # flatten columns multi-index, `date` will become the dataframe index
    data.columns = [col[1] for col in data.columns.values]
    returns_portfolio = data.pct_change().fillna(method='bfill')

    skewness = {}
    kurtosis = {}

    for i in range(len(tickerList)):
        skewness[tickerList[i]] = stats.skew(returns_portfolio[tickerList[i]])
        kurtosis[tickerList[i]] = stats.kurtosis(returns_portfolio[tickerList[i]], fisher=False)

        skewness = {k: v for k, v in sorted(skewness.items(), key=lambda item: item[1], reverse=True)}
        kurtosis = {k: v for k, v in sorted(kurtosis.items(), key=lambda item: item[1])}
    return skewness, kurtosis


def filterTheDict(dictObj, callback):
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict


######################################################
# Stock Selection based on P/E and P/B
######################################################


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
    # p = 0.2
    # if mix:
    #     p = 0.1
    return normalized_df.index[:10].tolist()


# PE and PB data for 3 possible stock pools
c = pd.read_csv('./data/contrarian_stock_data.csv', index_col=0).dropna()
momentum = pd.read_csv('./data/momentum_stock_data.csv', index_col=0).dropna()
# possible stocks from both momentum and contrarian
mix = momentum.append(c)
mix = mix[~mix.index.duplicated(keep='first')]
# save the portfolio following all three strategies

# calculate kurtosis for stocks in each stock pool
contrarian_kurtosis = skewness_and_kurtosis(c.index.tolist())[1]
momentum_kurtosis = skewness_and_kurtosis(momentum.index.tolist())[1]
mix_kurtosis = skewness_and_kurtosis(mix.index.tolist())[1]

df_c = pd.DataFrame.from_dict(contrarian_kurtosis, orient='index', columns=['kurtosis'])
margin = 1.5*(df_c.describe().loc['75%'][0] - df_c.describe().loc['25%'][0]) + df_c.describe().loc['75%'][0]
stock_pool_c = filterTheDict(contrarian_kurtosis, lambda elem: elem[1] <= margin)
df_m = pd.DataFrame.from_dict(momentum_kurtosis, orient='index', columns=['kurtosis'])
margin = 1.5*(df_m.describe().loc['75%'][0] - df_m.describe().loc['25%'][0]) + df_m.describe().loc['75%'][0]
stock_pool_m = filterTheDict(momentum_kurtosis, lambda elem: elem[1] <= margin)
df_mix = pd.DataFrame.from_dict(mix_kurtosis, orient='index', columns=['kurtosis'])
margin = 1.5*(df_mix.describe().loc['75%'][0] - df_mix.describe().loc['25%'][0]) + df_mix.describe().loc['75%'][0]
stock_pool_mix = filterTheDict(mix_kurtosis, lambda elem: elem[1] <= margin)
fig, axs = plt.subplots(3, 2, figsize=(8, 6))
fig.suptitle('Kurtosis Distribution')

colors = ['#86bf91']
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))

df_c.hist(column='kurtosis', bins=25, ax=axs[0, 0], color='#86bf91')
# df_c.boxplot(column='kurtosis', ax=axs[0, 1], color='#86bf91')
sns.boxplot(y='kurtosis',
                 data=df_c,
                 width=0.5,
                ax= axs[0,1])
colors = ['coral']
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))
df_m.hist(column='kurtosis', bins=25, ax=axs[1, 0], color='coral')
# df_m.boxplot(column='kurtosis', ax=axs[1, 1], color='coral')
sns.boxplot(y='kurtosis',
                 data=df_m,
                 width=0.5, ax= axs[1,1])
colors = ['skyblue']
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))
df_mix.hist(column='kurtosis', bins=25, ax=axs[2, 0], color='skyblue')
sns.boxplot(y='kurtosis',
                 data=df_mix,
                 width=0.5, ax= axs[2,1])

legend_elements = [Line2D([0], [0], color='#86bf91', lw=2, label='Contrarian'),
                   Line2D([0], [0], color='coral', lw=2, label='Momentum'),
                   Line2D([0], [0], color='skyblue', lw=2, label='Mix')]



axs[2,0].legend(handles=legend_elements,loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)

# fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(1, 0.5))

plt.savefig('./data/kurtosis_distribution.jpg')
plt.show()
# filter out stocks with high kurtosis and then perform stock selection
contrarian_portfolio = stock_selection(c.loc[list(stock_pool_c.keys())])
momentum_portfolio = stock_selection(momentum.loc[list(stock_pool_m.keys())])
mix_portfolio = stock_selection(mix.loc[list(stock_pool_mix.keys())])

with open('./data/contrarian_portfolio.txt', 'w') as f:
    for item in contrarian_portfolio:
        f.write("%s\n" % item)

with open('./data/momentum_portfolio.txt', 'w') as f:
    for item in momentum_portfolio:
        f.write("%s\n" % item)

with open('./data/mixed_portfolio.txt', 'w') as f:
    for item in mix_portfolio:
        f.write("%s\n" % item)

# contrarian_kurtosis = skewness_and_kurtosis('./data/contrarian_portfolio.txt')[1]
# momentum_kurtosis = skewness_and_kurtosis('./data/momentum_portfolio.txt')[1]
# mix_kurtosis = skewness_and_kurtosis('./data/mixed_portfolio.txt')[1]
#
# contrarian_portfolio = list(contrarian_kurtosis.keys())[:10]
# momentum_portfolio = list(momentum_kurtosis.keys())[:10]
# mix_portfolio = list(mix_kurtosis.keys())[:10]
#
# with open('./data/contrarian_portfolio.txt', 'w') as f:
#     for item in contrarian_portfolio:
#         f.write("%s\n" % item)
#
# with open('./data/momentum_portfolio.txt', 'w') as f:
#     for item in momentum_portfolio:
#         f.write("%s\n" % item)
#
# with open('./data/mixed_portfolio.txt', 'w') as f:
#     for item in mix_portfolio:
#         f.write("%s\n" % item)


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
combined = pd.read_csv('/Users/huaerli/Desktop/2021 AYS1/DSA5205/group-project-local/code/data/stock_pool_data.csv', index_col=0)
combined['time'] = pd.Index(pd.to_datetime(combined.index))
combined = combined.set_index('time')

# calculate stock returns
data_raw = combined[['ticker', 'adjclose']]

data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
# flatten columns multi-index, `date` will become the dataframe index
data.columns = [col[1] for col in data.columns.values]
daily_stock_return = data.pct_change()


def normalized_portfolio_covariance(portfolio, period):
    month = relativedelta(months=period)
    start = datetime(2021, 7, 31) - month
    daily = daily_stock_return[daily_stock_return.index > start]
    data = daily[portfolio]
    correlation_matrix = data.corr()
    avg_corr = sum(sum(np.triu(correlation_matrix.values))) / sum(range(len(portfolio) + 1))
    return 1 / len(portfolio) + avg_corr * (len(portfolio) - 1) / len(portfolio)


normalized_cov_time = pd.DataFrame()
duration = [3, 6, 9, 12, 18]
for n in duration:
    temp = [normalized_portfolio_covariance(contrarian_portfolio, n),
            normalized_portfolio_covariance(momentum_portfolio, n),
            normalized_portfolio_covariance(mix_portfolio, n)]
    df = pd.DataFrame(temp, index=['contrarian', 'momentum', 'mix'], columns=['{} month'.format(n)])
    normalized_cov_time = normalized_cov_time.append(df.T, ignore_index=False)
print(normalized_cov_time)

plt.rcParams["figure.figsize"] = [10, 6]
normalized_cov_time.plot(kind="bar", rot=0, color=['lightsalmon', 'coral', 'peru'])
plt.title("Normalized Portfolio Covariance calculated across Various Time Scale")
plt.xlabel("Number of Months")
plt.ylabel("Portfolio Covariance")
plt.savefig('./data/portfolio_covariance.jpg')
plt.show()
