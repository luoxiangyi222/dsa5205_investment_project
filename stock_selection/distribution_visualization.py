import warnings
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

mix_tickers = []
with open('./data/mixed_portfolio.txt') as file:
    for line in file:
        mix_tickers.append(line.rstrip())

c = pd.read_csv('./data/stock_pool_data.csv', index_col=0).dropna()


date_from = datetime.date(2020, 1, 1)
date_to = datetime.date(2021, 7, 31)
tickerList = mix_tickers
multiData = c[c['ticker'].isin(mix_tickers)]
df = multiData.copy()
data_raw = multiData[['ticker', 'adjclose']]


data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
# flatten columns multi-index, `date` will become the dataframe index
data.columns = [col[1] for col in data.columns.values]
print(data)
weights = [0.1]*10

returns_portfolio = data.pct_change().fillna(method='bfill')
returns_portfolio['portfolio'] = (returns_portfolio.mul(weights, axis=1)).sum(axis=1)
fig2, axs2 = plt.subplots(3,4, figsize=(16, 12), facecolor='w', edgecolor='k')
# fig.set_title("Probplot")
fig2.subplots_adjust(hspace =.5, wspace=.001)
axs2 = axs2.ravel()

for i in range(11):
    returns_portfolio[returns_portfolio.columns[i]].hist(bins=50, ax=axs2[i])
    axs2[i].set_xlabel('Return')
    axs2[i].set_ylabel('Frequency')
    axs2[i].set_title(returns_portfolio.columns[i])
plt.show()





import scipy.stats as stats


# fig, axs = plt.subplots(5,2, figsize=(15, 25), facecolor='w', edgecolor='k')
# # fig.set_title("Probplot")
# fig.subplots_adjust(hspace =.5, wspace=.001)
#
# axs = axs.ravel()
#
# for i in range(11):
#     stats.probplot(returns_portfolio[tickerList[i]], dist='norm', plot=axs[i])
#     axs[i].set_title(tickerList[i]+" has kurtosis of {}".format(stats.kurtosis(returns_portfolio[tickerList[i]])))
#     print("skewness of {} is {}".format(tickerList[i], stats.skew(returns_portfolio[tickerList[i]])))
# plt.show()

# for i in tickerList:
#     temp = df[df['ticker'] == i]
#     # print(temp)
#     temp['Return'] = temp['adjclose'].pct_change().fillna(method='bfill')
#     print(temp)
#
#     fig = plt.figure(figsize=(15, 7))
#     ax1 = fig.add_subplot(1, 1, 1)
#     temp['Return'].hist(bins=50, ax=ax1)
#     ax1.set_xlabel('Return')
#     ax1.set_ylabel('Frequency')
#     ax1.set_title('Return distribution')
#     plt.show()