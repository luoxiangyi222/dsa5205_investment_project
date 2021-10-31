import warnings
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

import scipy.stats as stats

mix_tickers = []
with open('./data/mixed_portfolio.txt') as file:
    for line in file:
        mix_tickers.append(line.rstrip())

c = pd.read_csv('/Users/huaerli/Desktop/2021 AYS1/DSA5205/group-project-local/code/data/stock_pool_data.csv', index_col=0).dropna()


date_from = datetime.date(2020, 1, 1)
date_to = datetime.date(2021, 7, 31)
tickerList = mix_tickers
multiData = c[c['ticker'].isin(mix_tickers)]
df = multiData.copy()
data_raw = multiData[['ticker', 'adjclose']]


data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
# flatten columns multi-index, `date` will become the dataframe index
data.columns = [col[1] for col in data.columns.values]
weights = [0.1]*10

returns_portfolio = data.pct_change().fillna(method='bfill')
returns_portfolio['portfolio'] = (returns_portfolio.mul(weights, axis=1)).sum(axis=1)
fig2, axs2 = plt.subplots(3,4, figsize=(16, 12), facecolor='w', edgecolor='k', sharex=True, sharey=True)
fig2.suptitle("Probability Distribution of Stocks Daily Returns")
fig2.subplots_adjust(hspace =.5, wspace=.001)
axs2 = axs2.ravel()

for i in range(11):
    returns_portfolio[returns_portfolio.columns[i]].hist(bins=50, ax=axs2[i], color='#86bf91')
    axs2[i].text(0.08, 95,
                "skewness: {}".format(round(stats.skew(returns_portfolio[returns_portfolio.columns[i]]),4)),
                style='italic', fontsize=6,
                bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    axs2[i].set_xlabel('Return')
    axs2[i].set_ylabel('Frequency')
    axs2[i].set_title(returns_portfolio.columns[i])
axs2[11].axis('off')
plt.savefig('./data/prob_dist_returns.jpg')
plt.show()





fig, axs = plt.subplots(3,4, figsize=(16, 12), facecolor='w', edgecolor='k', sharex=True, sharey=True)
fig.suptitle("QQ Plot for Stocks Daily Returns")
fig.subplots_adjust(hspace =.5, wspace=.001)

axs = axs.ravel()

for i in range(11):
    stats.probplot(returns_portfolio[returns_portfolio.columns[i]], dist='norm', plot=axs[i])
    axs[i].get_lines()[0].set_markerfacecolor('#86bf91')
    axs[i].get_lines()[0].set_markeredgecolor('#86bf91')
    axs[i].get_lines()[1].set_linewidth(2.0)
    axs[i].get_lines()[1].set_color('grey')
    axs[i].text(-0.2, 0.2, "kurtosis of {}".format(round(stats.kurtosis(returns_portfolio[returns_portfolio.columns[i]]),4)),
                                                             style='italic', fontsize=6,
            bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    axs[i].set_title(returns_portfolio.columns[i])
axs[11].axis('off')
plt.savefig('./data/qq_plot_returns.jpg')
plt.show()
