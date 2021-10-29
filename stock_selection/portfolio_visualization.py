import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from datetime import datetime
import requests
from yahoo_fin import stock_info as si
import time
from functools import reduce
import plotly.graph_objects as go
import random

##########################################################
# NASDAQ index & Portfolio Data
##########################################################
ndq = pd.read_csv('./data/NASDAQ.csv', index_col=0).dropna()
ndq['time'] = pd.Index(pd.to_datetime(ndq.index))
ndq = ndq.set_index('time')
ndq = ndq[ndq.index > datetime(2021, 7, 31)]

# load portfolio
momentum_tickers = []
with open('./data/momentum_portfolio.txt') as file:
    for line in file:
        momentum_tickers.append(line.rstrip())

contrarian_tickers = []
with open('./data/contrarian_portfolio.txt') as file:
    for line in file:
        contrarian_tickers.append(line.rstrip())

stock_pool = []
with open('./data/stock_pool.txt') as file:
    for line in file:
        stock_pool.append(line.rstrip())

# assigning weights
# equal weights
w = [0.1] * 10
ndq_cum = ndq['Adj Close'] / ndq.iloc[0]['Adj Close']
init_investment = 10000
ndq_total = init_investment * ndq_cum
ndq_cum = (ndq_cum - 1)*100


def cum_return_calculation(tickers, weights):
    price_data = {ticker: si.get_data(ticker, start_date="2021-07-31") for ticker in tickers}
    combined = reduce(lambda x, y: x.append(y), price_data.values())
    # calculate stock returns
    data_raw = combined[['ticker', 'adjclose']]

    data = data_raw.pivot_table(index=data_raw.index, columns='ticker', values=['adjclose'])
    # flatten columns multi-index, `date` will become the dataframe index
    data.columns = [col[1] for col in data.columns.values]

    daily_stock_return = data.pct_change()
    cum_returns = (daily_stock_return + 1).cumprod()
    cum_returns = cum_returns.iloc[1:]
    cum_return = cum_returns[tickers]

    data_por = init_investment * cum_return.mul(weights, axis=1)

    sum = data_por.sum(axis=1)

    cum_return = (sum / sum.iloc[0] - 1)*100
    return cum_return, sum


def random_10_stocks(k):
    cum_sum = 0
    total_sum = 0
    for i in range(k):
        ind = random.sample(range(len(stock_pool)), 10)
        portfolio = [stock_pool[i] for i in ind]
        w = [0.1] * 10
        cum, total = cum_return_calculation(portfolio, w)
        cum_sum += cum
        total_sum += total
    avg_cum = cum_sum/k
    avg_total = total_sum/k
    return avg_cum, avg_total

a, b = random_10_stocks(10)
wc = [0,0.1203, 0, 0.4536, 0, 0.2662, 0.1599, 0,0,0]
wd = [0.0042, 0, 0, 0.0961, 0, 0.2427, 0, 0, 0.3778, 0.2792]
cum_return, y = cum_return_calculation(momentum_tickers, w)
wcc, y = cum_return_calculation(contrarian_tickers, wc)
wdc, y = cum_return_calculation(contrarian_tickers, wd)
c, d = cum_return_calculation(contrarian_tickers, w)
fig = go.Figure()
fig.add_trace(go.Scatter(x=cum_return.index, y=cum_return, name='Momentum Portfolio Cumulative Return %'))
fig.add_trace(go.Scatter(x=c.index, y=c, name='Contrarian Portfolio Cumulative Return %'))
# fig.add_trace(go.Scatter(x=wcc.index, y=wcc, name='Max Shape Ratio Weighted Contrarian Portfolio Cumulative Return %'))
# fig.add_trace(go.Scatter(x=wdc.index, y=wdc, name='Min Volatility Weighted Contrarian Portfolio Cumulative Return %'))
fig.add_trace(go.Scatter(x=a.index, y=a, name='Cumulative Return % averaged over 10 randomly selected portfolio'))
# fig.add_trace(go.Scatter(x=ndq_cum.index, y=ndq_cum, name='NASDAQ Index Cumulative Return %'))
fig.update_layout(title="Cumulative Return % (Portfolio vs NASDAQ Index)")
fig.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=cum_return.index, y=cum_return, name='Momentum Portfolio Cumulative Return %'))
fig2.add_trace(go.Scatter(x=c.index, y=c, name='Contrarian Portfolio Cumulative Return %'))
fig2.add_trace(go.Scatter(x=ndq_cum.index, y=ndq_cum, name='NASDAQ Index Cumulative Return %'))
fig2.show()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=c.index, y=c, name='Contrarian Portfolio Cumulative Return %'))
fig3.add_trace(go.Scatter(x=wcc.index, y=wcc, name='Max Shape Ratio Weighted Contrarian Portfolio Cumulative Return %'))
fig3.add_trace(go.Scatter(x=wdc.index, y=wdc, name='Min Volatility Weighted Contrarian Portfolio Cumulative Return %'))
fig3.show()

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=cum_return.index, y=cum_return, name='Momentum Portfolio Cumulative Return %'))
fig4.add_trace(go.Scatter(x=c.index, y=c, name='Contrarian Portfolio Cumulative Return %'))
fig4.add_trace(go.Scatter(x=wcc.index, y=wcc, name='Max Shape Ratio Weighted Contrarian Portfolio Cumulative Return %'))
fig4.add_trace(go.Scatter(x=wdc.index, y=wdc, name='Min Volatility Weighted Contrarian Portfolio Cumulative Return %'))
fig4.add_trace(go.Scatter(x=a.index, y=a, name='Cumulative Return % averaged over 10 randomly selected portfolio'))
# fig.add_trace(go.Scatter(x=ndq_cum.index, y=ndq_cum, name='NASDAQ Index Cumulative Return %'))
fig4.update_layout(title="Cumulative Return % (Portfolio vs NASDAQ Index)")
fig4.show()