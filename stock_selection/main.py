# This is a sample Python script.

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
# API documentation: https://polygon.io/docs/getting-started

tickers = si.tickers_nasdaq()
POLYGON_API_KEY = 'NrBIieL09DapVNqM0syN8k8_kYUTvVDq'
nasdaq_tickers = []
miss = 0
for t in tickers:
    print(t)
    url = f'https://api.polygon.io/v1/meta/symbols/{t}/company?apiKey={POLYGON_API_KEY}'
    # url = f'https://api.polygon.io/v1/meta/symbols/AACG/company?apiKey=NrBIieL09DapVNqM0syN8k8_kYUTvVDq'
    r = requests.get(url)
    data = r.json()
    print(data)
    if "error" in data:
        miss += 1
        time.sleep(12)
        continue
    else:
        try:
            date = datetime.strptime(data['listdate'], '%Y-%m-%d').date()
        except TypeError:
            continue
        start_date = datetime(2020, 1, 1).date()
        if date < start_date:
            nasdaq_tickers.append(t)
        time.sleep(12)
with open('./data/stock_list.txt', 'w') as f:
    for item in nasdaq_tickers:
        f.write("%s\n" % item)
# price_data = {ticker: si.get_data(ticker) for ticker in tickers}
# combined = reduce(lambda x, y: x.append(y), price_data.values())
# print(combined)

# get data from yahoo finance
# from 2020-1-1 to 2021-7-31
# nasdaq_stocks = web.get_data_yahoo(tickers, start="2020-01-01", end="2021-07-31")
# nasdaq_stock_monthly_returns = nasdaq_stocks['Adj Close'].resample('M').ffill().pct_change()
# temp = nasdaq_stock_monthly_returns[0:5]
# fig = plt.figure()
# (temp + 1).cumprod().plot()
# plt.show()