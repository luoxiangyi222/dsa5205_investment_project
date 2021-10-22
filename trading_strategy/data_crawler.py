import yfinance as yf
import yahoo_fin.stock_info as si
import pandas as pd
import requests
from datetime import date
from functools import reduce

import json

pd.set_option('display.max_columns', None)  # enable to see all columns
# pd.set_option('display.max_rows', None)  # enable to see all columns

# ################# API keys #############################
# Get Technical Analysis indicators
# API documentation: https://www.alphavantage.co/documentation/
ALPHA_VANTAGE_API_KEY = '6OFO07W5FJ2R943U'

# API documentation: https://polygon.io/docs/getting-started
POLYGON_API_KEY = 'xKi2aeZYxFcpbe8xJXxwHlV7cj50AU6X'

SELECTED_STOCK_TICKS = ['aapl', 'cost']

ticker_objects = [yf.Ticker(s) for s in SELECTED_STOCK_TICKS]


# ##### Micro #####


# def get_stock_info(ticker_list):
#     for i, t in enumerate(ticker_list):
#         with open(f'../stocks_data/{SELECTED_STOCK_TICKS[i]}_info.json', 'w') as fp:
#             json.dump(t.info, fp)
#
#
# get_stock_info(ticker_objects)

# ################# Historical Price Data #######################
# Open, High, Low, Volume, Dividends, Stock Splits


# def get_daily_price_data(ticker_list):
#
#     for i, t in enumerate(ticker_list):
#         current_t_history = t.history(period='1d',
#                                       start="2019-12-31",
#                                       end="2021-10-01")
#
#         # save as csv
#         current_t_history.to_csv(f'../stocks_data/{SELECTED_STOCK_TICKS[i]}_price_data.csv')
#
#
# get_daily_price_data(ticker_objects)


# ################# Historical Technical Indicator Data #######################

def truncate_by_date(result_dict, ta_name, start='2019-12-31', end='2021-10-01'):
    start = date.fromisoformat(start)
    end = date.fromisoformat(end)
    result_dict = result_dict[f'Technical Analysis: {ta_name}']
    date_ta_dict = {date.fromisoformat(d): v for d, v in result_dict.items()
                    if start <= date.fromisoformat(d) <= end}
    date_ta_list = sorted(date_ta_dict.items(), key=lambda kv: (kv[0], kv[1]))
    return date_ta_list


def get_TA_features(stock_ticker, ta_function_list):
    ta_df_list = []

    for ta in ta_function_list:
        if ta == 'SMA':  # SMA 10
            url = f'https://www.alphavantage.co/query?function=SMA&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=10&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            sma_list = truncate_by_date(data, 'SMA')
            print(sma_list)
            sma_df = {'Date': [row[0] for row in sma_list],
                      'SMA_10': [row[1]['SMA'] for row in sma_list]}
            sma_df = pd.DataFrame(sma_df)
            print(sma_df)

        elif ta == 'MACD':
            url = f'https://www.alphavantage.co/query?function=MACD&symbol={stock_ticker}' \
                  f'&interval=daily&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            macd_list = truncate_by_date(data, 'MACD')

            macd_df = {'Date': [row[0] for row in macd_list]}

            macd_components = ['MACD', 'MACD_Signal', 'MACD_Hist']
            for c in macd_components:
                macd_df[c] = [row[1][c] for row in macd_list]

            macd_df = pd.DataFrame(macd_df)


        elif ta == 'CCI':  # CCI 24
            url = f'https://www.alphavantage.co/query?function=CCI&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=24&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()

        elif ta == 'ROC':  # ROC 10
            url = f'https://www.alphavantage.co/query?function=ROC&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=10&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
        elif ta == 'RSI':  # RSI 5
            url = f'https://www.alphavantage.co/query?function=RSI&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=5&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
        elif ta == 'ADOSC':
            url = f'https://www.alphavantage.co/query?function=ADOSC&symbol={stock_ticker}' \
                  f'&interval=daily&fastperiod=5&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
        elif ta == 'STOCH':
            url = f'https://www.alphavantage.co/query?function=STOCH&symbol={stock_ticker}' \
                  f'&interval=daily&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
        elif ta == 'ADX':
            url = f'https://www.alphavantage.co/query?function=ADX&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=10&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
        elif ta == 'AROON':
            url = f'https://www.alphavantage.co/query?function=AROON&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=14&apikey={stock_ticker}'
            r = requests.get(url)
            data = r.json()
        elif ta == 'BBANDS':
            url = f'https://www.alphavantage.co/query?function=BBANDS&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=5&series_type=close&nbdevup=2&nbdevdn=2&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
        elif ta == 'AD':
            url = f'https://www.alphavantage.co/query?function=AD&symbol={stock_ticker}' \
                  f'&interval=daily&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()

        # ta_df_list.append(ta_df)

    # merge all technical indicators
    # df = reduce(lambda left, right: pd.merge(left, right, on='Date'), ta_df_list)
    # print(df.head())


ta_list = ['SMA']
get_TA_features('COST', ta_list)

# for s in SELECTED_STOCK_TICKS:
#
#
#     # SMAS = [v['SMA'] for v in result.values()]
#     # print(dates)
#     print(SMAS)


# ##### Macro #####

# ################# Historical Investor Attention Data #######################
# TODO: what can we use to measure investor attention? Google volume index? twitter trend? choose one is enough
#

# ################# Historical Economic Data #######################
# TODO: what economic measures will influence the stock market? GDP? Treasury Yeild? Inflation? Retail Sale? ...


# ################# Historical Pandemic Data #######################
# TODO: Since we focus on Nasdaq, determine: global data or local(the US) data?
# TODO daily confirmed cases? vaccine rate? daily death?
