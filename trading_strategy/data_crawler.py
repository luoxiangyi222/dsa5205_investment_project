import yfinance as yf

import pandas as pd
import requests
from datetime import date
from functools import reduce

import time
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


def get_stock_info(ticker_list):
    for i, t in enumerate(ticker_list):
        with open(f'../trading_strategy_data/selected_stocks_data/{SELECTED_STOCK_TICKS[i]}_info.json', 'w') as fp:
            json.dump(t.info, fp)


get_stock_info(ticker_objects)

# ################# Historical Price Data #######################
# Open, High, Low, Volume, Dividends, Stock Splits


def get_daily_price_data(ticker_list):

    for i, t in enumerate(ticker_list):
        current_t_history = t.history(period='1d',
                                      start="2019-12-31",
                                      end="2021-10-01")

        # save as csv
        current_t_history.to_csv(f'../trading_strategy_data/selected_stocks_data/{SELECTED_STOCK_TICKS[i]}_price_data.csv')


get_daily_price_data(ticker_objects)

breakpoint()


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
    TA_df_list = []

    for ta in ta_function_list:
        ta_df = None
        if ta == 'SMA':  # SMA 10
            url = f'https://www.alphavantage.co/query?function=SMA&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=10&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            sma_list = truncate_by_date(data, 'SMA')
            sma_df = {'Date': [row[0] for row in sma_list],
                      'SMA_10': [row[1]['SMA'] for row in sma_list]}
            ta_df = pd.DataFrame(sma_df)

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

            ta_df = pd.DataFrame(macd_df)

        elif ta == 'CCI':  # CCI 20
            url = f'https://www.alphavantage.co/query?function=CCI&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=20&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            cci_list = truncate_by_date(data, 'CCI')
            cci_df = {'Date': [row[0] for row in cci_list],
                      'CCI_20': [row[1]['CCI'] for row in cci_list]}
            ta_df = pd.DataFrame(cci_df)

        elif ta == 'ROC':  # ROC 10
            url = f'https://www.alphavantage.co/query?function=ROC&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=20&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            roc_list = truncate_by_date(data, 'ROC')
            roc_df = {'Date': [row[0] for row in roc_list],
                      'ROC_10': [row[1]['ROC'] for row in roc_list]}
            ta_df = pd.DataFrame(roc_df)

        elif ta == 'RSI':  # RSI 5
            url = f'https://www.alphavantage.co/query?function=RSI&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=5&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            rsi_list = truncate_by_date(data, 'RSI')
            rsi_df = {'Date': [row[0] for row in rsi_list],
                      'RSI_5': [row[1]['RSI'] for row in rsi_list]}
            ta_df = pd.DataFrame(rsi_df)

        elif ta == 'ADOSC':
            url = f'https://www.alphavantage.co/query?function=ADOSC&symbol={stock_ticker}' \
                  f'&interval=daily&fastperiod=5&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            adosc_list = truncate_by_date(data, 'ADOSC')
            adosc_df = {'Date': [row[0] for row in adosc_list],
                        'ADOSC': [row[1]['ADOSC'] for row in adosc_list]}
            ta_df = pd.DataFrame(adosc_df)

        elif ta == 'STOCH':
            url = f'https://www.alphavantage.co/query?function=STOCH&symbol={stock_ticker}' \
                  f'&interval=daily&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()

            stoch_list = truncate_by_date(data, 'STOCH')
            stoch_df = {'Date': [row[0] for row in stoch_list],
                        'STOCH_SlowD': [row[1]['SlowD'] for row in stoch_list],
                        'STOCH_SlowK': [row[1]['SlowK'] for row in stoch_list]}
            ta_df = pd.DataFrame(stoch_df)

        elif ta == 'ADX':
            url = f'https://www.alphavantage.co/query?function=ADX&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=10&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()

            adx_list = truncate_by_date(data, 'ADX')
            adx_df = {'Date': [row[0] for row in adx_list],
                      'ADX': [row[1]['ADX'] for row in adx_list]}
            ta_df = pd.DataFrame(adx_df)

        elif ta == 'AROON':
            url = f'https://www.alphavantage.co/query?function=AROON&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=14&apikey={stock_ticker}'
            r = requests.get(url)
            data = r.json()

            aroon_list = truncate_by_date(data, 'AROON')
            aroon_df = {'Date': [row[0] for row in aroon_list],
                        'AROON_Up': [row[1]['Aroon Up'] for row in aroon_list],
                        'AROON_Down': [row[1]['Aroon Down'] for row in aroon_list]}
            ta_df = pd.DataFrame(aroon_df)

        elif ta == 'BBANDS':
            url = f'https://www.alphavantage.co/query?function=BBANDS&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=5&series_type=close&nbdevup=2&nbdevdn=2&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()

            bbands_list = truncate_by_date(data, 'BBANDS')
            bbands_df = {'Date': [row[0] for row in bbands_list],
                         'BBANDS_Lower_Band': [row[1]['Real Lower Band'] for row in bbands_list],
                         'BBANDS_Middle_Band': [row[1]['Real Middle Band'] for row in bbands_list],
                         'BBANDS_Upper_Band': [row[1]['Real Upper Band'] for row in bbands_list]}

            ta_df = pd.DataFrame(bbands_df)

        elif ta == 'AD':
            url = f'https://www.alphavantage.co/query?function=AD&symbol={stock_ticker}' \
                  f'&interval=daily&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            ad_list = truncate_by_date(data, 'Chaikin A/D')
            ad_df = {'Date': [row[0] for row in ad_list],
                     'Chaikin A/D': [row[1]['Chaikin A/D'] for row in ad_list]}
            ta_df = pd.DataFrame(ad_df)

        TA_df_list.append(ta_df)

        # Alpha Vantage limits 5 request per minutes
        time.sleep(20)

    # merge all technical indicators
    all_TA_df = reduce(lambda left, right: pd.merge(left, right, on='Date'), TA_df_list)
    ticker_column = pd.DataFrame({'Ticker': [stock_ticker] * all_TA_df.shape[0]})
    all_TA_df = pd.concat([ticker_column, all_TA_df], axis=1)

    all_TA_df.to_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_TA_indicators.csv')

    print('ALL TA')
    print(all_TA_df.head())
    print(all_TA_df.tail())
    return all_TA_df


ta_list = ['SMA', 'MACD', 'CCI', 'ROC', 'RSI', 'STOCH', 'ADX', 'AROON', 'BBANDS', 'AD']

get_TA_features('aapl', ta_list)


# ##### Macro #####

# ################# Historical Investor Attention Data #######################
# get from google trend website

# ################# Historical Pandemic Data #######################
# get from CDC website
