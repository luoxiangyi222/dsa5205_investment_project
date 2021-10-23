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


# ################# Historical Price Data #######################
# Open, High, Low, Volume, Dividends, Stock Splits


def get_daily_price_data(ticker_list):

    for i, t in enumerate(ticker_list):
        current_t_history = t.history(period='1d',
                                      start="2019-12-31",
                                      end="2021-10-01")

        # save as csv
        current_t_history.to_csv(f'../trading_strategy_data/selected_stocks_data/{SELECTED_STOCK_TICKS[i]}_price_data.csv')



# ################# Historical Technical Indicator Data #######################

def truncate_by_date(result_dict, ta_name, start='2019-12-31', end='2021-10-01'):
    start = date.fromisoformat(start)
    end = date.fromisoformat(end)
    result_dict = result_dict[f'Technical Analysis: {ta_name}']
    date_ta_dict = {date.fromisoformat(d): v for d, v in result_dict.items()
                    if start <= date.fromisoformat(d) <= end}
    date_ta_list = sorted(date_ta_dict.items(), key=lambda kv: (kv[0], kv[1]))
    return date_ta_list


def add_min_max_scaling(df, col_name):
    df[col_name + '_min_max'] = (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
    return df


def add_fluctuation_percentage(df, col_name):
    df[col_name + '_fluc_pctg'] = (df[col_name] - df[col_name].shift(periods=1, fill_value=0)) / df[col_name]
    return df


def add_polarize(df, col_name):
    df[col_name + '_polarize'] = 0
    df.loc[(df[col_name] > 0), col_name+'_polarize'] = 1
    df.loc[(df[col_name] < 0), col_name + '_polarize'] = -1
    return df


def get_TA_features_with_extension(stock_ticker, ta_function_list):
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
                      'SMA_10': [float(row[1]['SMA']) for row in sma_list]}

            ta_df = pd.DataFrame(sma_df)
            ta_df = add_min_max_scaling(ta_df, 'SMA_10')
            ta_df = add_fluctuation_percentage(ta_df, 'SMA_10')

        elif ta == 'MACD':
            url = f'https://www.alphavantage.co/query?function=MACD&symbol={stock_ticker}' \
                  f'&interval=daily&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            macd_list = truncate_by_date(data, 'MACD')

            macd_df = {'Date': [row[0] for row in macd_list]}

            macd_components = ['MACD', 'MACD_Signal', 'MACD_Hist']
            for c in macd_components:
                macd_df[c] = [float(row[1][c]) for row in macd_list]

            ta_df = pd.DataFrame(macd_df)
            for c in macd_components:
                ta_df = add_polarize(ta_df, c)

        elif ta == 'CCI':  # CCI 20
            url = f'https://www.alphavantage.co/query?function=CCI&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=20&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            cci_list = truncate_by_date(data, 'CCI')
            cci_df = {'Date': [row[0] for row in cci_list],
                      'CCI_20': [float(row[1]['CCI']) for row in cci_list]}
            ta_df = pd.DataFrame(cci_df)

            # polarize
            ta_df['CCI_20_polarize'] = 0
            ta_df.loc[(ta_df['CCI_20'] > 100), 'CCI_20_polarize'] = -1
            ta_df.loc[(ta_df['CCI_20'] < -100), 'CCI_20_polarize'] = 1

            # fluctuation percentage
            ta_df = add_fluctuation_percentage(ta_df, 'CCI_20')

        elif ta == 'ROC':  # ROC 10
            url = f'https://www.alphavantage.co/query?function=ROC&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=20&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            roc_list = truncate_by_date(data, 'ROC')
            roc_df = {'Date': [row[0] for row in roc_list],
                      'ROC_10': [float(row[1]['ROC']) for row in roc_list]}
            ta_df = pd.DataFrame(roc_df)

            ta_df = add_polarize(ta_df, 'ROC_10')
            ta_df = add_fluctuation_percentage(ta_df, 'ROC_10')

        elif ta == 'RSI':  # RSI 5
            url = f'https://www.alphavantage.co/query?function=RSI&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=5&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            rsi_list = truncate_by_date(data, 'RSI')
            rsi_df = {'Date': [row[0] for row in rsi_list],
                      'RSI_5': [float(row[1]['RSI']) for row in rsi_list]}
            ta_df = pd.DataFrame(rsi_df)

            # polarize
            ta_df['RSI_5_polarize'] = 0
            ta_df.loc[(ta_df['RSI_5'] > 70), 'RSI_5_polarize'] = -1
            ta_df.loc[(ta_df['RSI_5'] < 30), 'RSI_5_polarize'] = 1

            ta_df = add_fluctuation_percentage(ta_df, 'RSI_5')

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
                        'STOCH_SlowD': [float(row[1]['SlowD']) for row in stoch_list],
                        'STOCH_SlowK': [float(row[1]['SlowK']) for row in stoch_list]}
            ta_df = pd.DataFrame(stoch_df)

            ta_df = add_min_max_scaling(ta_df, 'STOCH_SlowD')
            ta_df = add_fluctuation_percentage(ta_df, 'STOCH_SlowD')
            ta_df = add_min_max_scaling(ta_df, 'STOCH_SlowK')
            ta_df = add_fluctuation_percentage(ta_df, 'STOCH_SlowK')

        elif ta == 'ADX':
            url = f'https://www.alphavantage.co/query?function=ADX&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=10&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()

            adx_list = truncate_by_date(data, 'ADX')
            adx_df = {'Date': [row[0] for row in adx_list],
                      'ADX': [float(row[1]['ADX']) for row in adx_list]}
            ta_df = pd.DataFrame(adx_df)

            ta_df = add_fluctuation_percentage(ta_df, 'ADX')

        elif ta == 'AROON':
            url = f'https://www.alphavantage.co/query?function=AROON&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=14&apikey={stock_ticker}'
            r = requests.get(url)
            data = r.json()

            aroon_list = truncate_by_date(data, 'AROON')
            aroon_df = {'Date': [row[0] for row in aroon_list],
                        'AROON_Up': [float(row[1]['Aroon Up']) for row in aroon_list],
                        'AROON_Down': [float(row[1]['Aroon Down']) for row in aroon_list]}
            ta_df = pd.DataFrame(aroon_df)

            ta_df = add_min_max_scaling(ta_df, 'AROON_Up')
            ta_df = add_min_max_scaling(ta_df, 'AROON_Down')

            # polarize
            ta_df['AROON_polarize'] = 0
            ta_df.loc[(ta_df['AROON_Up'] - ta_df['AROON_Down'] > 0), 'AROON_polarize'] = 1
            ta_df.loc[(ta_df['AROON_Up'] - ta_df['AROON_Down'] < 0), 'AROON_polarize'] = -1

        elif ta == 'BBANDS':
            url = f'https://www.alphavantage.co/query?function=BBANDS&symbol={stock_ticker}' \
                  f'&interval=daily&time_period=5&series_type=close&nbdevup=2&nbdevdn=2&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()

            bbands_list = truncate_by_date(data, 'BBANDS')
            bbands_df = {'Date': [row[0] for row in bbands_list],
                         'BBANDS_Lower_Band': [float(row[1]['Real Lower Band']) for row in bbands_list],
                         'BBANDS_Middle_Band': [float(row[1]['Real Middle Band']) for row in bbands_list],
                         'BBANDS_Upper_Band': [float(row[1]['Real Upper Band']) for row in bbands_list]}

            ta_df = pd.DataFrame(bbands_df)

            # band width
            ta_df['BBANDS_width'] = ta_df['BBANDS_Upper_Band'] - ta_df['BBANDS_Lower_Band']

        elif ta == 'AD':
            url = f'https://www.alphavantage.co/query?function=AD&symbol={stock_ticker}' \
                  f'&interval=daily&apikey={ALPHA_VANTAGE_API_KEY}'
            r = requests.get(url)
            data = r.json()
            ad_list = truncate_by_date(data, 'Chaikin A/D')
            ad_df = {'Date': [row[0] for row in ad_list],
                     'Chaikin_AD': [float(row[1]['Chaikin A/D']) for row in ad_list]}
            ta_df = pd.DataFrame(ad_df)

            ta_df = add_fluctuation_percentage(ta_df, 'Chaikin_AD')

        TA_df_list.append(ta_df)

        # Alpha Vantage limits 5 request per minutes
        time.sleep(15)

    # merge all technical indicators
    all_TA_df = reduce(lambda left, right: pd.merge(left, right, on='Date'), TA_df_list)
    ticker_column = pd.DataFrame({'Ticker': [stock_ticker] * all_TA_df.shape[0]})
    all_TA_df = pd.concat([ticker_column, all_TA_df], axis=1)

    all_TA_df.to_csv(f'../trading_strategy_data/selected_stocks_data/{stock_ticker}_TA_indicators.csv')

    print('ALL TA')
    print(all_TA_df.head())
    print(all_TA_df.tail())
    return all_TA_df


def prepare_data():
    get_stock_info(ticker_objects)

    # price data
    get_daily_price_data(ticker_objects)

    # TA data
    ta_list = ['SMA', 'MACD', 'CCI', 'ROC', 'RSI', 'STOCH', 'ADX', 'AROON', 'BBANDS', 'AD']
    get_TA_features_with_extension('aapl', ta_list)


# ##### Macro #####

# ################# Historical Investor Attention Data #######################
# get from google trend website

# ################# Historical Pandemic Data #######################
# get from CDC website
