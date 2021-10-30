"""
Author: Luo Xiangyi
Create on: 2021-10-23

This module crawls data from different sources and save them locally.
"""


from datetime import date
from functools import reduce
import json
import pandas as pd
from pytrends import dailydata
import requests
import time
import yfinance as yf
import random


# pandas setting
pd.set_option('display.max_columns', None)  # enable to see all columns
pd.set_option('display.max_rows', None)  # enable to see all rows

START_DATE_STR = '2019-12-01'
END_DATE_STR = '2021-10-15'

START_DATE = date.fromisoformat(START_DATE_STR)
END_DATE = date.fromisoformat(END_DATE_STR)


# ################# API keys #############################
# Get Technical Analysis indicators
# API documentation: https://www.alphavantage.co/documentation/
# ALPHA_VANTAGE_API_KEY = '6OFO07W5FJ2R943U'
ALPHA_VANTAGE_API_KEY ='PIY8WPDP3QZSJZXR'

# API documentation: https://polygon.io/docs/getting-started
POLYGON_API_KEY = 'xKi2aeZYxFcpbe8xJXxwHlV7cj50AU6X'


# momentum based
MOMENTUM_PORTFOLIO = ['PLUS', 'BKSC', 'ARTNA', 'HWKN', 'PDCO', 'LNT', 'XEL', 'PINC', 'CMCSA', 'JBSS']
momentum_portfolio_kw_list = ['ePlus', 'Bank of South Carolina', 'Artesian Resources', 'Hawkins', 'Patterson Companies',
                              'Alliant Energy', 'Xcel Energy', 'Premier', 'Comcast', 'John B. Sanfilippo & Son']

# contrarian based
CONTRARIAN_PORTFOLIO = ['PLPC', 'RILY', 'ULH', 'FSBC', 'WABC',
                        'WVVI', 'LGIH', 'WIRE', 'SLGN', 'ARTNA']
contrarian_portfolio_kw_list = ['Preformed Line Products', 'B Riley Financial', 'Universal Logistics Holdings',
                                'Five Star Bancorp', 'Westamerica',
                                'Willamette Valley Vineyards', 'LGI Homes', 'Encore Wire', 'Silgan Holdings',
                                'Artesian Resources']

MIX_PORTFOLIO = ['SAGE', 'FSFG', 'RNDB', 'OPRA', 'LARK', 'CFFI', 'UBOH', 'ANTE', 'HMNF', 'MRBK']
mix_portfolio_kw_list = ['SAGE Therapeutics', 'First Savings Financial', 'RNDB', 'Opera Limited', 'stock LARK',
                         'C&F Financial','United Bancshares', 'Airnet Technology','HMNF', 'Meridian']


# Randomly selected stocks for RFE
RANDOM_STOCKS = ['AAPL', 'COST', 'UTHR', 'LBTYK', 'SBUX', 'TXN', 'ERIC', 'DGRW', 'FMB', 'THRM',
                 'OTEX', 'IBB', 'IUSG', 'AVGO', 'HURC', 'ASML', 'MEDP', 'GRMN', 'PIE', 'NFLX',
                 'TACO', 'HOFT', 'ULCC', 'BABA', 'PDBC', 'HUBG', 'JBSS', 'FROG', 'DOYU', 'SRCL']
random_kw_list = ['Apple', 'Costco', 'United Therapeutics', 'Liberty Global', 'Starbucks', 'Texas Instruments', 'Ericsson', 'WisdomTree', 'FMB', 'Gentherm',
                  'OpenText', 'iShares Biotechnology', 'IUSG', 'Broadcom', 'Hurco', 'asml', 'Medpace', 'Garmin', 'PIE', 'Netflix',
                  'Del Taco Restaurants', 'Hooker Furniture', 'Frontier Group Holdings', 'Alibaba', 'PDBC', 'Hub Group', 'John B. Sanfilippo & Son', 'JFrog', 'DouYu', 'Stericycle']


# RANDOM_STOCKS = ['TACO', 'HOFT', 'ULCC', 'BABA', 'PDBC', 'HUBG', 'JBSS', 'FROG', 'DOYU', 'SRCL']
# random_kw_list = ['Del Taco Restaurants', 'Hooker Furniture', 'Frontier Group Holdings', 'Alibaba', 'PDBC', 'Hub Group', 'John B. Sanfilippo & Son', 'JFrog', 'DouYu', 'Stericycle']


# prepare objects lit
momentum_portfolio_ticker_objects = [yf.Ticker(s) for s in MOMENTUM_PORTFOLIO]
contrarian_portfolio_ticker_objects = [yf.Ticker(s) for s in CONTRARIAN_PORTFOLIO]
mix_portfolio_ticker_objects = [yf.Ticker(s) for s in MIX_PORTFOLIO]

random_stocks_ticker_objects = [yf.Ticker(s) for s in RANDOM_STOCKS]


# ##### company basic information #####


def get_stock_info(stocks, ticker_list, folder_name):
    for i, t in enumerate(ticker_list):
        with open(f'../trading_strategy_data/{folder_name}/{stocks[i]}_info.json', 'w') as fp:
            json.dump(t.info, fp)
            print(f'========== {stocks[i]} company info saved! ==========')


# ################# Historical Price Data #######################
# Open, High, Low, Volume, Dividends, Stock Splits


def get_daily_price_data(stocks, ticker_list, folder_name):

    for i, t in enumerate(ticker_list):
        current_t_history = t.history(period='1d',
                                      start=START_DATE_STR,
                                      end=END_DATE_STR)

        # save as csv
        current_t_history.to_csv(f'../trading_strategy_data/{folder_name}/'
                                 f'{stocks[i]}_daily_price_data.csv', index=False)
        print(f'========== {stocks[i]} daily price saved! ==========')


# ################# Historical Technical Indicator Data #######################


def truncate_by_date(result_dict, ta_name, start=START_DATE, end=END_DATE):
    """
    Only keep the required timeframe.
    """

    result_dict = result_dict[f'Technical Analysis: {ta_name}']
    date_ta_dict = {date.fromisoformat(d): v for d, v in result_dict.items()
                    if start <= date.fromisoformat(d) <= end}

    # sort by Date, ascending order
    date_ta_list = sorted(date_ta_dict.items(), key=lambda kv: (kv[0], kv[1]))
    return date_ta_list


def add_min_max_scaling(df, col_name):
    df[col_name + '_min_max'] = (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
    return df


def add_fluctuation_percentage(df, col_name):
    df[col_name + '_fluc_pctg'] = (df[col_name] - df[col_name].shift(periods=1)) / df[col_name].shift(periods=1)
    return df


def add_polarize(df, col_name):
    """
    Only available for the simplest polarization.
    """
    df[col_name + '_polarize'] = 0
    df.loc[(df[col_name] > 0), col_name+'_polarize'] = 1
    df.loc[(df[col_name] < 0), col_name + '_polarize'] = -1
    return df


def get_TA_features_with_extension(stock_ticker, ta_function_list, folder_name):
    """
    Get all TA indicators and proper extension of each indicator.
    Extensions are inspired by existing paper and investpedia.
    """
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

    # add Ticker
    ticker_column = pd.DataFrame({'Ticker': [stock_ticker] * all_TA_df.shape[0]})
    all_TA_df = pd.concat([ticker_column, all_TA_df], axis=1)

    # Save TA dataframe
    all_TA_df.to_csv(f'../trading_strategy_data/{folder_name}/{stock_ticker}_TA_indicators.csv', index=False)

    print(f'========== {stock_ticker} TA indicators saved. ==========')
    return all_TA_df


def get_google_trend_data(ticker, keyword, folder_name, start=START_DATE, end=END_DATE):

    daily_df = dailydata.get_daily_data(word=keyword,
                                        start_year=start.year, start_mon=start.month,
                                        stop_year=end.year, stop_mon=end.month,
                                        geo=''
                                        )

    daily_df = daily_df.reset_index()
    daily_df = daily_df.iloc[:, 0:2]
    daily_df.columns = ['Date', 'daily_trend']

    daily_df.to_csv(f'../trading_strategy_data/{folder_name}/{ticker}_daily_trend.csv', index=False)
    print(f'========== {ticker} daily google search volume index saved! ==========')
    return daily_df


def get_data(stocks, yf_objects, folder):

    # basic company information
    get_stock_info(stocks, yf_objects, folder_name=folder)

    # price data
    get_daily_price_data(stocks, yf_objects, folder_name=folder)

    # TA data
    ta_list = ['SMA', 'MACD', 'CCI', 'ROC', 'RSI', 'STOCH', 'ADX', 'AROON', 'BBANDS', 'AD']
    for ticker in stocks:
        get_TA_features_with_extension(ticker, ta_list, folder_name=folder)

    print('=====++++++++++ Portfolio Data Crawling finished ++++++++++======')


def get_google_portfolio(select_strategy):

    # Google daily trend volume index data
    if select_strategy == 'momentum':
        for i in range(10):
            get_google_trend_data(MOMENTUM_PORTFOLIO[i], momentum_portfolio_kw_list[i],
                                  folder_name='portfolio_data/momentum')

    elif select_strategy == 'contrarian':
        for i in range(10):
            get_google_trend_data(CONTRARIAN_PORTFOLIO[i], contrarian_portfolio_kw_list[i],
                                  folder_name='portfolio_data/contrarian')

    elif select_strategy == 'mix':
        for i in range(10):
            get_google_trend_data(MIX_PORTFOLIO[i], mix_portfolio_kw_list[i],
                                  folder_name='portfolio_data/mix')

    return True


def get_google_random_stocks():
    # Google daily trend volume index data
    for i in range(30):
        get_google_trend_data(RANDOM_STOCKS[i], random_kw_list[i], folder_name='random_stocks_data')


if __name__ == "__main__":

    # get_data(MIX_PORTFOLIO, mix_portfolio_ticker_objects, 'portfolio_data/mix')
    get_google_portfolio('mix')
    print('----- MIX done -----')

    # get_data(MOMENTUM_PORTFOLIO, momentum_portfolio_ticker_objects, 'portfolio_data/momentum')
    # get_google_portfolio('momentum')
    # print('----- Momentum done -----')
    #
    # get_data(CONTRARIAN_PORTFOLIO, contrarian_portfolio_ticker_objects, 'portfolio_data/contrarian')
    # get_google_portfolio('contrarian')
    # print('----- Contrarian done -----')

    # get_data(RANDOM_STOCKS, random_stocks_ticker_objects, 'random_stocks_data')
    # get_google_random_stocks()
    # print('----- Random Stocks done -----')


