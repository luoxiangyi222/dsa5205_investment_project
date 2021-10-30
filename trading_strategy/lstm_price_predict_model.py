"""
Author: Luo Xiangyi
Create on: 2021-10-25

This module crawls data from different sources and save them locally.
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import trading_strategy.data_crawler as crawler

import tensorflow as tf

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

pd.set_option('display.max_columns', None)  # enable to see all columns
pd.set_option('display.max_rows', None)  # enable to see all rows


scaler = MinMaxScaler(feature_range=(0, 1))


def time_series_conversion_for_LSTM(df, term_length):
    N = df.shape[0]
    D = df.shape[1]
    cols = []

    for i in range(term_length, 0, -1):
        shift_df = df.shift(periods=i).iloc[:, :].copy()

        cols.append(shift_df.iloc[term_length:, :])

    X = pd.concat(cols, axis=1)
    Y = df[['Close']].copy()

    # Add classification task
    Y['Target'] = 0
    Y.loc[Y['Close'] - Y['Close'].shift(periods=1) > 0, 'Target'] = 1

    X = np.array(X).reshape([N-term_length, term_length, D])
    Y = Y.iloc[term_length:, :]

    return X, Y


def compile_LSTM_model(X_shape_1, X_shape_2):
    model = Sequential()
    layer_1 = LSTM(50, input_shape=(X_shape_1, X_shape_2))
    layer_2 = Dense(1)

    model.add(layer_1)
    model.add(layer_2)

    model.compile(loss='mae', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error'])

    return model


def inverse_back_to_price(scaled_price, original_price):
    min_price = min(original_price)
    max_price = max(original_price)
    width = max_price - min_price

    price = scaled_price * width + min_price
    return price


def train(stock_name, term_length=2, data_description='all'):

    validation_length = 43
    df = None


    if data_description == 'rfe_pca':

        df = pd.read_csv(f'../trading_strategy_data/portfolio_data/{stock_name}_pca_data.csv')
        df = df[df['Date'] < '2021-10-01'].copy()

        stock_df = df.iloc[:, 2:]
        stock_df.columns = df.columns[2:]

        X_arr, Y_df = time_series_conversion_for_LSTM(stock_df, term_length)
        Y_arr = np.array(Y_df['Close'])

        # split data
        train_X_arr, val_X_arr = X_arr[:-validation_length, :], X_arr[-validation_length:, :]
        train_Y_arr, val_Y_arr = Y_arr[:-validation_length], Y_arr[-validation_length:]


        lstm = compile_LSTM_model(train_X_arr.shape[1], train_X_arr.shape[2])
        history = lstm.fit(train_X_arr, train_Y_arr, batch_size=2, epochs=50)

        subtitle_string = f'data: {data_description} term_length: {term_length}\n' \
                          f' batch_size=2, epochs=50 lstm output dim: 50'

        # get prediction
        train_pred_Y = lstm.predict(train_X_arr)
        val_pred_Y = lstm.predict(val_X_arr)

        # save training period prediction
        output_df = df.iloc[term_length:-validation_length, 1]

        train_yy_df = pd.concat(
            [pd.DataFrame(output_df).reset_index(drop=True), pd.DataFrame(train_Y_arr), pd.DataFrame(train_pred_Y)],
            axis=1)
        train_yy_df.columns = ['Date', 'true_Close', 'pred_Close']
        train_yy_df.to_csv(
            f'../trading_strategy_data/lstm_results/{stock_name}_{data_description}_{term_length}_train_prediction.csv',
            index=False)
        print('===== Train prediction saved =====')

        # save validation period prediction
        output_df = df.iloc[-validation_length:, 1]
        val_yy_df = pd.concat(
            [pd.DataFrame(output_df).reset_index(drop=True), pd.DataFrame(val_Y_arr), pd.DataFrame(val_pred_Y)], axis=1)
        val_yy_df.columns = ['Date', 'true_Close', 'pred_Close']
        val_yy_df.to_csv(
            f'../trading_strategy_data/lstm_results/{stock_name}_{data_description}_{term_length}_val_prediction.csv',
            index=False)
        print('===== Valid prediction saved =====')

        # plot training loss
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'], linestyle='--', marker='o')
        plt.title(f'Training loss for {stock_name}')
        plt.suptitle(subtitle_string)
        plt.xlabel('epochs')
        plt.ylabel('mean absolute error')
        # plt.show()
        plt.savefig(f'../trading_strategy_figure/{stock_name}_{data_description}_{term_length}_training_loss.jpeg')
        print('===== Train loss figure saved =====')


        # Plot the data
        # TODO 2: also show LSTM model setting,
        #  e.g. original data / selected features / after PCA, aiming to show improvement,
        #  e.g. batch size / epoch
        #  e.g. lstm layer / hidden unit number

        data = df[['Date', 'Close']].copy()
        data = data.iloc[term_length:, :]

        # prepare plot data
        train_set = data.iloc[:-validation_length]
        train_set['pred'] = train_pred_Y
        valid_set = data.iloc[-validation_length:]
        valid_set['pred'] = val_pred_Y

        # plot prediction
        plt.figure(figsize=(14, 8))
        plt.title(f'LSTM Model prediction for {stock_name}', fontsize=18)
        plt.suptitle(subtitle_string)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price USD ($)', fontsize=12)
        plt.plot(train_set[['Close', 'pred']])
        plt.plot(valid_set[['Close', 'pred']])
        plt.legend(['Train', 'Train_predict', 'Val', 'Val_predict'], loc='lower right')
        # plt.show()
        plt.savefig(f'../trading_strategy_figure/{stock_name}_{data_description}_{term_length}_prediction.jpeg')
        print('===== Price prediction saved =====')

        print(' ++++++++++ ')

    else:

        if data_description == 'all':
            df = pd.read_csv(f'../trading_strategy_data/portfolio_data/{stock_name}_combined_data.csv')
        elif data_description == 'rfe':
            df = pd.read_csv(f'../trading_strategy_data/portfolio_data/{stock_name}_combined_data.csv')

        df = df[df['Date'] < '2021-10-01']

        # min-max scaling
        stock_df = pd.DataFrame(scaler.fit_transform(df.iloc[:, 2:]).copy())
        stock_df.columns = df.columns[2:]

        # convert to time series data
        X_arr, Y_df = time_series_conversion_for_LSTM(stock_df, term_length)
        # do regression task
        Y_arr = np.array(Y_df['Close'])

        # print(df.iloc[:5, [1, 2, -1]])
        # print(stock_name)
        # for i in range(5):
        #     print(X_arr[i])
        #     print(Y_arr[i])
        # breakpoint()

        # split data
        train_X_arr, val_X_arr = X_arr[:-validation_length, :], X_arr[-validation_length:, :]
        train_Y_arr, val_Y_arr = Y_arr[:-validation_length], Y_arr[-validation_length:]

        lstm = compile_LSTM_model(train_X_arr.shape[1], train_X_arr.shape[2])
        history = lstm.fit(train_X_arr, train_Y_arr, batch_size=2, epochs=50)

        subtitle_string = f'data: {data_description} term_length: {term_length}\n' \
                          f' batch_size=2, epochs=50 lstm output dim: 50'

        # get prediction
        train_pred_Y = lstm.predict(train_X_arr)
        train_pred_Y = inverse_back_to_price(train_pred_Y, np.array(df['Close']))
        val_pred_Y = lstm.predict(val_X_arr)
        val_pred_Y = inverse_back_to_price(val_pred_Y, np.array(df['Close']))

        # save training period prediction
        output_df = df.iloc[term_length:-validation_length, 1]
        train_Y_arr = inverse_back_to_price(train_Y_arr, np.array(df['Close']))
        train_yy_df = pd.concat(
            [pd.DataFrame(output_df).reset_index(drop=True), pd.DataFrame(train_Y_arr), pd.DataFrame(train_pred_Y)], axis=1)
        train_yy_df.columns = ['Date', 'true_Close', 'pred_Close']
        train_yy_df.to_csv(
            f'../trading_strategy_data/lstm_results/{stock_name}_{data_description}_{term_length}_train_prediction.csv',
            index=False)
        print(train_yy_df)

        # save validation period prediction
        output_df = df.iloc[-validation_length:, 1]
        val_Y_arr = inverse_back_to_price(val_Y_arr, np.array(df['Close']))
        val_yy_df = pd.concat([pd.DataFrame(output_df).reset_index(drop=True), pd.DataFrame(val_Y_arr), pd.DataFrame(val_pred_Y)], axis=1)
        val_yy_df.columns = ['Date', 'true_Close', 'pred_Close']
        val_yy_df.to_csv(f'../trading_strategy_data/lstm_results/{stock_name}_{data_description}_{term_length}_val_prediction.csv', index=False)
        print(val_yy_df)


        # # plot training loss
        # plt.figure(figsize=(10, 8))
        # plt.plot(history.history['loss'], linestyle='--', marker='o')
        # plt.title(f'Training loss for {stock_name}')
        # plt.suptitle(subtitle_string)
        # plt.xlabel('epochs')
        # plt.ylabel('mean absolute error')
        # plt.show()
        # # plt.savefig(f'../trading_strategy_figure/{stock_name}_{data_description}_{term_length}_training_loss.jpeg')
        #
        #
        # # Plot the data
        # # TODO 2: also show LSTM model setting,
        # #  e.g. original data / selected features / after PCA, aiming to show improvement,
        # #  e.g. batch size / epoch
        # #  e.g. lstm layer / hidden unit number
        #
        # data = df[['Date', 'Close']].copy()
        # data = data.iloc[term_length:, :]
        #
        # # prepare plot data
        # train_set = data.iloc[:-validation_length]
        # train_set['pred'] = train_pred_Y
        # valid_set = data.iloc[-validation_length:]
        # valid_set['pred'] = val_pred_Y
        #
        # # plot prediction
        # plt.figure(figsize=(14, 8))
        # plt.title(f'LSTM Model prediction for {stock_name}', fontsize=18)
        # plt.suptitle(subtitle_string)
        # plt.xlabel('Date', fontsize=12)
        # plt.ylabel('Close Price USD ($)', fontsize=12)
        # plt.plot(train_set[['Close', 'pred']])
        # plt.plot(valid_set[['Close', 'pred']])
        # plt.legend(['Train', 'Train_predict', 'Val', 'Val_predict'], loc='lower right')
        # plt.show()
        # # plt.savefig(f'../trading_strategy_figure/{stock_name}_{data_description}_{term_length}_prediction.jpeg')

    # TODO 3: output format
    #  Date, true open price, true close price, predict next-day close price


if __name__ == "__main__":
    #
    # for stock in crawler.PORTFOLIO:
    #     train(stock)

    term_length_list = [2, 5, 10]
    descrip_list = ['all', 'rfe', 'rfe_pca']

    # Train using original data
    for stock in crawler.PORTFOLIO:
        train(stock, 2, 'all')
        train(stock, 5, 'all')
        train(stock, 10, 'all')

    # Train using RFE selected data
    for stock in crawler.PORTFOLIO:
        train(stock, 2, 'rfe')
        train(stock, 5, 'rfe')
        train(stock, 10, 'rfe')

    # Train using PCA selected data
    for stock in crawler.PORTFOLIO:
        train(stock, 2, 'rfe_pca')
        train(stock, 5, 'rfe_pca')
        train(stock, 10, 'rfe_pca')


