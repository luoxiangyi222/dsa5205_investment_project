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

    # metrics = mean_square_error, mean absolute error
    return model


def inverse_back_to_price(scaled_price, original_price):
    min_price = min(original_price)
    max_price = max(original_price)
    width = max_price - min_price

    price = scaled_price * width + min_price
    return price


def train(stock_name, feature_subset=None, is_PCA=False):

    stock_df = pd.read_csv(f'../trading_strategy_data/combined_data/{stock_name}_combined_data.csv')

    stock_df = stock_df[stock_df['Date'] < '2021-10-01']

    stock_df = stock_df[['Ticker', 'Date', 'SMA_10', 'MACD', 'Open', 'Close']]

    scaled_df = pd.DataFrame(scaler.fit_transform(stock_df.iloc[:, 2:]).copy())
    scaled_df.columns = stock_df.columns[2:]

    N = stock_df.shape[0]
    term_length_list = [2, 5, 10]

    term_length = 2

    # convert to time series data
    X_arr, Y_df = time_series_conversion_for_LSTM(scaled_df, term_length)
    # do regression task
    Y_arr = np.array(Y_df['Close'])

    # split data
    validation_length = 43
    train_X_arr, val_X_arr = X_arr[:-validation_length, :], X_arr[-validation_length:, :]
    train_Y_arr, val_Y_arr = Y_arr[:-validation_length], Y_arr[-validation_length:]

    lstm = compile_LSTM_model(train_X_arr.shape[1], train_X_arr.shape[2])
    history = lstm.fit(train_X_arr, train_Y_arr, batch_size=2, epochs=50)

    # return lstm, history
    plt.plot(history.history['loss'])
    plt.title(f'Training loss for {stock_name}')
    plt.savefig(f'../trading_strategy_figure/{stock_name}_training_loss.jpeg')

    # get prediction
    train_pred_Y = lstm.predict(train_X_arr)
    train_pred_Y = inverse_back_to_price(train_pred_Y, np.array(stock_df['Close']))
    val_pred_Y = lstm.predict(val_X_arr)
    val_pred_Y = inverse_back_to_price(val_pred_Y, np.array(stock_df['Close']))

    val_Y_arr = inverse_back_to_price(val_Y_arr, np.array(stock_df['Close']))
    yy = pd.concat([pd.DataFrame(val_Y_arr), pd.DataFrame(val_pred_Y)], axis=1)
    yy.columns = ['truth', 'pred']

    # Plot the data
    data = stock_df[['Date', 'Close']].copy()
    data = data.iloc[term_length:, :]

    train_set = data.iloc[:-validation_length]
    train_set['pred'] = train_pred_Y
    valid_set = data.iloc[-validation_length:]
    valid_set['pred'] = val_pred_Y

    # Visualize the data
    plt.figure(figsize=(14, 8))
    plt.title(f'LSTM Model prediction for {stock_name}', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price USD ($)', fontsize=12)
    plt.plot(train_set[['Close', 'pred']])
    plt.plot(valid_set[['Close', 'pred']])
    plt.legend(['Train', 'Train_predict', 'Val', 'Val_predict'], loc='lower right')
    plt.savefig(f'../trading_strategy_figure/{stock_name}_prediction.jpeg')

    # TODO: output Date, true open price, true close price, predict next-day close price


if __name__ == "__main__":

    for stock in crawler.PORTFOLIO:
        train(stock)


