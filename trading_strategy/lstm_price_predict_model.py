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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)  # enable to see all columns
# pd.set_option('display.max_rows', None)  # enable to see all rows
import tensorflow as tf


def time_series_conversion_for_LSTM(df, term_length):
    N = df.shape[0]
    D = df.shape[1]
    cols = []

    for i in range(term_length, 0, -1):
        shift_df = df.shift(periods=i).iloc[:, 2:].copy()

        print('=======')
        print(shift_df)

        cols.append(shift_df.iloc[term_length:, :])

    X = pd.concat(cols, axis=1)
    Y = df[['Close']].iloc[term_length:].copy()

    X = np.array(X).reshape([N-term_length, term_length, D-2])
    Y = np.array(Y)

    return X, Y


def compile_LSTM_model(X_shape_1, X_shape_2):
    model = Sequential()
    layer_1 = LSTM(50, input_shape=(X_shape_1, X_shape_2))
    layer_2 = Dense(1)
    model.add(layer_1)
    model.add(layer_2)

    model.compile(loss='mse', optimizer='adam')

    # metrics = mean_square_error, mean absolute error
    return model


def fit_LSTM():
    pass


def evaluate_LSTM():
    pass


def train(stock_name, feature_subset=None, is_PCA=False):

    stock_df = pd.read_csv(f'../trading_strategy_data/combined_data/{stock_name}_combined_data.csv')

    stock_df = stock_df[['Ticker', 'Date', 'SMA_10', 'MACD', 'Open', 'Close']]

    # TODO min-max scaling to speeding convergence
    scaler = MinMaxScaler(feature_range=(0, 1))
    # stock_df[['Close']] = scaler.fit_transform(stock_df[['Close']])

    N = stock_df.shape[0]
    D = stock_df.shape[1]
    term_length = 3

    # convert to time series data
    X_arr, Y_arr = time_series_conversion_for_LSTM(stock_df, term_length)

    # TODO: use Aug and Sep for validation
    training_size = int(N * 0.90)

    # split data
    train_X_arr, val_X_arr = X_arr[:training_size, :], X_arr[training_size:, :]
    train_Y_arr, val_Y_arr = Y_arr[:training_size], Y_arr[training_size:]

    lstm = compile_LSTM_model(train_X_arr.shape[1], train_X_arr.shape[2])
    lstm.fit(train_X_arr, train_Y_arr, batch_size=2, epochs=200)

    train_pred_Y = lstm.predict(train_X_arr)

    val_pred_Y = lstm.predict(val_X_arr)

    # Plot the data
    data = stock_df[['Date', 'Close']]
    train = data.iloc[term_length:term_length + training_size]
    train['pred'] = train_pred_Y

    valid = data.iloc[term_length + training_size:]
    valid['pred'] = val_pred_Y

    # Visualize the data
    plt.figure(figsize=(16, 6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train[['Close', 'pred']])
    plt.plot(valid[['Close', 'pred']])
    plt.legend(['Train', 'Train_predict', 'Val', 'Val_predict'], loc='lower right')
    plt.show()


if __name__ == "__main__":

    train('AAPL')


