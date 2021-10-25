"""
Author: Luo Xiangyi
Create on: 2021-10-25

This module crawls data from different sources and save them locally.
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def time_series_conversion_for_LSTM(df, term_length, lag):
    cols = []
    for i in range(term_length, 0, -1):
        shift_df = df.shift(periods=i)
        cols.append(shift_df)

    for i in range(lag):
        shift_df = df.shift(periods=-1)
        cols.append(shift_df)

    time_series_df = pd.concat(cols, axis=1)
    return time_series_df


def compile_LSTM_model(feature_count, X_shape_1, X_shape_2):
    model = Sequential()
    layer_1 = LSTM(feature_count, input_shape=(X_shape_1, X_shape_2))
    layer_2 = Dense(1)
    model.add(layer_1)
    model.add(layer_2)

    loss_function = 'mean_absolute_error'
    optimizer = 'adam'
    model.compile(loss=loss_function, optimizer=optimizer)

    # metrics = mean_square_error, mean absolute error
    return model
