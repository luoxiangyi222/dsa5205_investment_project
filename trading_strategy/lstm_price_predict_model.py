"""
Author: Luo Xiangyi
Create on: 2021-10-25

This module build LSTM model and save formatted output.
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
        shift_df = df.shift(periods=i).copy()

        cols.append(shift_df.iloc[term_length:, :])

    X = pd.concat(cols, axis=1)

    # Close: regression task
    # Target: classification task
    Y = df[['Close']].copy()
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

    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error'])

    return model


def inverse_back_to_price(scaled_price, original_price):
    min_price = min(original_price)
    max_price = max(original_price)
    width = max_price - min_price

    price = scaled_price * width + min_price
    return price


def train(stock_name, term_length, read_folder_name,  data_description):

    port_type = read_folder_name.split('/')[1]

    validation_length = 43  # validation starting from 2021-08-02

    df = None

    if data_description == 'rfe_pca':

        batch_size = 2
        epochs = 50

        df = pd.read_csv(f'../trading_strategy_data/{read_folder_name}/{stock_name}_pca_data.csv')
        df = df[df['Date'] < '2021-10-01'].copy()

        # get rid of Ticker and Date
        stock_df = df.iloc[:, 2:]
        stock_df.columns = df.columns[2:]

        # convert to time series data
        X_arr, Y_df = time_series_conversion_for_LSTM(stock_df, term_length)
        Y_arr = np.array(Y_df['Close'])

        # split data
        train_X_arr, val_X_arr = X_arr[:-validation_length, :], X_arr[-validation_length:, :]
        train_Y_arr, val_Y_arr = Y_arr[:-validation_length], Y_arr[-validation_length:]

        lstm = compile_LSTM_model(train_X_arr.shape[1], train_X_arr.shape[2])
        history = lstm.fit(train_X_arr, train_Y_arr, batch_size=batch_size, epochs=epochs, validation_data=(val_X_arr, val_Y_arr))

        subtitle_string = f'data: {data_description}, term_length: {term_length}\n' \
                          f' batch_size={batch_size}, epochs={epochs}, lstm output dim: 50'

        # get train and validation prediction
        train_pred_Y = lstm.predict(train_X_arr)
        val_pred_Y = lstm.predict(val_X_arr)



        # save training period prediction
        train_yy_df = df.iloc[term_length:-validation_length, 1]

        train_yy_df = pd.concat(
            [pd.DataFrame(train_yy_df).reset_index(drop=True), pd.DataFrame(train_Y_arr), pd.DataFrame(train_pred_Y)],
            axis=1)
        train_yy_df.columns = ['Date', 'true_Close', 'pred_Close']
        train_yy_df.to_csv(
            f'../trading_strategy_data/lstm_results/{port_type}/{stock_name}_{data_description}_{term_length}_train_prediction.csv',
            index=False)
        print('===== Train prediction saved ! =====')

        # save validation period prediction
        val_yy_df = df.iloc[-validation_length:, 1]
        val_yy_df = pd.concat(
            [pd.DataFrame(val_yy_df).reset_index(drop=True), pd.DataFrame(val_Y_arr), pd.DataFrame(val_pred_Y)], axis=1)
        val_yy_df.columns = ['Date', 'true_Close', 'pred_Close']
        val_yy_df.to_csv(
            f'../trading_strategy_data/lstm_results/{port_type}/{stock_name}_{data_description}_{term_length}_val_prediction.csv',
            index=False)
        print('===== Valid prediction saved ! =====')

        return lstm, history, train_yy_df, val_yy_df

    else:
        batch_size = 2
        epochs = 20

        if data_description == 'all':
            df = pd.read_csv(f'../trading_strategy_data/{read_folder_name}/{stock_name}_combined_data.csv')
        elif data_description == 'rfe':
            df = pd.read_csv(f'../trading_strategy_data/{read_folder_name}/{stock_name}_rfe_data.csv')

        df = df[df['Date'] < '2021-10-01']

        # min-max scaling
        stock_df = pd.DataFrame(scaler.fit_transform(df.iloc[:, 2:]).copy())
        stock_df.columns = df.columns[2:]

        # convert to time series data
        X_arr, Y_df = time_series_conversion_for_LSTM(stock_df, term_length)
        # do regression task
        Y_arr = np.array(Y_df['Close'])

        # split data
        train_X_arr, val_X_arr = X_arr[:-validation_length, :], X_arr[-validation_length:, :]
        train_Y_arr, val_Y_arr = Y_arr[:-validation_length], Y_arr[-validation_length:]

        lstm = compile_LSTM_model(train_X_arr.shape[1], train_X_arr.shape[2])
        history = lstm.fit(train_X_arr, train_Y_arr, batch_size=batch_size, epochs=epochs, validation_data=(val_X_arr, val_Y_arr))

        subtitle_string = f'data: {data_description}, term_length: {term_length}\n' \
                          f' batch_size={batch_size}, epochs={epochs}, lstm output dim: 50'

        # get prediction
        train_pred_Y = lstm.predict(train_X_arr)
        train_pred_Y = inverse_back_to_price(train_pred_Y, np.array(df['Close']))
        val_pred_Y = lstm.predict(val_X_arr)
        val_pred_Y = inverse_back_to_price(val_pred_Y, np.array(df['Close']))

        # save training period prediction
        train_yy_df = df.iloc[term_length:-validation_length, 1]
        train_Y_arr = inverse_back_to_price(train_Y_arr, np.array(df['Close']))
        train_yy_df = pd.concat(
            [pd.DataFrame(train_yy_df).reset_index(drop=True), pd.DataFrame(train_Y_arr), pd.DataFrame(train_pred_Y)], axis=1)
        train_yy_df.columns = ['Date', 'true_Close', 'pred_Close']
        train_yy_df.to_csv(
            f'../trading_strategy_data/lstm_results/{port_type}/{stock_name}_{data_description}_{term_length}_train_prediction.csv',
            index=False)

        # save validation period prediction
        val_yy_df = df.iloc[-validation_length:, 1]
        val_Y_arr = inverse_back_to_price(val_Y_arr, np.array(df['Close']))
        val_yy_df = pd.concat([pd.DataFrame(val_yy_df).reset_index(drop=True), pd.DataFrame(val_Y_arr), pd.DataFrame(val_pred_Y)], axis=1)
        val_yy_df.columns = ['Date', 'true_Close', 'pred_Close']
        val_yy_df.to_csv(
            f'../trading_strategy_data/lstm_results/{port_type}/{stock_name}_{data_description}_{term_length}_val_prediction.csv',
            index=False)

        return lstm, history, train_yy_df, val_yy_df


if __name__ == "__main__":
    #
    # for stock in crawler.PORTFOLIO:
    #     train(stock)

    term_length_list = [2, 5, 10]
    descrip_list = ['all', 'rfe', 'rfe_pca']

    # Train using original data
    # for stock in crawler.MIX_PORTFOLIO:
    #
    #     train(stock, 5, folder_name='portfolio_data/mix', data_description='rfe_pca')


    for stock in crawler.MOMENTUM_PORTFOLIO:

        pairs_str_list = []

        train_loss_list = []
        valid_loss_list = []
        train_pred_y_list = []
        valid_pred_y_list = []

        original_date_price_df_list = []

        for data_des in descrip_list:
            for tl in term_length_list:

                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                tag = stock + '_' + data_des + '_' + str(tl)
                pairs_str_list.append(tag)
                print(tag)

                model, history, train_yy, valid_yy = \
                    train(stock, tl, read_folder_name='portfolio_data/momentum', data_description=data_des)

                train_loss_list.append(history.history['loss'])
                valid_loss_list.append(history.history['val_loss'])
                train_pred_y_list.append(train_yy)
                valid_pred_y_list.append(valid_yy)

        # plot train and valid loss
        fig_1, ax = plt.subplots(3, 3, figsize=(16, 12))
        fig_1.suptitle(stock, fontsize=20)
        ax = ax.flat
        for i in range(len(pairs_str_list)):

            ax[i].set_title(pairs_str_list[i])
            if i < 6:
                ax[i].set_ylim([0, 0.03])

            else:
                ax[i].set_ylim([0, 8000])
            ax[i].plot(train_loss_list[i], linestyle='--', marker='o', color='lightsalmon' )
            ax[i].plot(valid_loss_list[i], linestyle='--', marker='o', color='lightskyblue')
            ax[i].legend(['train_loss', 'val_loss'])

        plt.savefig(f'../trading_strategy_figure/LSTM/momentum/{stock}_loss.jpeg', bbox_inches='tight')
        plt.show()

        # plot price prediction

        df = pd.read_csv(f'../trading_strategy_data/portfolio_data/momentum/{stock}_combined_data.csv')
        df = df[df['Date'] < '2021-10-01']
        df = df[['Date', 'Close']]

        # plot all
        fig_2 = plt.figure(figsize=(16, 12))
        plt.plot(list(range(df.shape[0])), df['Close'])
        valid_length = 43

        for i in range(6, 9):
            tl = 0
            if i % 3 == 0:
                tl = 2
            elif i % 3 == 1:
                tl = 5
            else:
                tl = 10

            train_yy = train_pred_y_list[i]
            valid_yy = valid_pred_y_list[i]
            yy = pd.concat([train_yy, valid_yy], axis=0)
            print(yy)

            plt.plot(list(range(tl, df.shape[0])), yy['pred_Close'], linestyle='--')

        plt.legend(['Truth'] + pairs_str_list)
        plt.show()







