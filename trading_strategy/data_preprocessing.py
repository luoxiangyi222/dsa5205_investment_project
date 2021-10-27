"""
Author: Luo Xiangyi
Create on: 2021-10-24

This module does data preprocessing and feature selection.
"""
# RFE blog
# https://machinelearningmastery.com/rfe-feature-selection-in-python/

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import trading_strategy.data_crawler as crawler
from collections import defaultdict

from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def RFE(stock):

    rfe_df = pd.read_csv(f'../trading_strategy_data/random_stocks_data/{stock}_combined_data.csv')

    X_df = rfe_df.iloc[:, 2:]
    X_arr = np.array(X_df)[:-1]
    Y_df = rfe_df[['Close']].shift(periods=-1).copy()
    Y_arr = np.array(Y_df)[:-1]

    selector = RFECV(estimator=DecisionTreeRegressor(), step=1, cv=5)

    selector.fit(X_arr, Y_arr)

    index_rank = selector.ranking_
    col_names = X_df.columns[1:]
    ranking = sorted((zip(index_rank, col_names)))
    ranking = [r[1] for r in ranking]
    return ranking


def RFE_voting():

    feature_ranking_list = []
    aggregate_ranking = defaultdict(lambda: 0)

    # RFE for feature selection
    for stock in crawler.RANDOM_STOCKS:
        ranking = RFE(stock)
        feature_ranking_list.append(ranking)

    for rank_list in feature_ranking_list:
        for i, f in enumerate(rank_list):
            aggregate_ranking[f] += i

    sort_f = sorted(aggregate_ranking.items(), key=lambda x: x[1])

    feature_ticks = [f[0] + ' ' + str(i) for i, f in enumerate(sort_f)]
    features = [f[0] for f in sort_f]
    scores = [f[1] for f in sort_f]

    plt.figure(figsize=(14, 12))
    plt.plot(scores, linestyle='--', marker='o')

    plt.xticks(list(range(len(feature_ticks))), feature_ticks, rotation=90, fontsize=8)
    plt.xlabel('Feature rank')
    plt.ylabel('Rank Score')
    plt.title('Feature Ranking Score', fontsize=18)
    plt.savefig('../trading_strategy_figure/feature_rank_score.jpeg', bbox_inches='tight')

    features = features[:43]
    with open("../trading_strategy_figure/rfe_selected_features.txt", 'w') as file:
        s = ",".join(features)
        file.write(s)
    return features


def preprocessing():

    with open('../trading_strategy_figure/rfe_selected_features.txt') as f:
        line = f.read()

    features = line.split(',')

    for stock in crawler.PORTFOLIO:
        print(f'=========={stock}===========')
        df = pd.read_csv(f'../trading_strategy_data/portfolio_data/{stock}_combined_data.csv')

        rfe_df = df[features]
        rfe_df = pd.concat([df[['Ticker','Date']], rfe_df], axis=1)
        print(rfe_df.head())
        rfe_df.to_csv(f'../trading_strategy_data/portfolio_data/{stock}_rfe_data.csv', index=False)

        # keep close price
        Y = df['Close'].copy()

        # get selected columns
        X = np.array(df[features])
        # min max transformation
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        pca = PCA(n_components=10)
        pca_X = pca.fit_transform(X)
        print(pca_X.shape)

        col_names = []
        for i in range(pca_X.shape[1]):
            col_names.append(f'PCA_{i}')

        pca_df = pd.DataFrame(pca_X, columns=col_names)
        pca_df = pd.concat([df[['Ticker', 'Date']], pca_df, Y], axis=1)
        pca_df.to_csv(f'../trading_strategy_data/portfolio_data/{stock}_pca_data.csv', index=False)

        # plot ratio
        # plt.figure(figsize=(10, 8))
        # plt.plot(pca.explained_variance_ratio_, linestyle='--', marker='o')
        # plt.xlabel('PC')
        # plt.ylabel('percentage of variance explained')
        # plt.title(f'{stock}: PCA results')
        # plt.savefig(f'../trading_strategy_figure/PCA_{stock}.jpeg', bbox_inches='tight')


if __name__ == "__main__":

    # rfe_result = RFE_voting()

    preprocessing()





