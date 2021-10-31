"""
Author: Luo Xiangyi
Create on: 2021-10-24

This module does data preprocessing and feature selection.
"""
# RFE blog
# https://machinelearningmastery.com/rfe-feature-selection-in-python/

import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import trading_strategy.data_crawler as crawler
from collections import defaultdict
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from yellowbrick.model_selection import RFECV as yellow_rfecv
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def RFE(stock):

    print(f'RFE {stock}')

    rfe_df = pd.read_csv(f'../trading_strategy_data/random_stocks_data/{stock}_combined_data.csv')

    D = rfe_df.shape[1] - 2

    X_df = rfe_df.iloc[:, 2:]
    X_arr = np.array(X_df)[:-1]
    Y_df = rfe_df[['Close']].shift(periods=-1).copy()
    Y_arr = np.array(Y_df)[:-1]

    lr = LinearRegression(normalize=True)
    dt = DecisionTreeRegressor()

    rfecv = RFECV(estimator=lr,
                  step=1,
                  cv=5,
                  scoring='neg_mean_squared_error',
                  )

    rfecv.fit(X_arr, Y_arr)

    # plt.figure()
    # plt.title(stock)
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (neg_mse)")
    # plt.plot(
    #     range(1, len(rfecv.grid_scores_) + 1),
    #     rfecv.grid_scores_,
    # )
    # plt.show()

    # # plot cv scores
    # plt.figure()
    # plt.title(stock)
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (neg mse)")
    # plt.plot(
    #     rfecv.cv_results_['mean_test_score']
    # )
    # plt.show()

    index_rank = rfecv.ranking_

    col_names = X_df.columns[1:]
    ranking = sorted((zip(index_rank, col_names)))

    return ranking


def RFE_voting():

    feature_ranking_list = []
    aggregate_ranking = defaultdict(lambda: 0)   # (total score)

    # RFE for feature selection
    for stock in crawler.RANDOM_STOCKS:
        # print('~~~~~' + stock + '~~~~~')
        ranking = RFE(stock)
        feature_ranking_list.append(ranking)
        for s, f in ranking:
            aggregate_ranking[f] += s

    sort_f_dict = dict(sorted(aggregate_ranking.items(), key=lambda item: item[1]))
    # change to average score
    sort_f_dict = {k: round(v/30.0, 6) for (k, v) in sort_f_dict.items()}
    print(sort_f_dict)

    sorted_features = list(sort_f_dict.keys())
    average_scores = list(sort_f_dict.values())
    print(sorted_features)
    print(average_scores)

    feature_ticks = [f + ' ' + str(i) for i, f in enumerate(sorted_features)]
    print(feature_ticks)

    # plot ranking information
    plt.figure(figsize=(20, 10))
    # plot average scores
    plt.plot(average_scores, linestyle='--', marker='o')

    # plot ranking for each stock
    scatter_plot_list = []
    for i, score_feature_pairs in enumerate(feature_ranking_list):
        print(score_feature_pairs)
        scores = [r[0] for r in score_feature_pairs]
        features = [r[1] for r in score_feature_pairs]
        x_score = [scores[features.index(x_f)] for x_f in sorted_features]
        scatter_plot = plt.scatter(x=range(len(x_score)), y=x_score, alpha=0.5, marker='^', s=14)
        scatter_plot_list.append(scatter_plot)
    plt.legend(scatter_plot_list,
               crawler.RANDOM_STOCKS,
               loc='center left',
               bbox_to_anchor=(1.0, 0.5)
               )

    plt.xticks(list(range(len(feature_ticks))), feature_ticks, rotation=90, fontsize=8)
    plt.xlabel('Features')
    plt.ylabel('Rank Score')
    plt.title('Feature Ranking Score', fontsize=18)
    plt.savefig('../trading_strategy_figure/RFE/feature_rank_score.jpeg', bbox_inches='tight')
    plt.show()

    # Save first 40 selected features
    selected_features = sorted_features[:40]
    print(selected_features)
    with open("../trading_strategy_figure/RFE/rfe_selected_features.txt", 'w') as file:
        s = ",".join(selected_features)
        file.write(s)

    return selected_features


def preprocessing(select_strategy, folder_name):

    # get RFE selected features
    with open('../trading_strategy_figure/RFE/rfe_selected_features.txt') as f:
        line = f.read()

    features = line.split(',')

    if select_strategy == 'momentum':
        portfolio = crawler.MOMENTUM_PORTFOLIO
    elif select_strategy == 'contrarian':
        portfolio = crawler.CONTRARIAN_PORTFOLIO
    elif select_strategy == 'mix':
        portfolio = crawler.MIX_PORTFOLIO
    else:
        raise ValueError()

    for stock in portfolio:
        print(f'=========={stock}===========')
        df = pd.read_csv(f'../trading_strategy_data/{folder_name}/{stock}_combined_data.csv')

        # ############## RFE ################
        rfe_df = df[features]
        rfe_df = pd.concat([df[['Ticker', 'Date']], rfe_df], axis=1)
        rfe_df.to_csv(f'../trading_strategy_data/{folder_name}/{stock}_rfe_data.csv', index=False)
        print(f'~~~~~ {stock} RFE data saved ! ~~~~~')

        # ############## PCA ################
        # keep close price
        Y = df['Close'].copy()
        # get selected columns
        X = np.array(df[features])
        # min max transformation
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        # PCA
        # pca = PCA()
        pca = PCA(n_components=10)  # 10 PCs keep 90% variance
        pca_X = pca.fit_transform(X)

        col_names = []
        for i in range(pca_X.shape[1]):
            col_names.append(f'PCA_{i}')

        pca_df = pd.DataFrame(pca_X, columns=col_names)
        pca_df = pd.concat([df[['Ticker', 'Date']], pca_df, Y], axis=1)
        pca_df.to_csv(f'../trading_strategy_data/{folder_name}/{stock}_pca_data.csv', index=False)
        print(f' $$$$$ {stock} PCA data saved ! $$$$$')

        # ############## PLOT ################

        # # plot PCA ratio
        # plt.figure(figsize=(10, 8))
        # plt.plot(pca.explained_variance_ratio_, linestyle='--', marker='o')
        # plt.xlabel('PC')
        # plt.ylabel('Percentage of variance explained')
        # plt.title(f'{stock}: PC explained variance plot')
        # plt.savefig(f'../trading_strategy_figure/PCA/{select_strategy}/PCA_{stock}.jpeg', bbox_inches='tight')
        # # plt.show()
        # print(f' $$$$$ {stock} PCA ratio plot figure saved ! $$$$$')

    return True


if __name__ == "__main__":

    rfe_result = RFE_voting()

    # preprocessing('mix', 'portfolio_data/mix')
    # preprocessing('momentum', 'portfolio_data/momentum')
    # preprocessing('contrarian', 'portfolio_data/contrarian')






