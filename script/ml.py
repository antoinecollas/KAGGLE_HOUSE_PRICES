#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
# %matplotlib inline

TEST_SIZE = 0.2

def data_preparation(path, train=True):
    df_original = pd.read_csv(path)

    if train == True:
        cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        df = df_original[cols]
        
        #remove outliars
        df = df.drop([1299, 524])
        # print(df)
        # print(df['SalePrice'].mean())
        # print(df['SalePrice'].std())

        # SalePrice
        # sns.distplot(df['SalePrice'], fit=norm)

        df['SalePrice'] = df['SalePrice'].apply(np.log)
        # print(df['SalePrice'].describe())

        
        # print(stats.kstest(df['SalePrice'], 'norm'))
        
        # sns.distplot(df['SalePrice'], fit=norm)
        df['GrLivArea'] = df['GrLivArea'].apply(np.log)
        scaler = StandardScaler()
        sale_price = df['SalePrice'].copy()
        df = pd.DataFrame(scaler.fit_transform(df), columns=cols)
        df['SalePrice'] = sale_price
    else:
        cols = ['Id', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        df = df_original[cols]
        Ids = df['Id'].copy()
        imputer = Imputer(strategy='median')
        imputer.fit(df)
        df= pd.DataFrame(imputer.transform(df), columns=cols)
        df['GrLivArea'] = df['GrLivArea'].apply(np.log)
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=cols)
        df['Id'] = Ids


    #GrLivArea
    # sns.distplot(df['GrLivArea'], fit=norm)

    #TotalBsmtSF
    # df['TotalBsmtSF'] = df['TotalBsmtSF'].apply(np.log1p)
    # scaler = StandardScaler()
    # df['TotalBsmtSF'] = scaler.fit_transform(df['TotalBsmtSF'])
    # sns.distplot(df['TotalBsmtSF'], fit=norm)

    # plt.show()
    #min-max transformation
    print(df)
    return df

def split_train_test_data(df):
    #échantillonnage aléatoire
    # X = df.copy()
    # X.drop(['SalePrice'], axis=1)
    # y = df.copy()
    # y = y['SalePrice']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1)

    #échantillonnage stratifié: création des strates puis échantillonnage
    # df["OverallQual"].hist()
    strats = df.copy()
    split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=1)
    for train_index, test_index in split.split(df, strats["OverallQual"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    #comparaison échantillonnage aléatoire vs stratifié
    # repartition_origi=df["OverallQual"].value_counts()/(df["OverallQual"].shape[0])
    # repartition_aleat=X_test["OverallQual"].value_counts()/(X_test["OverallQual"].shape[0])
    # repartition_strat=strat_test_set["OverallQual"].value_counts()/(strat_test_set["OverallQual"].shape[0])
    # repartition=pd.concat([repartition_origi, repartition_aleat, repartition_strat], keys=['ORIGINAL', 'ALEATOIRE', 'STRATIFIE'], axis=1)
    # print(repartition)

    #séparation features/labels
    y_train, y_test = strat_train_set["SalePrice"], strat_test_set["SalePrice"]
    X_train = strat_train_set.drop(["SalePrice"], axis=1)
    #because of outliars
    X_train = X_train.drop([1299, 524])
    y_train = y_train.drop([1299, 524])
    X_test = strat_test_set.drop(["SalePrice"], axis=1)

    return X_train, X_test, y_train, y_test

def random_forest(X_train, y_train):
    print("-------------Random Forest-------------")
    rf = RandomForestRegressor(n_estimators=1000, max_depth=30, max_features=1, n_jobs=-1, random_state=1)

    #optimisation
    # param_grid = [
    #     {'max_depth': [10, 20, 30, 40, 50, 60], 'max_features': [1, 2, 3]}
    # ]
    # grid_search = GridSearchCV(rf, param_grid, cv=10, scoring="neg_mean_squared_error", verbose=5, n_jobs=-1, refit=True)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)
    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #     print(-mean_score, params)


    #cross validation
    # scores = cross_val_score(rf, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
    # scores=-scores
    # print(scores.mean())

    rf.fit(X_train, y_train)
    # rf.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

    return rf

def linear_regression(X_train, y_train):
    print("-------------Linear Regression-------------")
    linear_regression = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)
    # optimisation
    # param_grid = [
    #     {'fit_intercept': [False, True], 'normalize': [False, True]}
    # ]
    # grid_search = GridSearchCV(linear_regression, param_grid, cv=10, scoring="neg_mean_squared_error", verbose=0, n_jobs=-1, refit=True)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)
    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #     print(-mean_score, params)


    #cross validation)
    # scores = cross_val_score(linear_regression, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
    # scores=-scores
    # print(scores.mean())

    linear_regression.fit(X_train, y_train)
    # linear_regression.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

    return linear_regression

def GB_Regressor(X_train, y_train):
    print("-------------Gradient Boosting Regression-------------")
    GB_regressor = GradientBoostingRegressor(learning_rate=0.05, max_depth=10, max_features=1, n_estimators=1000)
    # optimisation
    # param_grid = [
    #     {'learning_rate': [0.001, 0.01, 0.05], 'max_depth': [10, 20, 30]}
    # ]
    # grid_search = GridSearchCV(GB_regressor, param_grid, cv=5, scoring="neg_mean_squared_error", verbose=5, n_jobs=-1, refit=True)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)
    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #     print(-mean_score, params)


    #cross validation
    scores = cross_val_score(GB_regressor, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
    scores=-scores
    print(scores.mean())

    GB_regressor.fit(X_train, y_train)
    # GB_regressor.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

    return GB_regressor

def submission(model):
    #submission
    df_test = data_preparation('../input/test.csv', False)
    Id = df_test['Id']
    df_test = df_test.drop('Id', axis=1)
    # print(df_test.info())
    prediction = pd.DataFrame(model.predict(df_test), columns=['SalePrice'])
    prediction = prediction.apply(np.exp)
    res = pd.concat([Id, prediction], axis=1)
    res.to_csv(path_or_buf="submission",index=False)
    print(res)

def main():
    df = data_preparation('../input/train.csv')
    X_train, X_test, y_train, y_test = split_train_test_data(df)
    # model = linear_regression(X_train, y_train)
    model = random_forest(X_train, y_train)
    # model = GB_Regressor(X_train, y_train)
    submission(model)

main()