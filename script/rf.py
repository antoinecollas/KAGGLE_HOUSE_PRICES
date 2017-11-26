import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def load_data_set(path):
    housing = pd.read_csv(path, ",")
    return housing

def remove_outliers(training_set):
    # print('remove_outliers')
    # print(training_set.loc[training_set['GrLivArea'] >= 4600])
    training_set.drop([523, 1298], inplace=True)


def select_columns(data_set, features, label=[]):
    data_set = data_set[features+label]

def add_features(data_set, features=[]):
    features = features + ['TotalSF']
    data_set['TotalSF'] = data_set['GrLivArea'] + data_set['TotalBsmtSF']
    return features

def mat_corr(training_set):
    mat_corr = training_set.corr()
    mat_corr = mat_corr[['SalePrice', 'TotalSF']]
    print(mat_corr)
    ax = sns.heatmap(mat_corr)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    plt.show()

def optimize_hyperparameters_RF(model, X, y):
    param_grid = {
        "max_depth": [10, 50, 100],
        "max_features": [5, X.shape[1]],
        "min_samples_split": [10, 50, 100],
        "min_samples_leaf": [1, 10],
        "bootstrap": [True, False],
    }
    grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(mean_squared_log_error, greater_is_better=False), n_jobs=-1, cv=5, verbose=5)
    grid_search.fit(X, y)
    print(grid_search.best_params_)

def score(model, X, y):
    score = cross_val_score(model, X, y, scoring=make_scorer(mean_squared_log_error, greater_is_better=False), cv=5, n_jobs=-1)
    score = score.mean()
    score = np.sqrt(-score)
    return score

def fill_nan_test_set(test_set):
    test_set["GarageCars"].fillna(test_set["GarageCars"].mean(), inplace=True)
    test_set["TotalBsmtSF"].fillna(test_set["TotalBsmtSF"].mean(), inplace=True)

def submission(model, features):
    test_set = load_data_set('../input/test.csv')
    Id = test_set['Id']
    fill_nan_test_set(test_set)
    add_features(test_set)
    test_set = test_set[features]
    y_pred = model.predict(test_set)
    # print(df_test.info())
    prediction = pd.DataFrame(y_pred, columns=['SalePrice'])
    res = pd.concat([Id, prediction], axis=1)
    res.to_csv(path_or_buf="submission",index=False)

def main():
    training_set = load_data_set("../input/train.csv")
    features= ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'BedroomAbvGr']
    label= ['SalePrice']

    remove_outliers(training_set)
    select_columns(training_set, features, label)
    features = add_features(training_set, features)

    X_train = training_set[features]
    y_train = training_set[label]

    random_forest = RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True, max_depth=10, max_features=5, min_samples_leaf=1, min_samples_split=10)
    # optimize_hyperparameters_RF(random_forest, X_train, y_train.values.ravel())

    print(score(random_forest, X_train, y_train.values.ravel()))
    random_forest.fit(X_train, y_train.values.ravel())
    submission(random_forest, features)


main()