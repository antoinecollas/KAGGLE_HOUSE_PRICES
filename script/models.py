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
import data_preparation as prep
from sklearn.model_selection import train_test_split

def optimize_hyperparameters_RF(model, X, y):
    param_grid = {
        "max_depth": [50, 70, 100],
        "max_features": [5, 100, X.shape[1]],
        "min_samples_split": [1, 10, 100],
        "min_samples_leaf": [1, 5, 10],
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


def plot_learning_curves(model, X, y):
    print("======plot_learning_curves======")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        print(str(m)+"/"+str(len(X_train)))
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(np.sqrt(mean_squared_log_error(y_train_predict, y_train[:m])))
        val_errors.append(np.sqrt(mean_squared_log_error(y_val_predict, y_val)))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.show()

# def submission(model, features):
#     test_set = load_data_set('../input/test.csv')
#     Id = test_set['Id']
#     fill_nan_test_set(test_set)
#     add_features(test_set)
#     test_set = test_set[features]
#     y_pred = model.predict(test_set)
#     # print(df_test.info())
#     prediction = pd.DataFrame(y_pred, columns=['SalePrice'])
#     res = pd.concat([Id, prediction], axis=1)
#     res.to_csv(path_or_buf="submission",index=False)

def main():
    X_train, y_train, X_test = prep.main()

    random_forest = RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True, max_depth=50, max_features=254, min_samples_leaf=1, min_samples_split=10)
    # optimize_hyperparameters_RF(random_forest, X_train, y_train.values.ravel())
    print(score(random_forest, X_train, y_train.values.ravel())) #0.146
    plot_learning_curves(random_forest, X_train, y_train)

    # random_forest.fit(X_train, y_train.values.ravel())
    # submission(random_forest, features)


main()