#DATA PREPARATION REPRODUCING: https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    return train,test

def remove_outliers(train):
    train = train[train.GrLivArea < 4000]
    return train

def fill_missing_values(train):
    # Alley : data description says NA means "no alley access"
    train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")
    # BedroomAbvGr : NA most likely means 0
    train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
    # BsmtQual etc : data description says NA for basement features is "no basement"
    train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
    train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
    train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
    train.loc[:, "BsmtFinSF1"] = train.loc[:, "BsmtFinSF1"].fillna(0)
    train.loc[:, "BsmtFinSF2"] = train.loc[:, "BsmtFinSF2"].fillna(0)
    train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
    train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
    train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
    train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
    train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
    # CentralAir : NA most likely means No
    train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
    # Condition : NA most likely means Normal
    train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
    train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
    # EnclosedPorch : NA most likely means no enclosed porch
    train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
    # EnclosedPorch : NA most likely means no enclosed porch
    train.loc[:, "Electrical"] = train.loc[:, "Electrical"].fillna("SBrkr")
    # External stuff : NA most likely means average
    train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
    train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
    train.loc[:, "Exterior1st"] = train.loc[:, "Exterior1st"].fillna("VinylSd")
    train.loc[:, "Exterior2nd"] = train.loc[:, "Exterior2nd"].fillna("VinylSd")
    # Fence : data description says NA means "no fence"
    train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
    # FireplaceQu : data description says NA means "no fireplace"
    train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
    train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
    # Functional : data description says NA means typical
    train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
    # GarageType etc : data description says NA for garage features is "no garage"
    train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
    train.loc[:, "GarageYrBlt"] = train.loc[:, "GarageYrBlt"].fillna(1990) #TODO
    train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
    train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
    train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
    train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
    train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
    # HalfBath : NA most likely means no half baths above grade
    train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
    # HeatingQC : NA most likely means typical
    train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
    # KitchenAbvGr : NA most likely means 0
    train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
    # KitchenQual : NA most likely means typical
    train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
    # LotFrontage : NA most likely means no lot frontage
    train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
    # LotShape : NA most likely means regular
    train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")

    train.loc[:, "MSZoning"] = train.loc[:, "MSZoning"].fillna("RL")
    # MasVnrType : NA most likely means no veneer
    train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
    train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
    # MiscFeature : data description says NA means "no misc feature"
    train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
    train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
    # OpenPorchSF : NA most likely means no open porch
    train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
    # PavedDrive : NA most likely means not paved
    train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
    # PoolQC : data description says NA means "no pool"
    train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
    train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
    # SaleCondition : NA most likely means normal sale
    train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")
    train.loc[:, "SaleType"] = train.loc[:, "SaleType"].fillna("WD")
    # ScreenPorch : NA most likely means no screen porch
    train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
    # TotRmsAbvGrd : NA most likely means 0
    train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)
    train.loc[:, "TotalBsmtSF"] = train.loc[:, "TotalBsmtSF"].fillna(0)
    # Utilities : NA most likely means all public utilities
    train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
    # WoodDeckSF : NA most likely means no wood deck
    train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)
    return train

def some_numerical_features_to_categorical_features(train):
    train = train.replace({
        "MSSubClass" :
        {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
        "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun", 7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
        })
    return train

def some_categorical_features_to_numerical_features(train):
    train = train.replace({"Alley" : {"None": 0, "Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )
    return train


def main():
    train, test = load()
    train = remove_outliers(train)
    train.index = train["Id"]
    test.index = test["Id"]
    complete_set = pd.concat([train, test])
    complete_set = complete_set.drop(["Id"], axis=1)
    complete_set = fill_missing_values(complete_set)
    complete_set = some_numerical_features_to_categorical_features(complete_set)
    complete_set = some_categorical_features_to_numerical_features(complete_set)

    categorical_features = complete_set.select_dtypes(include = ["object"]).columns
    numerical_features = complete_set.select_dtypes(exclude = ["object"]).columns

    complete_set_categorical_features = complete_set[categorical_features]
    dummies = pd.get_dummies(complete_set_categorical_features)

    complete_set_numerical_features = complete_set[numerical_features]
    complete_set = pd.concat([dummies, complete_set_numerical_features], axis=1)

    sale_price = complete_set["SalePrice"]
    complete_set = complete_set.drop(["SalePrice"], axis=1)

    scaler = StandardScaler()
    complete_set = pd.DataFrame(scaler.fit_transform(complete_set), index=complete_set.index, columns=complete_set.columns)

    complete_set = pd.concat([complete_set, sale_price], axis=1)

    train = complete_set.loc[:1460]
    X_train = train.drop(["SalePrice"], axis=1)
    y_train = train["SalePrice"]
    X_test = complete_set.loc[1461:].drop(["SalePrice"], axis=1)

    return X_train, y_train, X_test

# main()