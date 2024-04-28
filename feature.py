import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import StandardScaler

def read_data( data_type="train" ):
    if data_type == "train" :
        unsw = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv", delimiter=",",low_memory=False)
    else :
        unsw = pd.read_csv("data/unsw_nb15/UNSW_NB15_testing-set.csv", delimiter=",",low_memory=False)

    return unsw

def __factor( train ,  test, col_name ):
    labels, unique = pd.factorize( train[col_name])
    train[col_name] = labels

    test_labels = pd.Series([unique.tolist().index(i) if i in unique else -1 for i in test[col_name]])
    test[col_name] = test_labels
    return labels, unique

# drop useless features
def feature_del(dataset):
    dataset.drop(["attack_cat","id"], axis = 1, inplace = True)

# simply change string feature to numbers, using label
def feature_factor( train,  test ):
    __factor( train, test, "proto" )
    __factor( train, test, "service")
    __factor( train, test, "state")

def feature_standard( train, test ):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform( test )
    return train, test

# simple feature deal 
def feature_simple( train, test ):
    feature_del( train )
    feature_del( test )
    feature_factor(train, test)
    train, test = feature_standard( train, test )
    return train, test


