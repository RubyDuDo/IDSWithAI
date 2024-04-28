import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

def read_data( data_type="train" ):
    if data_type == "train" :
        unsw = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv", delimiter=",",low_memory=False)
    else :
        unsw = pd.read_csv("data/unsw_nb15/UNSW_NB15_testing-set.csv", delimiter=",",low_memory=False)

    return unsw

def __factor( dataset, col_name ):
    labels, unique = pd.factorize( dataset[col_name])
    dataset[col_name] = labels
    return labels, unique

# drop useless features
def feature_del(dataset):
    dataset.drop(["attack_cat","id"], axis = 1, inplace = True)

# simply change string feature to numbers, using label
def feature_factor( dataset ):
    __factor( dataset, "proto" )
    __factor( dataset, "service")
    __factor( dataset, "state")

# simple feature deal 
def feature_simple( dataset ):
    feature_del( dataset )
    feature_factor(dataset)