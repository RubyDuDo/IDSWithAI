import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import skew
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

import algorithm as al


def read_data( data_type="train" ):
    if data_type == "train" :
        unsw = pd.read_csv("data/unsw_nb15/UNSW_NB15_training-set.csv", delimiter=",",low_memory=False)
    else :
        unsw = pd.read_csv("data/unsw_nb15/UNSW_NB15_testing-set.csv", delimiter=",",low_memory=False)

    feature_del( unsw )
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

def feature_onehot( train, test ):
    train = train.get_dummies( train, columns=['proto',"service","state"]);
    test = test.get_dummies( train, columns=['proto',"service","state"]);

# simple feature deal 
def feature_simple( train, test ):
    feature_factor(train, test)
    train, test = feature_standard( train, test )
    return train, test

def feature_current( train, test ):
    train, test = feature_log( train, test )
    print(train)
    train, processor  = feature_column_deal( train)
    test, processor = feature_column_deal( test, processor )
    return train, test

def convert_back_to_PD( df_transformed, preprocessor ):
    # 转换回DataFrame（如果需要查看转换后的DataFrame）
    # 获取One-Hot编码后的特征名
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
    num_feature_names = preprocessor.named_transformers_['num'].get_feature_names_out()

    # 合并所有特征名
    all_feature_names = num_feature_names.tolist() + ohe_feature_names.tolist()

    print( df_transformed.shape)
    print( len(all_feature_names))
    
    print( "all_feature_names", len(all_feature_names) )
    df_new = pd.DataFrame(data=df_transformed, columns=all_feature_names)
    print(df_new)
    return df_new


# lable(one hot) + num(standardscaler)
def feature_column_deal( df , preprocessor = None ):
    if( preprocessor == None ):
        numeric_features = df.select_dtypes( np.number ).columns
        categorical_features = df.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features ),  # 标准化数值数据
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features ) # 对类别数据进行One-Hot编码
            ],
            sparse_threshold=0
        )

        # preprocessor
        df_transformed = preprocessor.fit_transform(df)
    else:
        df_transformed = preprocessor.transform(df)

    

    new_df = convert_back_to_PD( df_transformed, preprocessor )
    return new_df, preprocessor


def is_skew( df, col ):
    skewness = skew( df[col] )
    SKEW_LIMIT = 4
    if skewness > SKEW_LIMIT :
        return True
    
def feature_log( train, test ):
    skew_features = []
    numeric_features = train.select_dtypes( np.number ).columns
    for col in numeric_features:
        if is_skew( train, col ):
            skew_features.append( col )

    # 对指定列进行对数操作（原地修改）
    train[skew_features] = np.log(train[skew_features] + 1  )
    test[skew_features] = np.log(test[skew_features] + 1 )

    return train, test 


def feature_reduce_desiciontree( X_train, Y_train, X_test, Y_test ):
    clf = DecisionTreeClassifier(random_state=42)
    al.train_and_evaluate(clf, X_train, Y_train, X_test, Y_test )

    #通过决策树，观察哪些字段对决策的重要性更大
    # 获取特征重要性并创建DataFrame
    feature_importances = clf.feature_importances_
    feature_names = X_train.columns
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # 按重要性排序
    features_df = features_df.sort_values(by='Importance', ascending=False)

    # 通过决策树，找到没啥影响的列，删掉他们
    features_with_zero_importance = [feature_names[i] for i, importance in enumerate(feature_importances) if importance == 0]
    print( features_with_zero_importance)

    X_train.drop( features_with_zero_importance, axis = 1, inplace = True )
    X_test.drop( features_with_zero_importance, axis = 1, inplace = True )

    print(X_train.shape)
    print(X_test.shape)
