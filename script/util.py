"""
all sort of utilities

"""


import sys
import numpy as np
import pandas as pd
from datetime import datetime
import sklearn.preprocessing as skpre

def cal_time(end, start):
    """return time spent"""
    # end = datetime.now(), start = datetime.now()
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    spend = datetime.strptime(str(end), datetimeFormat) - \
            datetime.strptime(str(start),datetimeFormat)
    return spend

def subsetting_feature_column(data_df, feature_str="PID_REACTOME"):
    """
    :param data_df: dataframe that columns are strings startswith PID, REACTOME
    :param feature_str: string representing pathway features. option=[PID, REACTOME, PID_REACTOME]
    :param df: dataframe with wanted pathway features at columns
    """
    # sanity check for inputs
    if feature_str not in ['PID', 'REACTOME', 'PID_REACTOME']:
        print('ERROR! feature_str={:} not supported'.format(feature_str))
        sys.exit(1)
    else:
        # extract columns
        feature_list = data_df.columns.tolist()
        
    chem_col_list = [col for col in feature_list if col.startswith("CHEM")]
    # search DGNet and EXP as they are pathway features
    if feature_str == 'PID':
            dg_col_list = [col for col in feature_list if col.startswith("DGNet_PID")]
            exp_col_list = [col for col in feature_list if col.startswith("EXP_PID")]
    elif feature_str == 'REACTOME':
            dg_col_list = [col for col in feature_list if col.startswith("DGNet_REACTOME")]
            exp_col_list = [col for col in feature_list if col.startswith("EXP_REACTOME")]
    elif feature_str == 'PID_Gene':
            dg_col_list = [col for col in feature_list if col.startswith("DGNet_PID")]
            exp_col_list = [col for col in feature_list if col.startswith("EXP")]
    elif feature_str == 'REACTOME_Gene':
            dg_col_list = [col for col in feature_list if col.startswith("DGNet_REACTOME")]
            exp_col_list = [col for col in feature_list if col.startswith("EXP")]
    else:
        dg_col_list = [col for col in feature_list if col.startswith("DGNet")]
        exp_col_list = [col for col in feature_list if col.startswith("EXP")]
    use_col_list = chem_col_list+dg_col_list+exp_col_list+['Response'] 


    if len(use_col_list) == 0:
        print('ERROR! feature_str={:} not found!'.format(feature_str))
        sys.exit(1)
    else:
        df = data_df[use_col_list]
    # return
    return df

def normalize_data(train_df, test_df, method="standard", feature_list=[]):
    """
    :param train_df: train set, assume the last column is label
    :param test_df: test set, assume the last column is lable
    :param method: string representing normalization approach. option=[standard, minmax]
    :param feature_list: list of prefix string in column name
    :return scaled_train_df: dataframe of train data
    :return scaled_test_df: dataframe of test data
    """
    # split X, y
    train_X_df = train_df.iloc[:, :-1]
    test_X_df = test_df.iloc[:, :-1]
    if method == 'standard':
        if len(feature_list) == 0:
            # scale all train at once
            scaler = skpre.StandardScaler()
            train_X_arr = scaler.fit_transform(train_X_df)
            test_X_arr = scaler.transform(test_X_df)
            # arr2df
            scaled_train_X_df = pd.DataFrame(train_X_arr, index=train_X_df.index, columns=train_X_df.columns)
            scaled_test_X_df = pd.DataFrame(test_X_arr, index=test_X_df.index, columns=test_X_df.columns)
        else:
            # create record list
            train_X_df_list = []
            test_X_df_list = []
            # scale feature by feature
            for feature in feature_list:
                # subsetting columns
                col_list = [col for col in train_df.columns if col.startswith(feature)]
                f_train_X_df = train_X_df[col_list]
                f_test_X_df = test_X_df[col_list]
                # scaling
                scaler = skpre.StandardScaler()
                f_train_X_arr = scaler.fit_transform(f_train_X_df)
                f_test_X_arr = scaler.transform(f_test_X_df)
                # arr2df
                scaled_f_train_X_df = pd.DataFrame(f_train_X_arr, index=f_train_X_df.index, columns=col_list)
                scaled_f_test_X_df = pd.DataFrame(f_test_X_arr, index=f_test_X_df.index, columns=col_list)
                # append to list
                train_X_df_list.append(scaled_f_train_X_df)
                test_X_df_list.append(scaled_f_test_X_df)
            # merge
            scaled_train_X_df = pd.concat(train_X_df_list,axis=1)
            scaled_test_X_df = pd.concat(test_X_df_list,axis=1)
    elif method == 'minmax':
        if len(feature_list) == 0:
            # scale all train at once
            scaler = skpre.MinMaxScaler()
            train_X_arr = scaler.fit_transform(train_X_df)
            test_X_arr = scaler.transform(test_X_df)
            # arr2df
            scaled_train_X_df = pd.DataFrame(train_X_arr, index=train_X_df.index, columns=train_X_df.columns)
            scaled_test_X_df = pd.DataFrame(test_X_arr, index=test_X_df.index, columns=test_X_df.columns)
        else:
            # create record list
            train_X_df_list = []
            test_X_df_list = []
            # scale feature by feature
            for feature in feature_list:
                # subsetting columns
                col_list = [col for col in train_df.columns if col.startswith(feature)]
                f_train_X_df = train_X_df[col_list]
                f_test_X_df = test_X_df[col_list]
                # scaling
                scaler = skpre.MinMaxScaler()
                f_train_X_arr = scaler.fit_transform(f_train_X_df)
                f_test_X_arr = scaler.transform(f_test_X_df)
                # arr2df
                scaled_f_train_X_df = pd.DataFrame(f_train_X_arr, index=f_train_X_df.index, columns=col_list)
                scaled_f_test_X_df = pd.DataFrame(f_test_X_arr, index=f_test_X_df.index, columns=col_list)
                # append to list
                train_X_df_list.append(scaled_f_train_X_df)
                test_X_df_list.append(scaled_f_test_X_df)
            # merge
            scaled_train_X_df = pd.concat(train_X_df_list,axis=1)
            scaled_test_X_df = pd.concat(test_X_df_list,axis=1)
    else:
        print("ERROR! normalization method={:} not supported.".format(method))
        sys.exit(1)

    # merge with y
    scaled_train_df = pd.concat([scaled_train_X_df, train_df.iloc[:, [-1]]], axis=1)
    scaled_test_df = pd.concat([scaled_test_X_df, test_df.iloc[:, [-1]]], axis=1)

    # return
    return scaled_train_df, scaled_test_df
