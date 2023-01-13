"""
evaluation metrices for regressor and classifier
"""


import numpy as np
import scipy.stats as scistat
import sklearn.metrics as skmts

def get_reg_metrics(df):
    """
    :param df: dataframe with headers=['Response', 'Predicted Response']
    :return mae, mse, rmse, r_square, pcc, spearman:
    """
    mae = skmts.mean_absolute_error(df['Response'], df['Predicted Response'])
    mse = skmts.mean_squared_error(df['Response'], df['Predicted Response'])
    rmse = np.sqrt(mse)
    r_square = skmts.r2_score(df['Response'], df['Predicted Response'])
    pcc, pval = scistat.pearsonr(df['Response'], df['Predicted Response'])
    spearman, spval = scistat.spearmanr(df['Response'], df['Predicted Response'])
    return mae, mse, rmse, r_square, pcc, spearman

def get_clf_metrics(df):
    """
    :param df: dataframe with headers=['Response', 'Predicted Probability']
    :return auc, auprc, accuracy, recall, precision, f1 scores:
    """
    df['Predicted Response'] = df['Predicted Probability'].apply(lambda x: 1 if x > 0.5 else 0)
    fpr, tpr, threshold = skmts.roc_curve(df['Response'], df['Predicted Probability'])
    auc = skmts.auc(fpr, tpr)
    auprc = skmts.average_precision_score(df['Response'], df['Predicted Probability'])
    accuracy = skmts.accuracy_score(df['Response'], df['Predicted Response'])
    recall = skmts.recall_score(df['Response'], df['Predicted Response'])
    precision = skmts.precision_score(df['Response'], df['Predicted Response'], labels=np.unique(df['Predicted Response']))
    f1 = skmts.f1_score(df['Response'], df['Predicted Response'], labels=np.unique(df['Predicted Response']))
    mcc = skmts.matthews_corrcoef(df['Response'], df['Predicted Response'])
    return auc, auprc, accuracy, recall, precision, f1, mcc

