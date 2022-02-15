"""
Return data of CHEM-DGNet-EXP
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre
import sklearn.model_selection as skms
import sklearn.metrics as skmts
import sklearn.utils as skut

def parse_parameter():
    parser = argparse.ArgumentParser(description='concatenating features of CHEM-DGNet-EXP')
    parser.add_argument("-chem", "--chem_path",
                        required = True,
                        help = "path to CHEM feature file, e.g., CHEM.256.MBits.txt")
    parser.add_argument("-dgnet", "--dgnet_path",
                        required = True,
                        help = "path to DGNet feature file, e.g., DGNet.NetPEA.txt")
    parser.add_argument("-exp", "--exp_path",
                        required = True,
                        help = "path to EXP feature file, e.g., EXP.ssGSEA.txt")
    parser.add_argument("-sample", "--sample_path",
                        nargs = '*',
                        required = True,
                        help = "path to sample annotation file, can be multiple files: resp1.txt resp2.txt")
    parser.add_argument("-o", "--output_path",
                        required = True,
                        help = "path to ouput file")
    return parser.parse_args()

def merge_df(f_list):
    """
    :param f_list: list of file strings
    :param df: dataframe
    """
    # loop through input annotation to merge into one dataframe
    df_list = []
    for fin in f_list:
        df = pd.read_csv(fin, header=0, sep="\t")
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    return df

def merge_feature(feature_dict, sample_df):
    """
    :param feature_dict: dictionary with keys=[CHEM, DGNet, EXP]
    :param sample_df: dataframe with headers=[Alias, Sample, Response]
    :return df: merged feature data
    :return shared_df: sample annotation dataframe
    [Note]
    Alias is a copy of Therapy replaced with drugs' frequently used synonym
    """
    # retrieve drug list
    drug_list = [ set(feature_dict['CHEM'].index), set(feature_dict['DGNet'].index), set(sample_df['Alias'].values) ]
    common_drug_list = set.intersection(*drug_list)
    # retrieve sample list
    sample_list = [ set(feature_dict['EXP'].index), set(sample_df['Sample'].values) ]
    common_sample_list = set.intersection(*sample_list)

    # subsetting to include common drugs and common samples
    shared_df = sample_df[ sample_df['Alias'].isin(common_drug_list) ]
    shared_df = shared_df[ shared_df['Sample'].isin(common_sample_list) ]
    #print(shared_df)
    if len(shared_df) == 0:
        print("ERROR! No shared samples")
        sys.exit(1)
    else:
        # retrieve features
        chem_df = feature_dict['CHEM'].loc[shared_df['Alias'].values.tolist()]
        dg_df = feature_dict['DGNet'].loc[shared_df['Alias'].values.tolist()]
        exp_df = feature_dict['EXP'].loc[shared_df['Sample'].values.tolist()]
        chem_df.index.name = 'Therapy'
        dg_df.index.name = 'drug'
        exp_df.index.name = 'Sample'

        # concatenate by adding features to columns in the order of CHEM-DGNet-EXP
        cdf = pd.concat([chem_df.reset_index(), dg_df.reset_index(), exp_df.reset_index()], axis=1)

        # add Response to the last column
        resp_df = shared_df[['Alias','Sample','Response']]
        resp_df.columns = ['Therapy','Sample','Response']
        resp_df = resp_df.set_index(['Therapy','Sample'])
        cdf = cdf.set_index(['Therapy','Sample'])
        df = pd.concat([cdf, resp_df[['Response']]], axis=1)
        df = df.drop(columns=['drug'], axis=1)
        #print(df)
        # sanity check
        n_chem = chem_df.shape[1]
        n_dg = dg_df.shape[1]
        n_exp = exp_df.shape[1]
        if df.shape[1]-1 != n_chem + n_dg + n_exp:
            print("ERROR! total features={:} but got {:} after concatenation".format(
                  n_chem + n_dg + n_exp, df.shape[1]-1))
        n_na = df.isnull().sum().sum()
        if n_na > 0:
            print("ERROR! after concatenation, #missing={:}".format(n_na))
            sys.exit(1)

        # return
        print('CHEM={:} | DGNet={:} | EXP={:} | total={:}'.format(
               n_chem, n_dg, n_exp, df.shape))
        return df, shared_df # index=['Therapy','Sample']

if __name__ == "__main__":
    np.random.seed(42)
    # get args
    args = parse_parameter()

    # get feature files
    chem_df = pd.read_csv(args.chem_path, header=0, index_col=0, sep="\t")
    dg_df = pd.read_csv(args.dgnet_path, header=0, index_col=0, sep="\t")
    exp_df = pd.read_csv(args.exp_path, header=0, index_col=0, sep="\t")
    # remove rows with nas
    chem_df = chem_df.dropna(how='all', axis=0)
    dg_df = dg_df.dropna(how='all', axis=0)
    exp_df = exp_df.dropna(how='all', axis=0)
    # modify column name
    chem_df.columns = ['CHEM_'+col for col in chem_df.columns]
    dg_df.columns = ['DGNet_'+col for col in dg_df.columns]
    exp_df.columns = ['EXP_'+col for col in exp_df.columns]
    feature_dict = {'CHEM':chem_df, 'DGNet':dg_df, 'EXP':exp_df}
    for d, df in feature_dict.items():
        n_na = df.isnull().sum().sum()
        if n_na > 0:
            print("ERROR! {:} has missing values={:}".format(
                  d, n_na))
            sys.exit(1)

    # get samples
    if len(args.sample_path) > 1:
        sample_df = merge_df(args.sample_path)
    else:
        sample_df = pd.read_csv(args.sample_path[0], header=0, sep="\t")

    # concat features CHEM-DGNet-EXP
    feature_df, sample_df = merge_feature(feature_dict, sample_df)

    # save to file
    feature_df.to_pickle(args.output_path+'.CHEM-DGNet-EXP.pkl')
    sample_df.to_csv(args.output_path+'.SampleInfo.txt',header=True, index=False, sep="\t")
    print('find merged Feature Data at {:}'.format(args.output_path+'.CHEM-DGNet-EXP.pkl'))
    print('find merged Sample Info at {:}'.format(args.output_path+'.SampleInfo.txt'))

    # summary stats report
    sample_size = sample_df.shape[0]
    n_drugs = len(sample_df['Alias'].unique())
    n_cancers = len(sample_df['Cancer types'].unique())
    resp_list = list(sample_df['Response'].unique())
    if 'R' in resp_list or 'NR' in resp_list:
        n_R = len(sample_df[sample_df['Response']=='R'])
        n_NR = len(sample_df[sample_df['Response']=='NR'])
        ratio_str = str(n_R)+':'+str(n_NR)
    else:
        ratio_str = 'NA'
    stat_df = pd.DataFrame({'Sample size':[sample_size],
                       'Number of drugs':[n_drugs],
                       'Number of cancer types':[n_cancers],
                       'R:NR':[ratio_str]})
    stat_df.to_csv(args.output_path+".DescriptiveStat.txt", header=True, index=False, sep="\t")
    print(stat_df)
