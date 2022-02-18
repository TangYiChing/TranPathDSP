"""
Compare performance between GDSC drug/cancer vs. Non-GDSC drug/cancer
"""


import sys
import glob
import argparse
import numpy as np
import pandas as pd
import scipy.stats as scistat
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager
sns.set_theme(font_scale=1.6,font='Arial')

import metrices as mts

def parse_parameter():
    parser = argparse.ArgumentParser(description='Compare performance difference in two groups')

    parser.add_argument("-cell", "--cell_path",
                        help = "path to GDSC sample information file (i.e., .SampleInfo.txt)")
    parser.add_argument("-transfer", "--transfer_path",
                        help = "path to prediction file (i.e., ., GroupMean.txt")
    parser.add_argument("-notransfer", "--notransfer_path",
                        help = "path to prediction file (i.e., ., GroupMean.txt")
    parser.add_argument("-task", "--task_str",
                        required = True,
                        choices = ["regression", "classification"],
                        help = "string representing job task")
    parser.add_argument("-groupby", "--groupby_str",
                        choices = ["Drug", "Cancer"],
                        help = "string representing per-Drug level or per-Cancer level.")
    parser.add_argument("-data", "--data_str",
                        help = "string representing Catplot title")
    parser.add_argument("-o", '--output_path',
                        required = True,
                        help = 'output prefix')
    return parser.parse_args()

if __name__ == "__main__":
    # get args
    args = parse_parameter()

    # get gdsc list
    cell_df = pd.read_csv(args.cell_path, header=0, sep="\t")
    cell_df = cell_df.dropna(how='any', axis=0)
    if args.groupby_str == "Drug":
        gdsc_list = sorted(list(cell_df['Alias'].unique()))
    else:
        gdsc_list = sorted(list(cell_df['Cancer types'].unique()))

    # get performance 
    transfer_df = pd.read_csv(args.transfer_path, header=0, index_col=0, sep="\t")
    notransfer_df = pd.read_csv(args.notransfer_path, header=0, index_col=0, sep="\t")
    # exclude sample size < 5
    transfer_df = transfer_df[transfer_df['Sample size']>5]
    notransfer_df = notransfer_df[notransfer_df['Sample size']>5]
    
    # merge
    if args.task_str == 'classification':
        df = pd.concat([notransfer_df[['AUC']], transfer_df[['AUC']]], axis=1)
        df.columns = ['No transfer', 'Transfer']
    else:
        df = pd.concat([notransfer_df[['RMSE']], transfer_df[['RMSE']]], axis=1)
        df.columns = ['No transfer', 'Transfer']
    print(df)

    # remove NAN
    df = df.dropna(how='any', axis=0)

    # add columns
    df['%change'] = (df['Transfer'] - df['No transfer']) / df['No transfer'] * 100
    df = df.reset_index()

    if args.groupby_str == "Drug":
        df['GDSC Drug'] = df['Name'].apply(lambda x: 'Yes' if x in gdsc_list else 'No') 
        df = df.sort_values(by=['GDSC Drug'])
    else:
        df['GDSC Cancer'] = df['Name'].apply(lambda x: 'Yes' if x in gdsc_list else 'No')
        df = df.sort_values(by=['GDSC Cancer'])
    #print(df)

    # save to file
    df.to_csv(args.output_path+".Performance."+args.groupby_str+".GDSCvs.nonGDSC.txt", header=True, index=False, sep="\t")
    print('find file at {:}'.format(args.output_path+".Performance."+args.groupby_str+".GDSCvs.nonGDSC.txt"))

    # prepare data for catplot
    if args.groupby_str == "Drug":
        ntl_df = df[["Name", "No transfer", "GDSC Drug"]].copy()
        tl_df = df[["Name", "Transfer", "GDSC Drug"]].copy()
        ntl_df.loc[:, 'Method'] = "No transfer"
        tl_df.loc[:, "Method"] = "Transfer"
        if args.task_str == "classification":
            ntl_df.columns = ["Name", "AUROC", "GDSC Drug", "Method"]
            tl_df.columns = ["Name", "AUROC", "GDSC Drug", "Method"]
            df = pd.concat([ntl_df, tl_df], axis=0)
            # plot catplot
            #fig, ax = plt.subplots(1, figsize=(10,4), dpi=300)
            fig = sns.catplot(x="Method", y="AUROC", data=df, hue="Method", palette="colorblind", 
                        col="GDSC Drug", kind="bar", aspect=.7, dodge=False)
            fig.fig.subplots_adjust(top=.8)
            fig.fig.suptitle(args.data_str, fontsize=18) 
            fig.savefig(args.output_path+".Drug.CATPLOT.png", bbox_inches="tight", dpi=300)
        else:
            ntl_df.columns = ["Name", "RMSE", "GDSC Drug", "Method"]
            tl_df.columns = ["Name", "RMSE", "GDSC Drug", "Method"]
            df = pd.concat([ntl_df, tl_df], axis=0)
            # plot catplot
            #fig, ax = plt.subplots(1, figsize=(10,4), dpi=300)
            fig = sns.catplot(x="Method", y="RMSE", data=df, hue="Method", palette="colorblind",
                        col="GDSC Drug", kind="bar", aspect=.7, dodge=False)
            fig.fig.subplots_adjust(top=.8)
            fig.fig.suptitle(args.data_str, fontsize=18)
            fig.savefig(args.output_path+".Drug.CATPLOT.png", bbox_inches="tight", dpi=300)

    else:
        ntl_df = df[["Name", "No transfer", "GDSC Cancer"]].copy()
        tl_df = df[["Name", "Transfer", "GDSC Cancer"]].copy()
        ntl_df.loc[:, 'Method'] = "No transfer"
        tl_df.loc[:, "Method"] = "Transfer"
        if args.task_str == "classification":
            ntl_df.columns = ["Name", "AUROC", "GDSC Cancer", "Method"]
            tl_df.columns = ["Name", "AUROC", "GDSC Cancer", "Method"]
            df = pd.concat([ntl_df, tl_df], axis=0)
            # plot catplot
            #fig, ax = plt.subplots(1, figsize=(10,4), dpi=300)
            fig = sns.catplot(x="Method", y="AUROC", data=df, hue="Method", palette="colorblind",
                        col="GDSC Cancer", kind="bar", aspect=.7, dodge=False)
            fig.fig.subplots_adjust(top=.8)
            fig.fig.suptitle(args.data_str, fontsize=18)
            fig.savefig(args.output_path+".Cancer.CATPLOT.png", bbox_inches="tight", dpi=300)
        else:
            ntl_df.columns = ["Name", "RMSE", "GDSC Cancer", "Method"]
            tl_df.columns = ["Name", "RMSE", "GDSC Cancer", "Method"]
            df = pd.concat([ntl_df, tl_df], axis=0)
            # plot catplot
            #fig, axes = plt.subplots(1, figsize=(10,4), dpi=300)
            fig = sns.catplot(x="Method", y="RMSE", data=df, hue="Method", palette="colorblind",
                        col="GDSC Cancer", kind="bar", aspect=.7, dodge=False)
            fig.fig.subplots_adjust(top=.8)
            fig.fig.suptitle(args.data_str, fontsize=18)
            fig.savefig(args.output_path+".Cancer.CATPLOT.png", bbox_inches="tight", dpi=300)