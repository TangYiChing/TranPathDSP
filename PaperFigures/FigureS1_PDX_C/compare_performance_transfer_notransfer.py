"""

"""

import sys
import argparse
import numpy as np
import pandas as pd
import scipy.stats as scistat
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font='Arial', font_scale=1.2)

def parse_parameter():
    parser = argparse.ArgumentParser(description='Compare performance difference in two groups')

    parser.add_argument("-transfer", "--transfer_path",
                        nargs = "+",
                        help = "path to performance file (i.e., .Performance.txt)")
    parser.add_argument("-notransfer", "--notransfer_path",
                        nargs = "+",
                        help = "path to performance file (i.e., ., Performance.txt")
    parser.add_argument("-task", "--task_str",
                        required = True,
                        choices = ["regression", "classification"],
                        help = "string representing job task") 
    parser.add_argument("-feature", "--feature_str",
                        nargs = "+",
                        help = "string representing pathway feature (i.e., PID REACTOME PID_REACTOME")
    parser.add_argument("-o", '--output_path',
                        required = True,
                        help = 'output prefix')
    return parser.parse_args()

if __name__ == "__main__":
    # get args
    args = parse_parameter()

    if len(args.transfer_path) != len(args.notransfer_path):
        print("ERROR! len(transfer_path)={:}!=len(notransfer_path)={:}".format(
        len(args.transfer_path), len(args.notransfer_path)))
        sys.exit()
    else:
        if len(args.feature_str) != len(args.transfer_path):
           print("ERROR! len(feature_str)={:}!=len(transfer_path)={:}".format(
           len(args.feature_str), len(args.transfer_path)))
           sys.exit()
        else:
            print("Comparing performance between transfer and no transfer in order={:}".format(
               args.feature_str))

    # create canvas for figures
    fig, axes = plt.subplots(1, len(args.feature_str), figsize=(10,4), dpi=300, sharey=True)
    for i in range(0, len(args.feature_str)):
        transfer_df = pd.read_csv(args.transfer_path[i], header=0, index_col=0, sep="\t")
        notransfer_df = pd.read_csv(args.notransfer_path[i], header=0, index_col=0, sep="\t")
        feature_str = args.feature_str[i]
        if args.task_str == "classification":
            transfer_arr = transfer_df['AUC'].values
            notransfer_arr = notransfer_df['AUC'].values
        else:
            transfer_arr = transfer_df['RMSE'].values
            notransfer_arr = notransfer_df['RMSE'].values
        # wilcoxon signed-rank test
        w, p = scistat.ranksums(transfer_arr, notransfer_arr) #wilcoxon(transfer_arr, notransfer_arr)
        print("Feature={:} | Transfer vs. Notransfer={:.1f} (pvalue={:})".format(feature_str, w,p))
        
        # plotting
        if args.task_str == "classification":
            # prepare df
            t_df = transfer_df[['AUC']].copy()
            t_df.loc[:, feature_str] = 'Transfer'
            n_df = notransfer_df[['AUC']].copy()
            n_df.loc[:, feature_str] = 'No transfer'
            df = pd.concat([n_df, t_df], axis=0)
            df.columns = ["AUROC", feature_str]
            # boxplot
            sns.boxplot(y="AUROC", x=feature_str, data=df, orient="v", ax=axes[i], palette="colorblind",
                        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white",
                                                   "markeredgecolor":"black", "markersize":"10"})
            axes[i].set_title(args.output_path, fontsize=18)
            #axes[i].set_title("Wilcoxon test p-value={:.3f}".format(p), fontsize=12)
            # barplot
            #sns.barplot(x=feature_str, y="AUROC", data=df, estimator=np.mean, ci=95, ax=axes[i])

        else:
            t_df = transfer_df[['RMSE']].copy()
            t_df.loc[:, feature_str] = 'Transfer'
            n_df = notransfer_df[['RMSE']].copy()
            n_df.loc[:, feature_str] = 'No transfer'
            df = pd.concat([n_df, t_df], axis=0)
            # boxplot
            sns.boxplot(y="RMSE", x=feature_str, data=df, orient="v", ax=axes[i], palette="colorblind",
                        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", 
                                                   "markeredgecolor":"black", "markersize":"10"})
            axes[i].set_title(args.output_path, fontsize=18)
            #axes[i].set_title("Wilcoxon test p-value={:}".format(p), fontsize=12)
            # barplot
            #sns.barplot(x=feature_str, y="RMSE", data=df, estimator=np.mean, ci=95, ax=axes[i])

    # save to file
    fig.savefig(args.output_path+".ComparePerformance.TransferNotransfer.BOXPLOT.png", bbox_inches="tight")
    print("find boxplot at {:}".format(args.output_path+".ComparePerformance.TransferNotransfer.BOXPLOT.png"))
    #fig.savefig(args.output_path+".ComparePerformance.TransferNotransfer.BARPLOT.png", bbox_inches="tight")
    #print("find barplot at {:}".format(args.output_path+".ComparePerformance.TransferNotransfer.BARPLOT.png"))
