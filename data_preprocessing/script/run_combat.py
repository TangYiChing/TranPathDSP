"""
Return homogenized expression data
"""
# built-in pkgs
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA

# customized pkgs
from combat.pycombat import pycombat


def plot_pca(before_df, after_df, batch_df, fig_str):
    """
    return side by side PCA figure
    """
    # set up figure canvas
    fig, axes = plt.subplots(1,2, figsize=(18,6), dpi=200)


    # sanitycheck on sample
    sample_list = [set(before_df.index),set(after_df.index),set(batch_df['Sample'].values)]
    common_sample_list = set.intersection(*sample_list)
    before_df = before_df.loc[common_sample_list]
    after_df = after_df.loc[common_sample_list]
    batch_df = batch_df[batch_df['Sample'].isin(common_sample_list)]

    # add batch_id column to file
    legend_str = 'Datasets' # options = [batch_id, Cancer types]
    sample_batchid_dict = dict(zip(batch_df["Sample"], batch_df[legend_str]))
    before_df['batch_id'] = [sample_batchid_dict[sample] for sample in before_df.index]
    after_df['batch_id'] = [sample_batchid_dict[sample] for sample in after_df.index]
   

    # defind colors for batches
    pal = sns.color_palette("colorblind", len(batch_df['batch_id'].unique()))

    # plot PCA
    df_list = [before_df, after_df]
    for i in range(0, len(df_list)):

        # get data
        df = df_list[i]
        X_cols = [col for col in df.columns if col != "batch_id"]
        y_cols = [col for col in df.columns if col == "batch_id"]
        X = df[X_cols].values

        # PCA transform
        pca = PCA(n_components=2)
        pca_arr = pca.fit_transform(X)
        pca_df = pd.DataFrame(pca_arr, columns=['PCA1', 'PCA2'], index=df_list[i].index)
        pca_df = pd.concat([pca_df, df[['batch_id']]], axis=1)
        print(pca_df)
        # plot        
        sns.scatterplot(pca_df['PCA1'], pca_df['PCA2'], data=pca_df, ax=axes[i],
                              hue='batch_id', legend='full', palette=pal, s=50)

        # set labels
        axes[i].set_xlabel('PC1', fontsize=14)
        axes[i].set_ylabel('PC2', fontsize=14)

    axes[0].set_title('Before batch correction', fontsize=12)
    axes[1].set_title('After batch correction', fontsize=12)
    # save to file
    fout_str = fig_str + '.PCA.png'
    plt.savefig(fout_str, dpi=200, bbox_inches='tight')
    return fig

def parse_parameter():
    parser = argparse.ArgumentParser(description='Batch correction gene expression data')

    parser.add_argument("-exp", "--exp_path",
                        nargs = '*',
                        required = True,
                        help = "path to expression file, i.e.,  sample by gene. can be multiple files: exp1.txt exp2.txt")
    parser.add_argument("-anno", "--anno_path",
                        nargs = '*',
                        required = True,
                        help = "path to sample annotation file, can be multiple files: exp1.txt exp2.txt")
    parser.add_argument("-o", "--output_path",
                        required = True,
                        help = "path to ouput file")
    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(42)
    # get args
    args = parse_parameter()

    # loop through input data list to get expression data
    gene_list = [] # list of gene set
    df_list = []
    for fin in args.exp_path:
        df = pd.read_csv(fin, header=0, index_col=0, sep="\t")
        gene_list.append( set(df.columns) )
        df_list.append(df)
    # merge gene expression data by selecting the same genes
    common_gene_list = set.intersection(*gene_list)
    exp_df_list = []
    for df in df_list:
        df = df[common_gene_list]
        exp_df_list.append(df)
    exp_df = pd.concat(exp_df_list, axis=0)
        
    # loop through input annotation to merge
    anno_df_list = []
    for fin in args.anno_path:
        df = pd.read_csv(fin, header=0, sep="\t")
        anno_df_list.append(df)
    anno_df = pd.concat(anno_df_list, axis=0)
    
    # subsetting to include same samples set
    common_sample_list = sorted(list(set(exp_df.index)&set(anno_df['Sample'])))
    exp_df = exp_df.loc[common_sample_list]
    anno_df = anno_df[anno_df['Sample'].isin(common_sample_list)]
    
    # generate batches by datasets
    print(datetime.now(), "generate batch list by Datasets")
    dataset_list = list(anno_df['Datasets'].unique())
    dataset_dict = { dataset_list[i]:i for i in range(len(dataset_list)) }
    batch_df = anno_df[['Sample', 'Datasets', 'Cancer types']].copy()
    batch_df = batch_df.drop_duplicates(keep="first")
    batch_df['batch_id'] = batch_df['Datasets'].replace(to_replace=dataset_dict)
    sample_batchid_dict = dict(zip(batch_df['Sample'],batch_df['batch_id']))
    # add batch id to each sample
    exp_df = exp_df.T.copy() # gene by samples
    batch_list = [int(sample_batchid_dict[sample]) for sample in exp_df.columns]
    #batch_df = batch_df[~batch_df.index.duplicated(keep='first')]
    print("#batch ids={:} | #Datasets={:}".format(len(batch_df['batch_id'].unique()), len(batch_df['Datasets'].unique())))

    print(exp_df.isnull().sum().sum())

    # standardarization
    print(datetime.now(), "standardization to have zero mean")
    scaler = skpre.StandardScaler()
    scaled_arr = scaler.fit_transform(exp_df)
    scaled_exp_df = pd.DataFrame(scaled_arr, index=exp_df.index, columns=exp_df.columns)
    print(scaled_exp_df)

    # run pyComBat
    print(datetime.now(), "run combat() for batch correction")
    corrected_exp_df = pycombat(scaled_exp_df, batch_list)
    print(corrected_exp_df)
    if corrected_exp_df.isnull().sum().sum() > 0: #sanity check
        print("WARNING!!!!! #missing values={:}".format(corrected_exp_df.isnull().sum().sum()))
        print("please check missing values before data harmonization")
        sys.exit(1)

    # transpose
    before_exp_df = scaled_exp_df.T   # sample by gene
    after_exp_df = corrected_exp_df.T # sample by gene

    # save to file
    print(datetime.now(), "save to file")
    fout_str = args.output_path + '.combat'
    before_exp_df.to_csv(fout_str+'.NotHomogenized.txt', header=True, index=True, sep="\t")
    after_exp_df.to_csv(fout_str+'.Homogenized.txt', header=True, index=True, sep="\t")
    batch_df.to_csv(fout_str+'.batch.txt', header=True, index=False, sep="\t")
    print("   #samples={:} in not homogenized | {:} in homogenized | {:} in batch".format(
              before_exp_df.shape[0], after_exp_df.shape[0], len(batch_df['Sample'].unique())))

    # PCA
    print(datetime.now(), "plot PCA")
    plot_pca(before_exp_df, after_exp_df, batch_df, fout_str)
