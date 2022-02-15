"""
Return GDSC response for Powell dataset

Note: 
Reid Powell used their pipeline on GDSC data, so that drug response values are in the same format as their data

Last update: 1/14/2022
"""

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    GDSC_ori = "../data_parsing/GDSCv2.resp_1-AUC.Alias.txt"
    GDSC_reid = "./GDSC_Drug_ALL_DRC.csv"

    ori_df = pd.read_csv(GDSC_ori, header=0, sep="\t")
    reid_df = pd.read_csv(GDSC_reid, header=0, sep=",")
    #print(ori_df)
    #print(reid_df)

    # uppercase drug name
    reid_df['DRUG_NAME'] = reid_df['DRUG_NAME'].str.upper()
    reid_df.columns = ['Therapy', 'Sample', 'Response']

    # subsetting to include drug-cell pairs
    ori_df = ori_df.set_index(['Therapy', 'Sample'])
    reid_df = reid_df.set_index(['Therapy', 'Sample'])
    idx_list = sorted(list(set(ori_df.index)&set(reid_df.index)))
    ori_df = ori_df.loc[idx_list]
    reid_df = reid_df.loc[idx_list]
    print("#drug-cell pairs={:}".format(len(idx_list)))

    # remove duplicates
    ori_df = ori_df[~ori_df.index.duplicated(keep='first')]
    print(ori_df)
    print(reid_df)

    # add annotations
    reid_df = pd.concat([reid_df, ori_df[['Cancer types', 'Datasets', 'Alias']]], axis=1).copy()
    print(reid_df)

    # save to file
    reid_df.to_csv("../data_parsing/GDSCv2.resp_PowellAUC.Alias.txt", header=True, index=True, sep="\t")

    # plot scatter to compare AUCs
    ori_df.loc[:, 'GDSC AUC'] = 1 - ori_df['Response']
    df = pd.concat([ori_df[['GDSC AUC']], reid_df[['Response']]], axis=1)
    df.columns = ['GDSC AUC', 'AUC_FA']
    print(df)

    fig, ax = plt.subplots(1, figsize=(10,10), dpi=200)
    df.plot.scatter(x='GDSC AUC', y='AUC_FA', ax=ax)
    plt.xlabel('GDSC AUC', fontsize=20)
    plt.ylabel('adjusted AUC', fontsize=20)
    fig.savefig("./retrieve_GDSC_from_Powell.SCATTERPLOT.png", bbox_inches="tight")
