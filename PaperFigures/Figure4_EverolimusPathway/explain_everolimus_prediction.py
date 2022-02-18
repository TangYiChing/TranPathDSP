"""
Return contributing pathway 

Step1. Select best-performing drug model based on prediction result
Step2. Define drug sensitive and resistant samples based on true positives, true negatives
Step3. Identify top 10 pathways based on absolute Shapley values for drug-sensitive, drug-resistant samples, respectively
Step4. Compare enrichment score between drug-resistant and drug-sensitive samples
Step5. Report only contributing pathway with significant differentially expressed 
"""

import sys
import argparse
import numpy as np
import pandas as pd
# set precision digit
pd.set_option("display.precision", 2)
import scipy.stats as scistat
import statsmodels.stats.multitest as multi
import matplotlib.pyplot as plt
import seaborn as sns
#from matplotlib_venn import venn2
sns.set_theme(font_scale=1.3, font='Arial')

def plot_catplot(df, x_str=None, y_str=None, ttl_str=None, fig_str=None):
    #fig, ax = plt.subplots(1, figsize=(12,10), dpi=300)
    fig = sns.catplot(y=y_str, x=x_str, hue="Group", kind="bar", data=df, # row="Set" 
                      height=4, aspect=1.3, legend=True)
    #fig.ax.legend(loc='lower right')
    #plt.title(ttl_str, fontsize=16)
    fig.savefig(fig_str+".PathwaysEnrichment.CATPLOT.png", dpi=300, figsize=(12,10))
    return fig

def plot_venn2(set1, set2, set1_str, set2_str, ttl_str=None, fig_str=None):
    fig, ax = plt.subplots(1, figsize=(8,8), dpi=200)
    ax = venn2([set1,set2], set_labels=(set1_str, set2_str), set_colors=('g', 'crimson'))
    for text in ax.subset_labels:
        text.set_fontsize(20)
    plt.title(ttl_str, fontsize=20)
    fig.savefig(fig_str+".VENN.png", bbox_inches='tight')
    return fig

def plot_confusion_matrix(cf_matrix, x_str="Predicted response", y_str="Actual response", ttl_str=None, fig_str=None):
    """
    :param cf_matrix: 2 by 2 np.array
    :return fig:

    Note: data structure of confusion matrix
     [[ 73   7]
     [  7 141]]
    """
    fig, ax = plt.subplots(1, figsize=(8,8), dpi=200)
    sns.heatmap(cf_matrix, annot=True, annot_kws={'size': 20},
                cmap='YlGnBu', ax=ax, cbar=False)

    n_tp = cf_matrix.iloc[0,0]
    n_tn = cf_matrix.iloc[1,1]
    fig_ttl_str = "Drug={:} | Responder={:} | Non-Responder={:}".format(ttl_str, n_tp, n_tn)
    ax.set_title(fig_ttl_str, fontsize=20);
    ax.set_xlabel('\nPredicted Values', fontsize=20)
    ax.set_ylabel('Actual Values ', fontsize=20);

    ## Ticket labels - List must be in alphabetical order
    ax.set_xticklabels(['1','0'], fontsize=16)
    ax.set_yticklabels(['1','0'], fontsize=16)

    fig.savefig(fig_str+".ConfusionMatrix.HEATMAP.png", bbox_inches='tight')
    return fig

def get_confusion_matrix(y_true, y_pred):
    # generate confusion matrix
    cf_matrix = pd.crosstab(y_true, y_pred,
                            rownames=['Actual response'], colnames=['Predicted response'])
    # sorting
    cf_matrix = cf_matrix[[1,0]]
    cf_matrix = cf_matrix.sort_index(ascending=False)
    print("Confusion matrix=\n{:}".format(cf_matrix))
    #print(cf_matrix.values)
    return cf_matrix

def parse_parameter():
    parser = argparse.ArgumentParser(description='Compare contributing pathway for TP and TN based on feature importance')

    parser.add_argument("-pred", "--pred_path",
                        required = True,
                        help = "path to prediction file (i.e., All.Prediction.txt)")
    parser.add_argument("-shap", "--shap_path",
                        required = True,
                        help = "path to shapley values file (i.e., All.SHAP.txt)")
    parser.add_argument("-enrich", "--enrich_path",
                        help = "path to pathway enrichment file (.CHEM-DGNet-EXP.pkl)")
    parser.add_argument("-info", "--info_path",
                        help = "path to sample information file (i.e., .SampleInfo.txt)")
    parser.add_argument("-drug", "--drug_str",
                        default = "EVEROLIMUS", 
                        help = "string representing drug name in uppercase, default=EVEROLIMUS")
    parser.add_argument("-top", "--top_int",
                        type = int,
                        default = 10,
                        help = "integer representing top N features, default=10")
    parser.add_argument("-o", '--output_path',
                        required = True,
                        help = 'output prefix')
    return parser.parse_args()

if __name__ == "__main__":
    # get args
    args = parse_parameter()

    # Step1. Select best-performing drug model based on prediction result
    prediction_df = pd.read_csv(args.pred_path, header=0, sep="\t")
    if args.drug_str in list(prediction_df['Therapy'].unique()):
        drug_pred_df = prediction_df[prediction_df['Therapy']==args.drug_str].copy()
        # load sample info
        info_df = pd.read_csv(args.info_path, header=0, sep="\t")
        sample_cancer_dict = dict(zip(info_df['Sample'], info_df['Cancer types']))
        drug_pred_df['Cancer types'] = drug_pred_df['Sample'].replace(to_replace=sample_cancer_dict)
        print("Drug={:} | #Cancer types={:}".format(args.drug_str, len(drug_pred_df['Cancer types'].unique())))
        print(drug_pred_df['Cancer types'].unique())
    else:
        print("drug string={:} not found".format(args.drug_str))
        sys.exit()
  
    # Step2. Select drug sensitive and resistant samples based on predicted values
    drug_pred_df.loc[:, 'Predicted Response'] = drug_pred_df['Predicted Probability'].apply(lambda x: 1 if x > 0.5 else 0)
    # get confusion matrix
    drug_cf_matrix = get_confusion_matrix(drug_pred_df['Response'], drug_pred_df['Predicted Response'])
    # plot confusion matrix
    #plot_confusion_matrix(drug_cf_matrix, x_str="Predicted response", y_str="Actual response", 
    #                      ttl_str=args.drug_str, fig_str=args.output_path)
    # separate TP, TN
    responder_df = drug_pred_df[drug_pred_df["Response"]==1]
    non_responder_df = drug_pred_df[drug_pred_df["Response"]==0]
    tp_df = responder_df[responder_df['Predicted Response']==1]
    tn_df = non_responder_df[non_responder_df['Predicted Response']==0]
    #tp_df.to_csv(args.output_path+".TP.Prediction.txt", header=True, index=False, sep="\t")
    #tn_df.to_csv(args.output_path+".TN.Prediction.txt", header=True, index=False, sep="\t")

    if len(tp_df) > 0 and len(tn_df) > 0:
        print("Drug={:} | #TP={:} | #TN={:}".format(args.drug_str, len(tp_df), len(tn_df)))
        # Step3. Identify top 10 pathways based on absolute Shapley values
        shap_df = pd.read_csv(args.shap_path, header=0, index_col=[0,1], sep="\t")
        tp_shap_df = shap_df.loc[tp_df.set_index(['Therapy', 'Sample']).index.tolist()]
        tn_shap_df = shap_df.loc[tn_df.set_index(['Therapy', 'Sample']).index.tolist()]
        # rank feature by mean|SHAP|
        tp_mean_df = tp_shap_df.abs().mean().to_frame(name="mean|SHAP|")
        tp_mean_df = tp_mean_df.sort_values(by=['mean|SHAP|'], ascending=False)
        tp_mean_df.index.name = "pathway"
        tn_mean_df = tn_shap_df.abs().mean().to_frame(name="mean|SHAP|")
        tn_mean_df = tn_mean_df.sort_values(by=['mean|SHAP|'], ascending=False)
        tn_mean_df.index.name = "pathway"
        # select top N
        print("Top{:} contributing pathways".format(args.top_int))
        tp_top_df = tp_mean_df.head(args.top_int)
        tn_top_df = tn_mean_df.head(args.top_int)
        # plot venn diagram
        #plot_venn2(set(tp_top_df.index), set(tn_top_df.index), 
        #           args.drug_str+"-Responder", args.drug_str+"-NonResponder", 
        #           ttl_str="Top contributing pathways", fig_str=args.output_path)
        tp_only_list = sorted(list(set(tp_top_df.index)-set(tn_top_df.index)))
        tn_only_list = sorted(list(set(tn_top_df.index)-set(tp_top_df.index)))
        tp_tn_list = sorted(list(set(tp_top_df.index)&set(tn_top_df.index)))
        print("R only={:} | NR only={:} | R & NR={:}".format(
               len(tp_only_list), len(tn_only_list), len(tp_tn_list)))
        # Step4. Compare pathway enrichment score for two group with Mann-Witney U test
        enrich_df = pd.read_pickle(args.enrich_path)
        tp_enrich_df = enrich_df.loc[tp_df.set_index(['Therapy', 'Sample']).index.tolist()]
        tn_enrich_df = enrich_df.loc[tn_df.set_index(['Therapy', 'Sample']).index.tolist()]
        data_dict = {"R only":tp_only_list, "NR only":tn_only_list, "R & NR":tp_tn_list}
        record_list = []
        for k, v in data_dict.items():
            if len(v) > 0:
                for feature in v:
                    if feature.startswith("CHEM"):
                        print("skipping CHEM features={:}".format(feature))
                    else:
                        w, p = scistat.ranksums(tp_enrich_df[feature].values, tn_enrich_df[feature].values)
                        #print("feature={:} | ranksum={:.2f} | pvalue={:}".format(w,p))
                        record_list.append( (feature, k, w, p) )
        col_list = ['Pathway', 'Set', 'Mann-Witney U Test statistic', 'P-value']
        stat_df = pd.DataFrame.from_records(record_list, columns=col_list)
        stat_df = stat_df.sort_values(by=['P-value', 'Set'], ascending=True)
        stat_df.loc[:, 'FDR'] = multi.multipletests(stat_df['P-value'].values, alpha=0.05, method="fdr_bh")[1]
        print("Mann-Witney U test for enrichment differences\n{:}".format(stat_df))
        stat_df.to_csv(args.output_path+".TopPathwaysEnrichment.TP-TN.Mann-Witney.Report.txt", 
                        header=True, index=False, sep="\t")
        # Step5. Report only contributing pathway with significant differentially expressed
        sig_stat_df = stat_df[stat_df['FDR']<=0.05]
        p_s_dict = dict(zip(sig_stat_df['Pathway'], sig_stat_df['Set']))
        print("#significant differentially expressed top contributing pathways=\n{:}".format(sig_stat_df))
        # skip CHEM feature
        pathway_list = [pathway for pathway in sig_stat_df['Pathway'] if not pathway.startswith("CHEM")]
        tp_enrich_df = tp_enrich_df[pathway_list]
        tn_enrich_df = tn_enrich_df[pathway_list]
        tn = tn_enrich_df.stack().to_frame(name="Enrichment Score").reset_index().copy()
        tn.columns = ['Therapy', 'Sample', 'Pathway', 'Enrichment Score']
        tn['Group'] = args.drug_str+"-NonResponder" #"TN"
        tp = tp_enrich_df.stack().to_frame(name="Enrichment Score").reset_index().copy()
        tp.columns = ['Therapy', 'Sample', 'Pathway', 'Enrichment Score']
        tp['Group'] = args.drug_str+"-Responder" #"TP"
        df = pd.concat([tp, tn], axis=0)
        # merge
        df.loc[:, 'Set'] = df['Pathway'].replace(to_replace=p_s_dict)
        #print(df)
        
        # catplot
        df = df.sort_values(by=['Set'], ascending=False)
        # modify pathway names
        df['Pathway'] = df['Pathway'].str.split('EXP_REACTOME_', n=1, expand=True)[1]
        df['Pathway'] = df['Pathway'].str.replace("_", " ")
        #print(df)
        plot_catplot(df, x_str="Enrichment Score", y_str="Pathway",
                     ttl_str="Enrichment of responder specific pathways", fig_str=args.output_path+".SigDEG")


    else:
        print("ERROR! zero samples found")
        print("    #TP={:} | #TN={:}".format(len(tp_df), len(tn_df)))
        sys.exit()
