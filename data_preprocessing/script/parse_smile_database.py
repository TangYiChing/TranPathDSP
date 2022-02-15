"""
return drug,smile
"""


import pandas as pd

def retrieve_GDSC2(f_str="/data/DR/db/GDSC/GDSC_DRUG_SMILE.txt"):
    """
    Parse from file: /data/DR/db/GDSC/GDSC_DRUG_SMILE.txt
    """
    df = pd.read_csv(f_str,header=0,sep="\t")
    df.columns = ['drug','smile']
    df['drug'] = df['drug'].str.upper()
    return df

def retrieve_CCLE(f_str="/data/DR/db/CCLE/CCLE_DRUG_SMILE.txt"):
    """
    Parse from file: /data/DR/db/CCLE/CCLE_DRUG_SMILE.txt
    """
    df = pd.read_csv(f_str,header=0,sep="\t")
    df.columns = ['drug','smile']
    df['drug'] = df['drug'].str.upper()
    return df

def retrieve_CTRPv2(f_str="/data/DR/db/CTRPv2/v20.meta.per_compound.txt"):
    """
    Parse from file: /data/DR/db/CTRPv2/v20.meta.per_compound.txt
    """
    df = pd.read_csv(f_str,header=0,sep="\t",usecols=['cpd_name', 'cpd_smiles'])
    df.columns = ['drug','smile']
    df['drug'] = df['drug'].str.upper()
    return df

def retrieve_LINCS(f_str="/data/DR/db/LINCS/LINCS.SMILE.txt"):
    """
    Parse from file: /data/DR/db/LINCS/LINCS.SMILE.txt
    """
    df = pd.read_csv(f_str,header=0,sep="\t")
    df.columns = ['drug','smile']
    df['drug'] = df['drug'].str.upper()
    return df

def retrieve_DrugBank(f_str="/data/DR/db/DrugBank/DrugBank.SMILE.txt"):
    """
    Parse from file: /data/DR/db/DrugBank/DrugBank.SMILE.txt
    """
    df = pd.read_csv(f_str,header=0,sep="\t")
    df.columns = ['drug','smile']
    df['drug'] = df['drug'].str.upper()
    return df

if __name__ == "__main__":
    # retrieve smile from other sources
    df1 = retrieve_CTRPv2()
    df2 = retrieve_LINCS()
    df3 = retrieve_GDSC2()
    df4 = retrieve_DrugBank()
    df5 = retrieve_CCLE()

    # merge
    df_list = [df1,df2,df3,df4,df5]
    merged_df = pd.concat(df_list, axis=0)#.reset_index()
    merged_df = merged_df.drop_duplicates(keep='first')

    # save to file
    fout = "./data/"
    merged_df.to_csv(fout+'DB.Drug.Smile.txt', header=True, index=False, sep="\t")
