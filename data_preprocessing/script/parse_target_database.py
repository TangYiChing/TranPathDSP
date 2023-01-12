import gzip
import pandas as pd

def retrieve_DrugBank(f_str="/data/DR/db/DrugBank/DrugBank.Target.GeneName.txt"):
    """
    Parse from file: /data/DR/db/DrugBank/DrugBank.Target.GeneName.txt
    :return target_df: dataframe with drug, gene headers
    """
    # load db
    db = pd.read_csv(f_str, header=0, sep="\t")
    target_df = db[['DrugBank_NAME', 'Gene_NAME']].copy()
    target_df.columns = ['drug', 'gene']
    target_df.loc[:, 'drug'] = target_df['drug'].str.upper()
    target_df.loc[:, 'gene'] = target_df['gene'].str.upper()

    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')

    #return
    return target_df

def retrieve_LINCS(f_str="/data/DR/db/LINCS/LINCS.TARGET.txt"):
    """
    Parse from file: /data/DR/db/LINCS/LINCS.TARGET.txt
    """
    # load db
    db = pd.read_csv(f_str, header=0, sep="\t")
    target_df = db
    target_df.columns = ['drug', 'gene']
    target_df['drug'] = target_df['drug'].str.upper()
    target_df['gene'] = target_df['gene'].str.upper()

    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')

    #return
    return target_df

def retrieve_TTD(f_str="/data/DR/db/TTD/TTD.DRUG-TARGET.txt"):
    """
    Parser from file: /data/DR/db/TTD/TTD.DRUG-TARGET.txt
    """
    # load db
    db = pd.read_csv(f_str, header=0, sep="\t")
    target_df = db
    target_df.columns = ['drug', 'gene']
    target_df['drug'] = target_df['drug'].str.upper()
    target_df['gene'] = target_df['gene'].str.upper()
   
    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')

    #return
    return target_df

def retrieve_STITCH(f_str='/data/DR/db/STITCH/9606.actions.v5.0.tsv.gz', f2_str='/data/DR/db/STRING/9606.protein.info.v11.0.txt.gz'):
    """
    :return df: dataframe with cid_m, target

    Note:
    =====
    required file: /data/DR/db/STRING/9606.protein.info.v11.0.txt.gz
    """
    # parse tsv.gz
    stitch_df = pd.read_csv(f_str, compression='gzip', sep='\t')
    stitch_df = stitch_df.loc[stitch_df['mode']!='pred_bind']
    stitch_df = stitch_df.loc[(stitch_df['item_id_a'].str.startswith("CID"))&(stitch_df['item_id_b'].str.startswith("9606"))]
    stitch_df = stitch_df[['item_id_a', 'item_id_b']]
    stitch_df.columns = ['cid', 'pid']
    # parse txt.gz
    string_df = pd.read_csv(f2_str, compression='gzip', sep="\t", usecols=['protein_external_id', 'preferred_name'])
    #pid_pnm_dict = dict(zip(string_df['protein_external_id'], string_df['preferred_name']))
    # replace protein id with name
    found_string_pid_list = sorted(list(set(stitch_df['pid'])&set(string_df['protein_external_id'])))
    string_df = string_df.loc[string_df['protein_external_id'].isin(found_string_pid_list)]
    string_df.columns = ['pid', 'target']
    # add preferred_name
    target_df = pd.merge(stitch_df, string_df, left_on='pid', right_on='pid', how='inner')
    target_df = target_df[['cid', 'target']]
    
    # replace cid with name
    f = '/data/DR/db/STITCH/chemicals.v5.0.tsv.gz'
    #name_df = pd.read_csv(f, header=0, sep="\t", compression='gzip')
    for name_df in pd.read_csv(f, header=0, sep='\t', compression='gzip', chunksize=1000):
        cid_name_dict = dict(zip(name_df['chemical'], name_df['name'])) 
        target_df['cid'] = target_df['cid'].replace(to_replace=cid_name_dict)
    # modify column names
    target_df.columns = ['drug', 'gene']
    # convert to upppercase
    target_df['drug'] = target_df['drug'].str.upper()
    target_df['gene'] = target_df['gene'].str.upper()
    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')
    print(target_df)
    #return
    return target_df


def retrieve_CTRPv2(f_str="/data/DR/db/CTRPv2/v20.meta.per_compound.txt"):
    """
    Parse from file: /data/DR/db/CTRPv2/v20.meta.per_compound.txt
    """
    # load db
    ctrpv2_df = pd.read_csv(f_str, header=0, sep="\t")
    ctrpv2_df['cpd_name'] = ctrpv2_df['cpd_name'].str.upper()
    ctrpv2_df['gene_symbol_of_protein_target'] = ctrpv2_df['gene_symbol_of_protein_target'].str.upper()
    ctrpv2_df = ctrpv2_df[['cpd_name', 'gene_symbol_of_protein_target', 'cpd_smiles']]
    ctrpv2_df.columns = ['drug', 'gene', 'smile']
    target_df = ctrpv2_df[['drug', 'gene']]

    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')


    # split ';' 
    target_df = wide2long(target_df, 'drug', 'gene', ';')
    #return
    return target_df

def retrieve_CCLE(f_str="/data/DR/db/CCLE/CCLE_NP24.2009_Drug_data_2015.02.24.csv"):
    """
    Parse from file: /data/DR/db/CCLE/CCLE_NP24.2009_Drug_data_2015.02.24.csv
    """
    # load db
    ccle_df = pd.read_csv(f_str)
    ccle_df = ccle_df[['Compound','Target']]
    ccle_df.columns = ['drug','gene']

    target_df = ccle_df.copy()
    target_df['drug'] = target_df['drug'].str.upper()
    target_df['gene'] = target_df['gene'].str.upper()
    
    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')

    #return
    return target_df

def retrieve_GDSC2(f_str="/data2/tang/PathTWIN/db/GDSC2/GDSC2.TARGET.txt"):
    """
    Parse from file: /data2/tang/PathTWIN/db/GDSC2/GDSC2.TARGET.txt
    """
    # load db
    target_df = pd.read_csv(f_str, header=0, sep="\t")
    
    target_df['drug'] = target_df['drug'].str.upper()
    target_df['gene'] = target_df['gene'].str.upper()

    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')

    #return
    return target_df

def retrieve_CTD(f_str="/data/DR/db/CTD/CTD_chem_gene_ixns.tsv.gz"):
    """
    Parse from file: /data/DR/db/CTD/CTD_chem_gene_ixns.tsv.gz
    """
    # load db
    header_str = b'ChemicalName\tChemicalID\tCasRN\tGeneSymbol\tGeneID\tGeneForms\tOrganism\tOrganismID\tInteraction\tInteractionActions\tPubMedIDs\n'
    header_list = header_str.decode().rstrip().split("\t")
    col_name_list = [ header_list[0], header_list[3] ]
    col_list_list = []
    with gzip.open(f_str) as lines:
        for line in lines:
            
            if line.startswith(b'#'):
                pass
            else:
                for line1 in lines:
                    #print(line1)
                    line1 = line1.decode()
                    data = line1.rstrip().split("\t")
                    #print(len(data),data) 
                    col_list_list.append( [data[0],data[3]] )
               
    # list2df
    record_list = col_list_list
    target_df = pd.DataFrame.from_records(record_list, columns=col_name_list)
    target_df.columns = ['drug','gene']

    target_df['drug'] = target_df['drug'].str.upper()
    target_df['gene'] = target_df['gene'].str.upper()

    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')
    
    #return
    return target_df

def retrieve_Powell(f_str="/data2/tang/PathTWIN/TranPathDSP/data_parsing/data/Paper1_Drug.csv"):
    """
    Parse from file: /data2/tang/PathTWIN/TranPathDSP/data_parsing/data/Paper1_Drug.csv
    """
    df = pd.read_csv(f_str, header=0, index_col=0, sep=",", encoding='latin1')

    # subsetting to include drugs rows, target columns
    col_list = ['Target']
    df = df.iloc[2:, :][col_list]

    df = df.reset_index()
    df.columns = ['drug', 'gene']
    df[['drug', 'source']] = df['drug'].str.split('\s\(|\)$', expand=True).iloc[:,[0,1]]
    # keep unique drug regardless source
    target_df = df[['drug', 'gene']]

    target_df['drug'] = target_df['drug'].str.upper()
    target_df['gene'] = target_df['gene'].str.upper()

    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')

    return target_df

def retrieve_Gao(f_str="/data2/tang/PathTWIN/TranPathDSP/data_parsing/data/41591_2015_BFnm3954_MOESM10_ESM.xlsx"):
    """
    Parse from file: /data2/tang/PathTWIN/TranPathDSP/data_parsing/data/41591_2015_BFnm3954_MOESM10_ESM.xlsx
    """
    df = pd.read_excel(f_str, sheet_name="PCT curve metrics")
    target_df = df[['Treatment', 'Treatment target']]
        
    target_df.columns = ['drug', 'gene']
    target_df['drug'] = target_df['drug'].str.upper()
    target_df['gene'] = target_df['gene'].str.upper()

    target_df = wide2long(target_df, 'drug', 'gene', ',')
    # remove duplicates, nas
    target_df = target_df.dropna(how='any', axis=0)
    target_df = target_df.drop_duplicates(keep='first')

    return target_df

def wide2long(df, idxStr, colStr, sepStr):
    """
    split multiple-item string in one row into separated row
        
    reference:
    https://medium.com/@sureshssarda/pandas-splitting-exploding-a-column-into-multiple-rows-b1b1d59ea12e
    """
    # create a new df with idxStr as the index
    new_df = pd.DataFrame(df[colStr].str.split(sepStr).tolist(), index=df[idxStr]).stack()
    # reset index as it will be duplicated
    new_df = new_df.reset_index([0, idxStr])
    # set column names
    new_df.columns = [idxStr, colStr]
    # return
    return new_df # headers=[idxStr, colStr]

def retrieveSMILE(drugList, db='PubChem'):
    """
    :param drugList: list containing drug names
    :param db: string representing database, default: PubChem
    :return df: dataframe containing columns=[drug, smile]
    """
    # load database
    print( 'retrieveSMILE, db={:}'.format(db) )
    try:
        import pubchempy as pcp
    except:
        print( '    {:} is not found, installing {:}....'.format('pubchempy') )
        os.system("pip install", pubchempy)
    # retrieve smile string
    drug_smile_dict = { 'drug':[], 'smile':[]}
    for drug in drugList:
        cmpd_list = pcp.get_compounds(drug, 'name')
        if len(cmpd_list) == 0:
            print( 'Not Found={:}'.format(drug) )
        if len(cmpd_list) >= 1:
            smile = cmpd_list[0].canonical_smiles # although multiplie pubchem_id, smiles are the same
            drug_smile_dict['drug'].append(drug)
            drug_smile_dict['smile'].append(smile)
    df = pd.DataFrame( drug_smile_dict )
    #df.set_index('drug', inplace=True)
    # return
    return df

if __name__ == "__main__":
    # replace to PPI gene names
    gene_dict = {"HSP70": "HSPA1A;HSPA8;HSPA1B;HSPA4;HSPA5;HSPA14;HSPA1L;HSPA2;HSPBP1;HSPA9;HSPA6;HSPA12B;HSPA4L;HSPA13;HSPA12A",
                 "HSP90": "HSP90AA1;HSP90AB1", "GSK-3": "GSK3A;GSK3B",
                 "INTERLEUKIN RECEPTOR": "IL1R1;IL1R2;IL2RA;IL2RB;IL3RA;IL4R;IL5RA;IL6R;IL7R;IL9R;IL10RA;IL10RB;IL11RA;IL12RB1;IL12RB2;IL13RA1;IL13RA2;IL15RA;IL17RB;IL17RC;IL17RD;IL17RE;IL18R1;IL20RA;IL20RB;IL21R;IL22RA1;IL22RA2;IL23R",
                 "AMPK": "PRKAA1;PRKAA2;PRKAB1;PRKAB2", "PI3K/P110": "PIK3CA;PIK3CB;PIK3CD;PIK3CG",
                 "AURK": "AURKA;AURKB", "SRC/ABL":"SRC;ABL1;ABL2", "MULTI-BCL":"BCL2;BCL2L1;BCL2L2",
                 "JAK1/2": "JAK1;JAK2", " JAK2":"JAK2", "ABL":"ABL1;ABL2", "ABL MRNA":"ABL1;ABL2", "PML-RAR":"PML;RARA",
                 "BCR-ABL": "BCR;ABL1;ABL2", "BCR/ABL FUSION": "BCR;ABL1;ABL2", "PLK":"PLK2;PLK3", "PLK1 MRNA":"PLK1",
                 "PAN-PI3K/MTOR": "PIK3CA;PIK3CB;PIK3CD;PIK3CG;MTOR", "P38":"MAPK14;MAPK1;MAPK11",
                 'FGFR':'FGFR1;FGFR2;FGFR3','PDGFR':'PDGFRA;PDGFRB', '17-BETA-HSD':'HSD17B13', 'TGFB2 MRNA':'TGFB2',
                 'MULTI-RTK':'BCR;ABL1;LYN;HCK;SRC;CDK2;MAP2K1;MAP2K2;MAP3K2;CAMK2G','PTPN13 MRNA':'PTPN13',
                 'PARP':'PARP1;PARP2;PARP3;PARP4', 'IKK':'IKBKB;IKBKG;IKBKE',"FGFR":"FGFR1;FGFR2;FGFR3;FGFR4",
                 'IR':'IGF1R;INSR', 'HGF/MET PATHWAY':'HGF;MET', ' EPHRINS':'EPHA2;EPHA5;EPHB4',
                 'TOPO':'TOP2A;TOP2B', 'MEK':'MAP2K1;MAP2K2', 'MAP3K':'MAP3K2;MAP3K3;MAP3K5',
                 'ROCK':'ROCK1;ROCK2', 'RAC':'RAC1;RAC2;RAC3', "PAN-HDAC": "HDAC1;HDAC2;HDAC3;HDAC6;HDAC8",
                 'CDC25':'CDC25A;CDC25B;CDC25C', 'CXCR':'CXCR2;CXCR3;CXCR4', "PKC":"PRKCA;PRKCB;PRKCD;PRKCE;PRKCG;PRKCI;PRKCQ",
                 'CSNK2':'CSNK2A1;CSNK2A2', 'BET FAMILY':'BRD1;BRD2;BRD3;BRD4;BRD7;BRD8', "VEGF MRNA":"VEGFA",
                 "PI3K": "PIK3CA;PIK3CB;PIK3CD;PIK3CG", "SIRT":"SIRT1;SIRT2;SIRT3;SIRT4;SIRT5;SIRT6;SIRT7",
                 "ACE-FAMILY":"ACE;ACE2", "TGFR":"TGFBR1;TGFBR2", 'UB E3 LIGASE INHIBITOR':'UBR1;UBR2;UBR3;UBR4;UBR5',
                 "IMPDH":"IMPDH1;IMPDH2", "SURVIVIN":"BIRC5", "CDD":"SOST", 'ERG11':'ERG','5HT3R':'HTR3A'}
    ppi_dict = {"P53":"TP53", 'TP53 ACTIVATION':'TP53', "RAR": "RARA", "B-CATENIN":"CTNNB1", "D2":"DRD2",
                "TUBULIN/MICOTUBULES":"TUBB", "VEGFR":"KDR", "SMOOTHENED":"SMO", "MYOSIN II":"MYH2",
                "S17AH":"CYP17A1", "CYP17":"CYP17A1", "BCL-2":"BCL2", "JNK":"MAPK8",
                "XANTHINE OXIDASE INHIBITOR":"XDH", "TRM":"ALPPL2", "AROMATASE":"CYP19A1",
                "IAP":"XIAP", "HIAP":"XIAP", "DNA METHYLTRANSFERASE":"DNMT1", "DNMT":"DNMT1",
                "BCL2L1 MRNA":"BCL2L1", "JAK-2":"JAK2", "JAK-1":"JAK1", "FLT-4":"FLT4",
                "BCL-XL":"BCL2L1", "BCL-W":"BCL2L2", "VEGFR":"KDR", "HER2":"ERBB2", "HER2 MRNA":"ERBB2",
                ' PLK2':"PLK2", ' PLK3':"PLK3", "VEGFR1 MRNA":"FLT1", "MTORC1":"MTOR", "MTORC2":"MTOR",
                "RETINIOIC X RECEPTOR (RXR) AGONIST":"RXRA", "RXR":"RXRA", ' TEC':'TEC', ' ABL':'ABL',
                ' FLT1':'FLT1', ' KIT':'KIT', ' TIE2':'TEK', ' FLT4':'FLT4', ' FLT3':'FLT3', 'FLT-3':'FLT3', ' MET':'MET', ' RET':'RET',
                'FGRF1':'FGFR1', ' FGFR3':'FGFR3', ' FGFR2':'FGFR2', ' VEGFR':'KDR', 'PDK-1':'PDK1', 'THYMIDINE PHOSPHORYLASE': 'TYMP',
                'CANDI TMP1':'TYMP', 'STK26':'MST4', 'FAK':'PTK2','ERK2':'MAPK1', ' EGFR':'EGFR', 'HDAC':'HDAC9',' HDAC6':'HDAC6',
                'CIAP1':'BIRC2', 'IGFR':'IGF1R', ' IR':'IR', 'PPAR':'PPARA', ' ALK':'ALK', ' ROS1':'ROS1', 'COQ8B':'ADCK4',
                'GSK-3A':'GSK3A', 'GSK-3B':'GSK3B', ' GSK3B':'GSK3B', 'ATP SYNTHASE':'ATP1A1',
                'COQ8A':'ADCK3', ' SRC':'SRC', ' PDGFR':'PDGFR', ' CDK9':'CDK9', ' CDK2':'CDK2', ' CDK5':'CDK5',
                ' PDGFRB':'PDGFRB', ' FLT2':'FGFR1', ' PARP2':'PARP2', 'TOP2': 'TOP2A', ' VEGFR2':'KDR',
                'FLT-1':'FLT1', 'VEGFR1':'FLT1', ' VEGFR3':'FLT4', 'MEK1':'MAP2K1', ' MEK2': 'MAP2K2',
                'P38 ALPHA':'MAPK14','PI3K (CLASS 1)':'PIK3CA', ' AURKB':'AURKB', 'COX-2':'MT-CO2', 'PAM PATHWAY':'PAM',
                'NFKB':'NFKBIA', ' FGFR':'FGFR', 'KSP':'CDH16','MDMX INHIBITOR':'MDM4',
                ' PIM3':'PIM3', ' CLK4':'CLK4', ' DAPK3':'DAPK3', ' HIPK2':'HIPK2', 'PI3KBETA':'PIK3CB', 'MDM2 MRNA':'MDM2',
                'MTORC2':'MTOR', 'RAF':'RAF1', 'BRAF MRNA':'BRAF', 'CHK1':'CHEK1', 'CHK2':'CHEK2',
                'PI3KGAMMA':'PIK3CG', 'PI3KALPHA':'PIK3CA', "RIP":"RIPK1", 'CLC-3':'CLCN3', 'TGFB':'TGFB1',
                'DPP-4': 'DPP4', 'CDK5/P25':'CDK5', ' CDK7':'CDK7', ' MTORC2':'MTOR', 'CDK9/CYCLINT':'CDK9', '  PDGFRA':'PDGFRA',
                'AKT':'AKT1', 'MCL-1 MRNA':'MCL1', 'MCL':'MCL1', ' AKT2':'AKT2', ' PLK1':'PLK1','IKK2':'IKBKB',
                'ER':'ESR1','COX-1':'MT-CO1', ' CSF1R':'CSF1R', 'TUBP':'TUBB', 'SRC PATHWAY':'SRC', 'PI3KDELTA':'PIK3CD',
                'EEF-2K':'EEF2K', ' ERBB2':'ERBB2', ' NTRK1':'NTRK1', ' NTRK2':'NTRK2', ' NTRK3':'NTRK3',
                ' VEGFR3/FLT4':'FLT4', ' RON':'RON', ' FGFR1':'FGFR1', ' KDR':'KDR', 'S1P RECEPTOR':'S1PR1',
                ' PDGFRA':'PDGFRA', 'PTK':'PTK2', 'PRKACA MRNA':'PRKACA', 'PKA':'PRKACA', 'TRKA':'NTRK1', 'PI3KCD':'PIK3CD',
                'ACK-1':'TNK2', 'IKKB':'IKBKB', 'RAF MRNA':'RAF1', '5-LOX':'ALOX5', 'EF1A':'EEF1A1',
                'UBIQUITIN-ACTIVATING ENZYME E1': 'UBA1', "PDE5":"PDE5A", 'TERT MRNA':'TERT', 'TYMS MRNA':'TYMS',
                'AURKB MRNA':'AURKB', 'DYNAMIN1':'DNM1', 'ATK':'BTK', 'ESR':'ESR1', 'PKCB':'PRKCB', 'CALCINEURIN':'CABIN1',
                ' CHEK2':'CHEK2', 'PI3KCB':'PIK3CB', 'HSP90A':'HSP90AA1', 'MPS1':'IDUA', 'EG5':'KIF11', 'HPH':'EGLN2',
                'RAD53':'CHEK2', 'H2R':'HRH2', 'P97':'VCP', 'PR':'PGR', 'G9A':'EHMT2', 'CTSL1':'CTSL', 'FPTASE':'FDFT1', 'CALM':'CALM1',
                'PLC':'PLCG1', 'RARRES1RARA':'RARRES1'
                }

    # retrieve target from other sources
    #df0 = retrieve_STITCH()
    #df7 = retrieve_CTD()
    df1 = retrieve_CTRPv2()
    df2 = retrieve_LINCS()
    df3 = retrieve_GDSC2()
    df4 = retrieve_DrugBank()
    df5 = retrieve_TTD()
    df6 = retrieve_CCLE()
    df7 = retrieve_Powell()
    df8 = retrieve_Gao()
    
    df_list = [df1,df2,df3,df4,df5,df6,df7,df8]
    new_df_list = []
    for df in df_list:
        # separate gene strings
        df = wide2long(df, 'drug', 'gene', ';').copy()
        df = wide2long(df, 'drug', 'gene', '|').copy()
        # replace gene
        if len(set(list(gene_dict.keys()))&set(df['gene'].values.tolist())) > 0:
            df['gene'] = df['gene'].replace(to_replace=gene_dict)
        # replace to PPI gene 
        if len(set(list(ppi_dict.keys()))&set(df['gene'].values.tolist())) > 0:
            df['gene'] = df['gene'].replace(to_replace=ppi_dict)
        # append to list
        new_df_list.append(df)
    # merge
    merged_df = pd.concat(new_df_list, axis=0)#.reset_index()
    merged_df = merged_df.drop_duplicates(keep='first')
        
    # save to file
    fout = "./data/"
    merged_df.to_csv(fout+'DB.Drug.Target.txt', header=True, index=False, sep="\t")
