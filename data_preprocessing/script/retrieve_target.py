"""
return target gene data
"""
import os
import sys
import pandas as pd

DB_PATH = "/data2/tang/PathTWIN/TranPathDSP/data_preprocessing/"

DRUG_GENE_DICT = {"MITOMYCINE": ["PALB2", "FANCC", "VWF"], "TEMOZOLOMIDE": ["MGMT"], "IFOSFAMIDE":["NR1I2"],
             "DACARBAZINE":["MGMT"],"ARGINASE": ["ARG1", "ARG2"],
             "DEXAMETHASONE":["NR3C1", "NR0B1", "ANXA1", "NOS2", "NR1I2"],
             "SORAFENIB": ["BRAF", "RAF1", "FLT4", "KDR", "FLT3", "PDGFRB", "KIT", "FGFR1", "RET", "FLT1"],
             "MANUMYCIN A": ["HRAS", "TNF"], "TOPOTECAN" :["TOP1", "TOP1MT"],"TG-100-115":["PIK3CD", "PIK3CG"],
             "(R)-(+)-Etomoxir (sodium salt)":["CPT1A", "CPT1B"], "ABT-737":["BCL2", "BCL2L1", "BCL2L2"],
             "ABT-869":["KDR", "FLT4"], "AFATINIB": ["EGFR", "ERBB2"],
             "AT7867":["AKT1","AKT2","AKT3","RPS6KB2"],"AXITINIB":["FLT1","FLT3","KDR","KIT","PDGFRA","PDGFRB"],
             "ALISERTIB": ["AURKA","AURKB"], "VORINOSTAT": ["HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC8"],
             "CARFILZOMIB": ["PSMB5", "PSMB8", "PSMB1", "PSMB9", "PSMB2", "PSMB10"],
             "CURCUMIN": ["PPARG", "VDR", "ABCC5", "CBR1", "GSTP1"],
             "CERULENIN":["FASN"],
             "CISPLATIN": ["MPG", "A2M", "TF", "ATOX1"],
             "DOCETAXEL": ['TUBB1','BCL2','MAP2','MAP4','MAPT', 'NR1I2'],
             "DOXORUBICIN": ['TOP2A', 'TERT', 'TOP2B', 'NOLC1'],
             "CLOFARABINE": ['RRM1', 'POLA1'],
             "BORTEZOMIB": ['PSMB5', 'PSMB1'],
             "ITRACONAZOLE": ['CYP51A1', 'ERG'],
             "ZOLEDRONIC ACID": ["FDPS", "GGPS1"],
             "TIPIFARNIB": ['FNTB', 'FNTA'],
             "TOZASERTIB": ['AURKB', 'AURKC'],
             "TAMOXIFEN": ["ESR1", "ESR2", "EBP", "AR", "KCNH2", "NR1I2", "ESRRG", "SHBG", "MAPK8"],
             "STAUROSPORINE": ["LCK", "PIM1", "ITK", "SYK", "MAPKAPK2", "GSK3B", "CSK", "CDK2", "PIK3CG",
                               "PDPK1", "PRKCQ", "ZAP70", "CHRM1"],
             "TRICIRIBINE": ["AKT1", "AKT2", "AKT3"],
             "RALOXIFENE": ["ESR1", "ESR2", "SERPINB9", "TFF1"],
             "PACLITAXEL":["TUBB1", "BCL2", "MAP4", "MAP2", "MAPT", "NR1I2"],
             "SUNITINIB": ["PDGFRB", "FLT1", "KIT", "KDR", "FLT4", "FLT3", "CSF1R", "PDGFRA"],
             "RUXOLITINIB": ["JAK2", "JAK1", "JAK3", "TYK2"],
             "PKC412": ['PRKCA', 'KDR', 'KIT', 'PDGFRA', 'PDGFRB', 'FLT3'],
             "MITOTANE": ["CYP11B1", "FDX1", "ESR1", "PGR", "AR"],
             "METHOTREXATE": ["DHFR", "TYMS", "ATIC"],
             "FLAVOPIRIDO": ["CDK2", "CDK5", "CDK9", "CDK1", "CDK6", "EGFR", "CDK4", "CDK8", "CDK7", "PYGM",
                             "PYGB", "PYGL"],
             "FLAVOPIRIDOL": ["CDK2", "CDK5", "CDK9", "CDK1", "CDK6", "EGFR", "CDK4", "CDK8", "CDK7", "PYGM",
                              "PYGB", "PYGL"],
             "FORETINIB": ["HGF", "KDR"], "FLUVASTATIN SODIUM": ['HMGCR', 'HDAC2'],
             "MG-132": ['CAPN1'], "LOVASTATIN": ['HMGCR','ITGAL', 'HDAC2'],
             "PENTOSTATIN":["ADA"], "TRETINOIN": ['RXRB', 'RXRG', 'RARG', 'ALDH1A1', 'GPRC5A', 'ALDH1A2', 'RARRES1' 'RARA', 'RARB',
                                                  'LCN1', 'OBP2A', 'RBP4', 'PDK4', 'RXRA', 'CYP26A1', 'CYP26B1', 'CYP26C1', 'HPGDS'],
             "SIMVASTATIN": ['HMGCR', 'ITGAL', 'HDAC2'], 'IMIQUIMOD': ['TLR7', 'TLR8'],
             "GEMCITABINE":["RRM1", "TYMS", "CMPK1"], "ISTRADEFYLLINE":["ADORA1", "ADORA2A"],
             "BAFILOMYCIN A1": ['ATP6V1A'], "HOMOHARRINGTONINE":["RPL3"],
             "CYCLOPHOSPHAMIDE": ['NR1I2'], "CYTARABINE":["POLB"], "GDC-0449": ['SMO'],
             "METFORMIN":["PRKAB1","ETFDH","GPD1"], "NERATINIB":["EGFR"],
             "NINTEDANIB":["FLT1", "KDR", "FLT4", "PDGFRA", "PDGFRB", "FGFR1", "FGFR2", "FGFR3", "FLT3", "LCK", "LYN", "SRC"],
             "OMACETAXINE MEPESUCCINATE": ['RPL3'], "OUABAIN": ['ATP1A1', 'ATP1A2', 'ATP1A3'],
             "TAS266":["TNFRSF10B"], 'LLM871':["FGFR"], 'HSP990':['HSP90'], 'INC280':['MET'],
             'LFA102':["PRLR"], 'AT406':["XIAP"], 'CID-2858522':['NFKBIA'], 'NECROSTATIN-7':['TNFRSF11A','TNFRSF11B'],'DNMDP':["PDE3A"],
             'SB743921':["CDH116"],'SR-8278':["NR1D1","NR1D2"], 'RAD51 INHIBITOR B02':["RAD51"],
             'ISX 9':["GRIN2B","GRIN1","GRIN2A","GRIN2C","GRIN2D","GRIN3A","GRIN3B"],'NECROSULFONAMIDE':['IL6'],
             'PYR 41':['AGT'], 'CYCLOSPORIN A':['CAMLG','PPP3R2','PPIA','PPIF'], 'FUMONISIN B1':['ABCB1', 'AGT'],
             'ISOEVODIAMINE':['ABCB1','AKT1','BCL2','BCL2A1','BCL2L1','BIRC3','BIRC5','CCND1','CFLAR','CHUK','FASLG','ICAM1','IL1B']}

DRUG_SYN_DICT = {'5FU':'FLUOROURACIL', 'ABT-263 (NAVITOCLAX)':'NAVITOCLAX', 'AC 55649':'AC55649', 'AGK 2':'AGK-2', "INC424":"RUXOLITINIB",
                 'LGH447':'LGH-447', 'PRL\xad3 INHIBITOR':'PRL-3 INHIBITOR I', 'YM155 (SEPANTRONIUM BROMIDE)':'SEPANTRONIUM BROMIDE',
                 'PX 12':'PX-12','A 804598':'A-804598','BINIMETINIB-3.5MPK':'BINIMETINIB','GSK 4112':'GSK4112', 'GSK J4':'GSK-J4', 
                 'GW 405833':'GW-405833', 'GW 843682X':'GW-843682X', 'HBX 41108':'HBX-41108', 'HC 067047':'HC-067047',
                 '(+/-)-BLEBBISTATIN':'BLEBBISTATIN', 'NSC 632839 HYDROCHLORIDE':'NSC632839','PAC 1':'PAC-1',
                 'SELUMETINIB (AZD6244)':'SELUMETINIB','SID 2668150':'SID 26681509', 'SP600125 (PYRAZOLANTHRONE)':'PYRAZOLANTHRONE',
                 'SPAUTIN 1':'SPAUTIN-1','TAMATINIB (R406)':'TAMATINIB','TIPIFARNIB (S ENANTIOMER)':'TIPIFARNIB',
                 "VER 155008":"VER-155008", "SB 225002":"SB-225002", "GW 405833":"GW-405833", "AM 580":"AM-580", 
                 "PF 750":"PF-750", "CHM 1":"CHM-1",'WNT974':'LGK974',
                 'GEMCITABINE-50MPK':'GEMCITABINE', 'ABRAXANE':'PACLITAXEL','SJ 172550':'SJ-172550','CH 55':'CH-55',
                 'AT406 (SM- 406, ARRY- 334543)':'AT406','PF 573228':'PF-573228',
                 'PIFITHRIN\xad±':'PIFITHRIN', 'CI 976':'CI-976', 'CID 2858522':'CID-2858522', 'CID 5951923':'CID-5951923',
                 'NECROSTATIN\xad7':'NECROSTATIN-7', 'EX 527 (SELISISTAT)':'SELISISTAT',
                 'ML 210':'ML210', 'ML029':'ML029', 'MLN 2480':'MLN2480', 'BENDAMUSTINE HCL':'BENDAMUSTINE HYDROCHLORIDE', 
                 'EPIGALLOCATECHIN-3-MONOGALLATE[(-)-EPIGALLOCATECHIN GALLATE]':'EPIGALLOCATECHIN GALLATE',
                 'L-685,458':'L-685458', 'LE 135':'LE-135', 'LY 2183240':'LY-2183240', 'OSI-906 (LINSITINIB)':'LINSITINIB', 
                 'NSC 95397':'NSC95397', 'NSC23766(HYDROCHLORIDE)':'NSC23766',
                 'SCH 529074':'SCH-529074', 'SCH 530348':'SCH-530348', 'SCH 79797 DIHYDROCHLORIDE':'SCH-79797', 
                 'SMER 3':'SMER-3', 'SN 38':'SN-38', 'SR 1001':'SR-1001', 'SR 8278':'SR-8278', 'STF 31':'STF-31',
                 'JW 480':'JW-480', 'JW 55':'JW-55', 'KO 143':'KO-143', 'PRIMA-1MET (Broad_2)':'PRIMA-1', 'CERANIB\xad2':'CERANIB-2',
                 'FLAVOPIRIDOL (ALVOCIDIB)':'ALVOCIDIB', 'I-BET 151 DIHYDROCHLORIDE':'I-BET151',
                 'S3I-201 (NSC 74859)':'NSC 74859','PROCHLORPERAZINE DIMALEATE':'PROCHLORPERAZINE','MK-2206 2HCL':'MK-2206',
                 'PROCARBAZINE HCL':'PROCARBAZINE','ERISMODEGIB':'SONIDEGIB','FINGOLIMOD (FTY720) HCL':'FINGOLIMOD',
                 'PF 4800567 HYDROCHLORIDE':'PF-4800567 HYDROCHLORIDE', 'PF 184':'PF-184','BIX01294(HYDROCHLORIDE HYDRATE)':'BIX-01294',
                 'PI 103 HYDROCHLORIDE':'PI-103', 'MDIVI 1':'MDIVI-1','MARITOCLAX':'MARINOPYRROLE A', 'FUMONISIN_B1':'FUMONISIN B1',
                 'NEURONAL DIFFERENTIATION INDUCER III':'ISX 9','SR-1001':'SR1001', 'BLEOMYCIN A2':'BLEOMYCIN', 
                 'BMS 195614':'BMS-195614', 'BMS 270394':'BMS-270394', 'BMS 345541':'BMS-345541','QS 11':'QS-11',
                 'BRD-K70511574 (HMN-214)':'HMN-214','CD 1530':'CD-1530','TRASTUZUMAB':'TRASTUZUMAB DERUXTECAN',
                 "(+)­ETOMOXIR(SODIUMSALT)":"(R)-(+)-Etomoxir (sodium salt)"}

def _manual_collected_record():
    """
    :return df: dataframe with headers=[drug,gene]
    """
    df_list = []
    for drug, gene_list in DRUG_GENE_DICT.items():
        target_df = pd.DataFrame({'drug':[drug]*len(gene_list),
                                  'gene':gene_list})
        df_list.append(target_df)
    df = pd.concat(df_list, axis=0)
    return df

def retrieve_target(df, db_str=DB_PATH+'/data/DB.Drug.Target.txt'):
    """
    :param df: dataframe with header=['Therapy']
    :param db_str: string represening path to target database, a df with headers=[drug,gene]
    :return target_df; dataframe with headers=[drug,gene]
    """
    # create result repository
    target_df_list = [] # drug,gene
    found_drug_list = []

    # rename drug if has one
    df['Alias'] = df['Therapy'].replace(to_replace=DRUG_SYN_DICT)


    # search database
    db_df = pd.read_csv(db_str, header=0, sep="\t")
    db_drug_list = sorted(list(set(db_df['drug'])&set(df['Alias']))) #Therapy
    if len(db_drug_list) > 0:
        found_drug_list += db_drug_list
        found_target_df = db_df[db_df['drug'].isin(db_drug_list)]
        target_df_list.append(found_target_df)

    # search manual created records
    db_df = _manual_collected_record()
    db_drug_list = sorted(list(set(db_df['drug'])&set(df['Alias']))) #Therapy
    
    if len(db_drug_list) > 0:
        found_drug_list += db_drug_list   
        found_target_df = db_df[db_df['drug'].isin(db_drug_list)]
        target_df_list.append(found_target_df)

    # missing drugs
    found_drug_list = list(set(found_drug_list))
    if len(found_drug_list) != len(df['Alias'].unique()): #Therapy
        miss_drug_list = sorted(list(set(df['Alias'])-set(found_drug_list)))
        print("found {:}/{:} drugs".format(len(found_drug_list), len(df['Alias'].unique())))
        print("    missing drugs={:}".format(miss_drug_list))
    
    # merge target_df
    if len(target_df_list) > 0:
        target_df = pd.concat(target_df_list, axis=0)
        target_df = target_df.drop_duplicates(keep='first')
    else:
        target_df = None

    # return anyways
    return target_df, df

def parse_parameter():
    parser = argparse.ArgumentParser(description='concatenating features of CHEM-DGNet-EXP')
    parser.add_argument("-db", "--db_path",
                        required = True,
                        help = "path to database. e.g. /data2/tang/PathTWIN/TranPathDSP/")
    parser.add_argument("-o", "--output_path",
                        required = True,
                        help = "path to ouput file")
    return parser.parse_args()

if __name__ == "__main__":
    # get args
    args = parse_parameter()

    # get files
    data_path = args.db_path #"/data2/tang/PathTWIN/TranPathDSP/data_parsing/"
    data_dict = {'GDSCv2':data_path+'GDSCv2.resp_1-AUC.txt',
                 'Gao2015':data_path+'Gao2015.resp.txt',
                 'Powell2020':data_path+'Powell2020.resp.txt',
                 'Lee2021':data_path+'Lee2021.resp.txt',
                 'GeoSearch':data_path+'GeoSearch.resp.txt',
                 'Ding2016':data_path+'Ding2016.resp.txt'}

    
    # count how many drugs have smile string available
    drug_list = []

    # retrieve target
    target_df_list = []
    for dataset, resp_str in data_dict.items():
        print('retriving target for dataset={:}'.format(dataset))
        df = pd.read_csv(resp_str, header=0, sep="\t")
        target_df, df = retrieve_target(df)
        # use Alias in replacement with Therapy
        drug_alias_dict = dict(zip(df['Therapy'],df['Alias']))
        target_df['drug'] = target_df['drug'].replace(to_replace=drug_alias_dict)
        # append to result list
        target_df_list.append(target_df)
        # save to file (use Alias for following mapping, not Therapy)
        f_str = os.path.basename(resp_str).split('.txt')[0]
        df.to_csv(data_path+f_str+'.Alias.txt',header=True, index=False, sep="\t")
        # for counting
        drug_list += list(df['Alias'].unique())
    # merge
    target_df = pd.concat(target_df_list,axis=0)
    target_df = target_df.drop_duplicates(keep='first')
    # save to file
    if args.output_path:
        fout = args.output_path + '.'.join(data_dict)
    else:
        fout = "./"+ '.'.join(data_dict)
    target_df.to_csv(fout+'.drug_gene.txt',header=True,index=False,sep="\t")
    print("find parsed file at {:}".format(fout+'.drug_gene.txt'))

    # stats report
    common_drug_list = list(set(target_df['drug'])&set(drug_list))
    print("{:} out of {:} have associated gene avilable".format(
          len(common_drug_list),len(drug_list)))
    print("    #genes per drug={:.1f}".format(target_df.groupby(by=['drug']).count().mean().values[0]))
