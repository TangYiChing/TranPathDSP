"""
return drug,smile
"""
import os
import pandas as pd

DB_PATH = "/data2/tang/PathTWIN/TranPathDSP/data_preprocessing/"

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
                 'BRD-K70511574 (HMN-214)':'HMN-214','CD 1530':'CD-1530', 'TRASTUZUMAB':'TRASTUZUMAB DERUXTECAN',
                 "(+)­ETOMOXIR(SODIUMSALT)":"(R)-(+)-Etomoxir (sodium salt)"}

DRUG_SMILE_DICT = {'TRASTUZUMAB DERUXTECAN':['CCC1(C2=C(COC1=O)C(=O)N3CC4=C5C(CCC6=C5C(=CC(=C6C)F)N=C4C3=C2)NC(=O)CO)O'],
                   '(R)-(+)-Etomoxir (sodium salt)':['C1[C@](O1)(CCCCCCOC2=CC=C(C=C2)Cl)C(=O)[O-].O.[Na+]']}

def _manual_collected_record():
    """
    :return df: dataframe with headers=[drug,smile]
    """
    df_list = []
    for drug, smile_list in DRUG_SMILE_DICT.items():
        smile_df = pd.DataFrame({'drug':[drug]*len(smile_list),
                                  'smile':smile_list})
        df_list.append(smile_df)
    df = pd.concat(df_list, axis=0)
    return df

def retrieve_PubChem(drugList, db='PubChem'):
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

def retrieve_smile(df, db_str=DB_PATH+'/data/DB.Drug.Smile.txt'):
    """
    :param df: dataframe with header=['Therapy']
    :param db_str: string represening path to target database, a df with headers=[drug,smile]
    :return target_df; dataframe with headers=[drug,smile]
    """
    # create result repository
    smile_df_list = [] # drug,smile
    found_drug_list = []

    # rename drug if has one
    df['Alias'] = df['Therapy'].replace(to_replace=DRUG_SYN_DICT)

    # search database
    db_df = pd.read_csv(db_str, header=0, sep="\t")
    db_drug_list = sorted(list(set(db_df['drug'])&set(df['Alias']))) #Therapy
    if len(db_drug_list) > 0:
        found_drug_list += db_drug_list
        found_smile_df = db_df[db_df['drug'].isin(db_drug_list)]
        smile_df_list.append(found_smile_df)

    # search manual created records
    db_df = _manual_collected_record()
    db_drug_list = sorted(list(set(db_df['drug'])&set(df['Alias']))) #Therapy
    if len(db_drug_list) > 0:
        found_drug_list += db_drug_list
        #found_smile_df = db_df[db_df['drug'].isin(db_drug_list)]
        smile_df_list.append(db_df)
        
    # search PubChem API
    drug_list = sorted(list(set(df['Alias'])-set(found_drug_list)))
    found_smile_df = retrieve_PubChem(drug_list, db='PubChem')
    db_drug_list = sorted(list(set(db_df['drug'])&set(df['Alias']))) #Therapy
    if len(db_drug_list) > 0:
        found_drug_list += db_drug_list
        smile_df_list.append(found_smile_df)
      
    # missing drugs
    found_drug_list = list(set(found_drug_list))
    if len(found_drug_list) != len(df['Alias'].unique()): #Therapy
        miss_drug_list = sorted(list(set(df['Alias'])-set(found_drug_list)))
        print("found {:}/{:} drugs".format(len(found_drug_list), len(df['Alias'].unique())))
        print("    missing drugs={:}".format(miss_drug_list))

    # merge target_df
    if len(smile_df_list) > 0:
        smile_df = pd.concat(smile_df_list, axis=0)
        smile_df = smile_df.drop_duplicates(keep='first')
    else:
        smile_df = None

    # return anyways
    return smile_df, df

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
                 'GeoSearch':data_path+'GeoSearch.resp.txt'}

    # count how many drugs have smile string available
    drug_list = []

    # retrieve smile
    smile_df_list = []
    for dataset, resp_str in data_dict.items():
        print('retriving smile for dataset={:}'.format(dataset))
        df = pd.read_csv(resp_str, header=0, sep="\t")
        smile_df, df = retrieve_smile(df)
        # use Alias in replacement with Therapy
        drug_alias_dict = dict(zip(df['Therapy'],df['Alias']))
        smile_df['drug'] = smile_df['drug'].replace(to_replace=drug_alias_dict)

        # append to result list
        smile_df_list.append(smile_df)
        # save to file (use Alias for following mapping, not Therapy)
        f_str = os.path.basename(resp_str).split('.txt')[0]
        df.to_csv(data_path+f_str+'.Alias.txt',header=True, index=False, sep="\t")
       
        # for counting
        drug_list += list(df['Alias'].unique())
    # merge
    smile_df = pd.concat(smile_df_list,axis=0)
    smile_df = smile_df.drop_duplicates(keep='first')
    # save to file
    fout = "./" + '.'.join(data_dict)
    smile_df.to_csv(fout+'.drug_smile.txt',header=True,index=False,sep="\t")
    print("find parsed file at {:}".format(fout+'.drug_smile.txt'))
    
    # stats report
    common_drug_list = list(set(smile_df['drug'])&set(drug_list))
    print("{:} out of {:} have smile string avilable".format(
          len(common_drug_list),len(drug_list)))
