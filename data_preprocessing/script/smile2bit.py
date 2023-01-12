"""
Convert SMILE to fingerprint bits
"""

import argparse
import pandas as pd
from datetime import datetime

# import RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# set timer
def cal_time(end, start):
    """return time spent"""
    # end = datetime.now(), start = datetime.now()
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    spend = datetime.strptime(str(end), datetimeFormat) - \
            datetime.strptime(str(start),datetimeFormat)
    return spend

def parse_parameter():
    parser = argparse.ArgumentParser(description = "return Morgan Fingerprint in bits")

    parser.add_argument("-debug", "--DEBUG",
                        default =  True,
                        type = bool,
                        help = "display print message")
    parser.add_argument("-s", "--smile_path",
                        required = True,
                        help = "path to chemical smile path. must have headers=[drug, smile]")
    parser.add_argument("-n", "--bit_int",
                        required = True,
                        type = int,
                        help = "integer prepresenting number of bits")
    parser.add_argument("-o", "--output_path",
                        required = True,
                        help = "output string")
    return parser.parse_args()

if __name__ == "__main__":
    # set timer
    start = datetime.now()
    # get args
    args = parse_parameter()
    # parse smile
    smile_df = pd.read_csv(args.smile_path, header=0,  sep="\t")#.set_index('drug')
    smile_df = smile_df.drop_duplicates(subset=['drug'], keep='first').set_index('drug')
    # create result dictionary
    drug_mbit_dict = {}
    record_list = []
    # smile2bits drug by drug
    n_drug = 1
    for idx, row in smile_df.iterrows():
        drug = idx
        smile = row['smile']
        mol = Chem.MolFromSmiles(smile)
        mbit = list( AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=args.bit_int) )
        #drug_mbit_dict.update({drug:mbit})
        # append to result
        record_list.append( tuple([drug]+mbit) )
        if len(mbit) == args.bit_int:
            n_drug+=1
        if drug == 'CHEMBL3348822':
            print(drug, len(mbit))
    print("total {:} drugs with bits".format(n_drug))
    # convert dict to dataframe
    colname_list = ['drug'] + ['mBit_'+str(i) for i in range(args.bit_int)]
    drug_mbit_df = pd.DataFrame.from_records(record_list, columns=colname_list)
    #drug_mbit_df = pd.DataFrame.from_dict(drug_mbit_dict, orient='index', columns=colname_list)
    #drug_mbit_df.index.name = 'drug'
    print("unique drugs={:}".format(len(drug_mbit_df['drug'].unique())))
    # save to file
    drug_mbit_df.to_csv(args.output_path + '.CHEM.' + str(args.bit_int) + '.MBits.txt', header=True, index=False, sep="\t")
    # display print message
    if args.DEBUG == True:
        print('Input=\n{:}'.format(smile_df.head()))
        print('    SMILE={:} (drug by smile)'.format(smile_df.shape))
        print('    MBits={:} (drug by bits)'.format(drug_mbit_df.shape))
    print('[Finished in {:}]'.format(cal_time(datetime.now(), start)))
