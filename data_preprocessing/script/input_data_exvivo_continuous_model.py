"""
Return 5-fold datasets 
"""


import os
import sys
import argparse
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre
import sklearn.model_selection as skms
import sklearn.metrics as skmts
import sklearn.utils as skut

def parse_parameter():
    parser = argparse.ArgumentParser(description='concatenating features of CHEM-DGNet-EXP')
    parser.add_argument("-db", "--db_path",
                        required = True,
                        help = "path to database. e.g. /data2/tang/PathTWIN/TranPathDSP/")
    parser.add_argument("-s", "--seed_int",
                        default = 42,
                        type = int,
                        help = "seed for reproducibility. default=42")
    parser.add_argument("-cv", "--cv_int",
                        default = 5,
                        type = int,
                        help = "integer representing K-fold cross validation. default=5")
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(42)
    # get args
    args = parse_parameter()

    # get data (i.e., feature data + label data)
    fin = args.db_path + 'data_concatenation/data/ExVivo_Continuous/ExVivo_Continuous.CHEM-DGNet-EXP.pkl'
    df = pd.read_pickle(fin)
    print(df)

    # create output folder
    OUTPATH = args.db_path + 'data_concatenation/' + os.path.basename(sys.argv[0]).split('.py')[0] + '/'
    try:
        # Create  Directory  MyDirectory
        os.makedirs(OUTPATH)
    except FileExistsError:
        ##print if directory already exists...
        print("Directory " , OUTPATH,  " already exists...")
    

    # create data folder
    try:
        os.makedirs(OUTPATH+'/')
    except FileExistsError:
        print("Director {:} already exists".format(OUTPATH))
    
    try:
        # Create  Directory  MyDirectory
        os.makedirs(OUTPATH+"/train/")
        os.makedirs(OUTPATH+"/valid/")
        os.makedirs(OUTPATH+"/test/")
    except FileExistsError:
        ##print if directory already exists...
        print("Directory " , OUTPATH+"/train/" ,  " already exists...")
        print("Directory " , OUTPATH+"/valid/" ,  " already exists...")
        print("Directory " , OUTPATH+"/test/" ,  " already exists...")
        

    # split data into K-folds
    kf = skms.KFold(n_splits=args.cv_int, random_state=args.seed_int, shuffle=True)
    # train, valid, test splits
    for i, (train_index, test_index) in enumerate(kf.split(df)):
        train_df = df.iloc[train_index]
        test_df =  df.iloc[test_index]
        train_df, valid_df = skms.train_test_split(train_df, test_size=0.2, random_state=args.seed_int)
        # stats
        pct_train = len(train_df)/len(df)*100
        pct_valid = len(valid_df)/len(df)*100
        pct_test = len(test_df)/len(df)*100
        print('    train={:} {:.2f}% | valid={:} {:.2f}% | test={:} {:.2f}%'.format(
                   train_df.shape, pct_train, valid_df.shape, pct_valid, test_df.shape, pct_test))
        # save train, valid, test
        train_df.to_pickle(OUTPATH+"/train/CHEM-DGNet-EXP.CV_Fold"+str(i)+".train.pkl")
        valid_df.to_pickle(OUTPATH+"/valid/CHEM-DGNet-EXP.CV_Fold"+str(i)+".valid.pkl")
        test_df.to_pickle(OUTPATH+"/test/CHEM-DGNet-EXP.CV_Fold"+str(i)+".test.pkl")
          
    print('find files at {:}'.format(OUTPATH))

