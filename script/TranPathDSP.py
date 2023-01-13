"""
Return validation score on the test set
"""


import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre
import sklearn.model_selection as skms
import sklearn.metrics as skmts
import sklearn.utils as skut


import util as ut
import metrices as mts

import shap as sp
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.pipeline import make_pipeline


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2


def get_optimizer(optimizer_str='adam', learning_rate=0.01):
    """
    :param optimizer_str: string representing avaliable optimizers in keras
    :param learning_rate: float representing learning rate assigned to optimizer
    :return optimizer_fn: optimizer function of keras
    """
    if optimizer_str == 'adam':
        optimizer_fn = keras.optimizers.Adam(lr=float(learning_rate), clipvalue=5.0)
    elif optimizer_str == 'adamax':
        optimizer_fn = keras.optimizers.Adamax(lr=float(learning_rate), clipvalue=5.0)
    elif optimizer_str == 'adadelta':
        optimizer_fn = keras.optimizers.Adadelta(lr=float(learning_rate), clipvalue=5.0)
    elif optimizer_str == 'rmsprop':
        optimizer_fn = keras.optimizers.RMSprop(lr=float(learning_rate), clipvalue=5.0)
    else:
        print('ERROR! optimizer={:} not in [adam, adamax, sgd, adadelta, rmsprop]'.format(
               optimizer_str))
        sys.exit(1)
    return optimizer_fn

def get_classifier_optimizer(optimizer_str, learning_rate):
    """
    :param optimizer_str: string representing avaliable optimizers in keras
    :param learning_rate: float representing learning rate assigned to optimizer
    :return optimizer_fn: optimizer function of keras
    """
    if optimizer_str == 'adam':
        optimizer_fn = keras.optimizers.Adam(lr=float(learning_rate))
    elif optimizer_str == 'adamax':
        optimizer_fn = keras.optimizers.Adamax(lr=float(learning_rate))
    elif optimizer_str == 'adadelta':
        optimizer_fn = keras.optimizers.Adadelta(lr=float(learning_rate))
    elif optimizer_str == 'rmsprop':
        optimizer_fn = keras.optimizers.RMSprop(lr=float(learning_rate))
    else:
        print('ERROR! optimizer={:} not in [adam, adamax, sgd, adadelta, rmsprop]'.format(
               optimizer_str))
        sys.exit(1)
    return optimizer_fn

def create_classifier(n_input, pretrained_model, param_dict):
    """
    :param n_input: integer representing the number of features
    :param pretrained_model: path to a pretrained model
    :param param_dict: dictionary of hyperparameters
    return model
    """
    # load pretrained model
    base_model = keras.models.load_model(pretrained_model)
    for layer in base_model.layers:
        layer.trainable = param_dict['retrain'] #False = freeze weights

    # Input
    model_input = base_model.get_layer(param_dict['use_layer']).output

    # hidden layer
    if param_dict['n_layer'] == 0:
        model_output = Dense(1, activation="sigmoid", name="outputs")(model_input)

    elif param_dict['n_layer'] == 1:

        FC1 = Dense(param_dict['n_neuron'], activation=param_dict['activation'], kernel_initializer="glorot_normal", name="FC1")(model_input)
        FC1 = Dropout(param_dict['dropout_float'], name="dropoutFC1")(FC1)

        model_output = Dense(1, activation="sigmoid", name="outputs")(FC1)
    else:

        FC1 = Dense(param_dict['n_neuron'], activation=param_dict['activation'], kernel_initializer="glorot_normal", name="FC1")(model_input)
        FC1 = Dropout(param_dict['dropout_float'], name="dropoutFC1")(FC1)

        FC2 = Dense(int(param_dict['n_neuron']/2), activation=param_dict['activation'], kernel_initializer="glorot_normal", name="FC2")(FC1)
        FC2 = Dropout(param_dict['dropout_float'], name="dropoutFC2")(FC2)

        model_output = Dense(1, activation="sigmoid", name="outputs")(FC2)

    # Build model
    model = Model(inputs=base_model.inputs, outputs=model_output)
    # Compile
    optimizer_fn = get_classifier_optimizer(param_dict['optimizer'], param_dict['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer_fn,
                  metrics=[keras.metrics.AUC()])
    
    # return
    return model

def create_regressor(n_input, pretrained_model, param_dict):
    """
    :param n_input: integer representing the number of features
    :param pretrained_model: path to a pretrained model
    :param param_dict: dictionary of hyperparameters
    return model
    """
    # load pretrained model
    base_model = keras.models.load_model(pretrained_model)
    for layer in base_model.layers:
        layer.trainable = param_dict['retrain'] #True #False # freeze weights

    # Input
    model_input = base_model.get_layer(param_dict['use_layer']).output

    # hidden layer
    if param_dict['n_layer'] == 0:
        model_output = Dense(1, activation="linear", name="outputs")(model_input)

    elif param_dict['n_layer'] == 1:

        FC1 = Dense(param_dict['n_neuron'], activation=param_dict['activation'], kernel_initializer="glorot_normal", name="FC1")(model_input)
        FC1 = Dropout(param_dict['dropout_float'], name="dropoutFC1")(FC1)

        model_output = Dense(1, activation="linear", name="outputs")(FC1)
    else:

        FC1 = Dense(param_dict['n_neuron'], activation=param_dict['activation'], kernel_initializer="glorot_normal", name="FC1")(model_input)
        FC1 = Dropout(param_dict['dropout_float'], name="dropoutFC1")(FC1)

        FC2 = Dense(int(param_dict['n_neuron']/2), activation=param_dict['activation'], kernel_initializer="glorot_normal", name="FC2")(FC1)
        FC2 = Dropout(param_dict['dropout_float'], name="dropoutFC2")(FC2)

        model_output = Dense(1, activation="linear", name="outputs")(FC2)

    # Build model
    model = Model(inputs=base_model.inputs, outputs=model_output)

    # Compile
    optimizer_fn = get_optimizer(param_dict['optimizer'], param_dict['learning_rate'])
    model.compile(loss='mean_squared_error', optimizer=optimizer_fn,
                      metrics=[keras.metrics.MeanSquaredError()])

    # return
    return model

def run_cv(train_df, valid_df, test_df, cv_str=None, task_str='classification', pretrained_model=None, param_dict=None, fout_str="./"):
    """
    :return pred_df: dataframe with headers=[Response, Predicted Response]
    """
    # get inputs
    if task_str == 'classification':
        ########################################################################
        # balance ratio of  positive:negative for training data
        #     upsampling minority class to double-size, 
        #     and undersampling majority class to the same size as the minority
        ########################################################################
        n_pos = np.count_nonzero(train_df.iloc[:,-1]==1)
        n_neg = np.count_nonzero(train_df.iloc[:,-1]==0)
        print("    Before, #1={:} | #0={:}".format(n_pos, n_neg))
        ratio = 3
        if n_pos/n_neg >= ratio:
            major_int = n_pos
            minor_int = n_neg
            pipe = make_pipeline( RandomOverSampler(sampling_strategy={0:minor_int*ratio}, random_state=args.seed_int),
                                  RandomUnderSampler(sampling_strategy={1:minor_int*ratio}, random_state=args.seed_int))

            #pipe = make_pipeline( SMOTE(sampling_strategy={0:minor_int*3}, random_state=args.seed_int),
            #                      NearMiss(sampling_strategy={1:minor_int*3}))
        elif n_neg/n_pos >= ratio:
            major_int = n_neg
            minor_int = n_pos
            pipe = make_pipeline( RandomOverSampler(sampling_strategy={1:minor_int*ratio}, random_state=args.seed_int),
                                  RandomUnderSampler(sampling_strategy={0:minor_int*ratio}, random_state=args.seed_int))

            #pipe = make_pipeline( SMOTE(sampling_strategy={1:minor_int*3}, random_state=args.seed_int),
            #                      NearMiss(sampling_strategy={0:minor_int*3}))
        #elif n_pos/n_neg < 3 or n_neg/n_pos < 3:
        else:
            pipe = make_pipeline( SMOTE(random_state=args.seed_int) )
        train_X_df, train_y_df = pipe.fit_resample(train_df.iloc[:, :-1], train_df.iloc[:, -1])
        print("    After, #1={:} | #0={:}".format(
              np.count_nonzero(train_y_df==1), np.count_nonzero(train_y_df==0)))
        train_X_arr, train_y_arr = train_X_df.values, train_y_df.values
        
    else:
        train_X_arr, train_y_arr = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values

    valid_X_arr, valid_y_arr = valid_df.iloc[:, :-1].values, valid_df.iloc[:, -1].values
    test_X_arr, test_y_arr = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values
    n_input = train_X_arr.shape[1]

    # create a model 
    if task_str == 'classification':
        model = create_classifier(n_input, pretrained_model, param_dict)
    else:
        model = create_regressor(n_input, pretrained_model, param_dict)

    print(model.summary())
    print(param_dict)
    print("seed number={:}".format(args.seed_int))
    # fit model 
    model_h5_str = fout_str +'.' + cv_str + '.TranPathDSP.h5'
    callback_list = [EarlyStopping(monitor='val_loss', mode='auto', patience=param_dict["earlyStop"]),
                     ModelCheckpoint((model_h5_str), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
                    ]
    history = model.fit(train_X_arr, train_y_arr, epochs=param_dict["epoch"], 
                        shuffle=True, batch_size=param_dict["batch_size"],verbose=0,
                        validation_data=(valid_X_arr, valid_y_arr),callbacks=callback_list)

    # evaluate on the test set
    pred_arr = model.predict(test_X_arr).flatten()
    pred_df = test_df[['Response']].copy()
    if task_str == 'classification':
        pred_df['Predicted Probability'] = pred_arr.tolist()
    else:
        pred_df['Predicted Response'] = pred_arr.tolist()
    print(pred_df)

    # calcuate shapley
    background = train_X_arr[:100] # random select 100 samples as baseline
    explainer = sp.DeepExplainer(model, background)
    shap_arr = explainer.shap_values(test_X_arr)[0]
    shap_df = pd.DataFrame(shap_arr, index=test_df.index, columns=test_df.iloc[:,:-1].columns)
    shap_df.to_csv(fout_str+"."+cv_str+".SHAP.txt", header=True, index=True, sep="\t")
    #print(shap_df)

    tf.keras.backend.clear_session() # avoid clutter from old model
    return pred_df

def parse_parameter():
    parser = argparse.ArgumentParser(description='Validate Model with Cross Validation')

    parser.add_argument("-data", "--data_path", 
                        help = "path to input data (i.e., .pkl)")
    parser.add_argument("-cv", "--cv_int",
                        type = int,
                        default = 5,
                        help = "integer representing K-fold. default=5")
    parser.add_argument("-f", "--feature_str",
                        default = "REACTOME",
                        choices = ["PID", "REACTOME", "PID_REACTOME"],
                        help = "string representing pathway features to be used. default=REACTOME")
    parser.add_argument("-norm", "--normalization_str",
                        default = "standard",
                        choices = ["standard", "minmax"],
                        help = "normalization methods")
    parser.add_argument("-task", "--task_str",
                        required = True,
                        choices = ["regression", "classification"],
                        help = "string representing job task")
    parser.add_argument("-pretrained", "--pretrained_str",
                        required = True,
                        help = "path to a pre-trained cell line model")
    parser.add_argument("-param", "--param_path",
                        default = None,
                        help = "path to hyperparameters file")
    parser.add_argument("-g", "--gpu_str",
                        default = '0',
                        type = str,
                        help = "gpu device ids for CUDA_VISIBLE_DEVICES")
    parser.add_argument("-s", "--seed_int",
                        required = False,
                        default = 42,
                        type = int,
                        help = "seed for reproducibility. default=42")
    parser.add_argument("-o", '--output_path',
                        required = True,
                        help = 'output prefix')
    # get parameters
    parser.add_argument(
        '--epochs',
        type=int,
        required=False,
        default=66,
        help='number of training epochs')
    parser.add_argument(
        '--batch_size',
        type=int,
        required=False,
        default=35,
        help='batch size')
    parser.add_argument(
        '--optimizer',
        type=str,
        required=False,
        default='adamax',
        help='keras optimizer to use')
    parser.add_argument(
        '--learning_rate',
        type=float,
        required=False,
        default=0.39,
        help='learning rate')
    parser.add_argument(
        '--early_stop',
        type=int,
        required=False,
        default=12,
        help='activates keras callback for early stopping of training in function of the monitored variable specified.')
    parser.add_argument(
        '--dense',
        type=int,
        required=False,
        default=0.01,
        help='n_neurons')
    parser.add_argument(
        '--activation',
        type=str,
        required=False,
        default='relu',
        help='activation function')
    parser.add_argument(
        '--dropout',
        type=float,
        required=False,
        default=0.1,
        help='dropout')

    return parser.parse_args()

if __name__ == "__main__":
    # get args
    args = parse_parameter()
    #Load parameters (originally from file)
    print(vars(args))
    param_dict = vars(args)
    param_dict['batch_size'] = round(float(args.batch_size))
    param_dict['dropout_float'] = float(args.dropout)
    param_dict['earlyStop'] = round(float(args.early_stop))
    param_dict['epoch'] = round(float(args.epochs))
    param_dict['learning_rate'] = float(args.learning_rate)
    param_dict['n_layer'] = 2
    param_dict['n_neuron'] = round(float(args.dense))
    param_dict['retrain'] = eval("False")
    param_dict['use_layer'] = "hidden1"
    param_dict['activation'] = args.activation
    #print(param_dict)

    # set seed
    np.random.seed(args.seed_int)
    tf.random.set_seed(args.seed_int)
    # set device
    num_cores = 8
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_str
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                                      inter_op_parallelism_threads=num_cores,
                                      allow_soft_placement=True,
                                      device_count = {'CPU' : 1,
                                                      'GPU' : 1})
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # timer
    start = datetime.now()

    # load data
    data_df = pd.read_pickle(args.data_path)
    if data_df.isnull().sum().sum() > 0:
        print('{:} contains #missing values (i.e., NAN)={:}'.format(
               data_df.isnull().sum().sum()))
    
    # subsetting to include wanted pathway features source: PID, REACTOME, or both
    data_df = ut.subsetting_feature_column(data_df, feature_str=args.feature_str)

    # binarize response
    if args.task_str == 'classification':
        data_df['Response'] = data_df['Response'].replace(to_replace={'R':1, 'NR':0})
        
    # cross validation
    cv_dict = {}
    if args.task_str == 'classification':
        kf = skms.StratifiedKFold(n_splits=args.cv_int, random_state=args.seed_int, shuffle=True)
        # train, valid, test splits
        X_df = data_df.iloc[:, :-1]
        y_df = data_df.iloc[:, -1]
        for i, (train_index, test_index) in enumerate(kf.split(X_df, y_df)):
            # create result dict
            data_dict = {}
            train_df = pd.concat([X_df.iloc[train_index], y_df.iloc[train_index]], axis=1)
            test_df =  pd.concat([X_df.iloc[test_index], y_df.iloc[test_index]], axis=1)
            train_df, valid_df = skms.train_test_split(train_df, test_size=0.2, random_state=args.seed_int)
            # stats
            pct_train = len(train_df)/len(data_df)*100
            pct_valid = len(valid_df)/len(data_df)*100
            pct_test = len(test_df)/len(data_df)*100
            print('    train={:} {:.2f}% | valid={:} {:.2f}% | test={:} {:.2f}%'.format(
                 train_df.shape, pct_train, valid_df.shape, pct_valid, test_df.shape, pct_test))
            # append to dictionary
            data_dict['train'] = train_df
            data_dict['valid'] = valid_df
            data_dict['test'] = test_df
            cv_str = "cv"+str(i)
            cv_dict[cv_str] = data_dict
    else:
        kf = skms.KFold(n_splits=args.cv_int, random_state=args.seed_int, shuffle=True)
        # train, valid, test splits
        for i, (train_index, test_index) in enumerate(kf.split(data_df)):
            # create result dict
            data_dict = {}
            train_df = data_df.iloc[train_index]
            test_df =  data_df.iloc[test_index]
            train_df, valid_df = skms.train_test_split(train_df, test_size=0.2, random_state=args.seed_int)
            # stats
            pct_train = len(train_df)/len(data_df)*100
            pct_valid = len(valid_df)/len(data_df)*100
            pct_test = len(test_df)/len(data_df)*100
            print('    train={:} {:.2f}% | valid={:} {:.2f}% | test={:} {:.2f}%'.format(
                 train_df.shape, pct_train, valid_df.shape, pct_valid, test_df.shape, pct_test))
            # append to dictionary
            data_dict['train'] = train_df
            data_dict['valid'] = valid_df
            data_dict['test'] = test_df
            cv_str = "cv"+str(i)
            cv_dict[cv_str] = data_dict


    # scaling data
    norm_str = args.normalization_str
    col_list = ['CHEM', 'DGNet', 'EXP'] # these features will be scaled separately
    for i in range(0, args.cv_int):
        cv_str = "cv"+str(i)
    
        scaled_train_df, scaled_valid_df = ut.normalize_data(cv_dict[cv_str]['train'], cv_dict[cv_str]['valid'],
                                                        method=norm_str, feature_list=col_list)
        scaled_train_df, scaled_test_df = ut.normalize_data(cv_dict[cv_str]['train'], cv_dict[cv_str]['test'],
                                                        method=norm_str, feature_list=col_list)
        ## fill colmean
        if scaled_train_df.isnull().sum().sum() > 0:
            print('WARNNING! after scaling, train data has #missing values={:}'.format(scaled_train_df.isnull().sum().sum()))
            scaled_train_df = scaled_train_df.apply(lambda x:x.fillna(x.mean()),axis=0)
        if scaled_valid_df.isnull().sum().sum() > 0:
            print('WARNNING! after scaling, valid data has #missing values={:}'.format(scaled_valid_df.isnull().sum().sum()))
            scaled_valid_df = scaled_valid_df.apply(lambda x:x.fillna(x.mean()),axis=0)
        if scaled_test_df.isnull().sum().sum() > 0:
            print('WARNNING! after scaling, test data has #missing values={:}'.format(scaled_test_df.isnull().sum().sum()))
            scaled_test_df = scaled_test_df.apply(lambda x:x.fillna(x.mean()),axis=0)

        # update cv_dict
        cv_dict[cv_str]['train'] = scaled_train_df
        cv_dict[cv_str]['valid'] = scaled_valid_df
        cv_dict[cv_str]['test'] = scaled_test_df
        
    # cross validation
    for i in range(0, args.cv_int):
        cv_str = "cv"+str(i)
        # get data
        scaled_train_df = cv_dict[cv_str]['train']
        scaled_valid_df = cv_dict[cv_str]['valid']
        scaled_test_df = cv_dict[cv_str]['test']
        print("Fold={:} | train={:} | valid={:} | test={:}".format(
              cv_str, scaled_train_df.shape, scaled_valid_df.shape, scaled_test_df.shape))

        # perform function
        print('Performing cross-validation, using parameters=\n{:}'.format(param_dict))
        pred_df = run_cv(scaled_train_df, scaled_valid_df, scaled_test_df, cv_str=cv_str, task_str=args.task_str, 
                         pretrained_model=args.pretrained_str, param_dict=param_dict, fout_str=args.output_path)
        pred_df.to_csv(args.output_path+"."+cv_str+".Prediction.txt",header=True,index=True,sep="\t")
    
        # get evaluation metrices
        fout_str=args.output_path
        record_list = []
        if args.task_str == 'classification':
            auc, auprc, accuracy, recall, precision, f1, mcc = mts.get_clf_metrics(pred_df)
            record_list.append( (fout_str, auc, auprc, accuracy, recall, precision, f1, mcc) )
            col_list = ['Name', 'AUC', 'AUPRC', 'Accuracy', 'Recall', 'Precision', 'F1', 'MCC']
        else:
            mae, mse, rmse, r_square, pcc, spearman = mts.get_reg_metrics(pred_df)
            record_list.append( (fout_str, mae, mse, rmse, r_square, pcc, spearman) )
            col_list = ['Name', 'MAE', 'MSE', 'RMSE', 'R-square', 'PCC', 'Spearman']
    
        # report metrices 
        mts_df = pd.DataFrame.from_records(record_list, columns=col_list)
        mts_df.to_csv(args.output_path+"."+cv_str+".Metrices.txt",header=True,index=False,sep="\t")
        print(mts_df)

    # finished
    print("[Finished in {:}]".format(ut.cal_time(datetime.now(), start)))
