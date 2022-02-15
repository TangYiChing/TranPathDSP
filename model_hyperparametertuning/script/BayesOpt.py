"""
Search the best parameters

"""
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

import util as ut
import metrices as mts


from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.pipeline import make_pipeline

import sklearn.model_selection as skms
import sklearn.metrics as skmts
import sklearn.preprocessing as skpre


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from bayes_opt import BayesianOptimization

MODEL_PARAM_DICT = {"learning_rate": (0.00001, 0.1),
                    "dropout_float": (0.0, 0.5),
                    "batch_size": (32, 64),
                    "epoch": (50, 100),
                    "earlyStop": (10, 50), 
                    "n_neuron": (32, 512),
                    "n_layer": (1, 5)}





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

def get_classifier_optimizer(optimizer_str='adam', learning_rate=0.01):
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


def create_regression(train_df, valid_df, test_df, pretrained=None, pb=MODEL_PARAM_DICT, retrain=True, use_layer='hidden1', n_layer=1, seed_int=42, activation_str='tanh', optimizer_str='adamax', fout_str='./'):
    """
    return the best parameters
    """
    tf.keras.backend.clear_session() # avoid clutter from old model
    # get data
    train_X_arr, train_y_arr = train_df.iloc[:, :-1].values, train_df.iloc[:,-1].values
    valid_X_arr, valid_y_arr = valid_df.iloc[:, :-1].values, valid_df.iloc[:,-1].values
    test_X_arr, test_y_arr = test_df.iloc[:, :-1].values, test_df.iloc[:,-1].values

    def regressor(batch_size=32, dropout_float=0.1, epoch=50, n_neuron=1024, learning_rate=0.1, earlyStop=10, n_layer=1):
        # get parameters
        param_dict = {}
        param_dict["batch_size"] = round(batch_size)
        param_dict["dropout_float"] = float(dropout_float)
        param_dict["epoch"] = round(epoch)
        param_dict["n_neuron"] = round(n_neuron)
        param_dict["learning_rate"] = float(learning_rate)
        param_dict["earlyStop"] = round(earlyStop)
        param_dict["n_layer"] = round(n_layer)
        print('regressor param_dict={:}'.format(param_dict))
        # load pretrained model
        base_model = keras.models.load_model(pretrained)
        for layer in base_model.layers:
            layer.trainable = retrain #True #False # freeze weights

        model_input = base_model.get_layer(use_layer).output

        # add new hidden layers
        if param_dict["n_layer"] == 0:
            model_output = Dense(1, activation="linear", name="outputs")(model_input)
       
        elif param_dict["n_layer"] == 1:

            FC1 = Dense(param_dict['n_neuron'], activation=activation_str, kernel_initializer="glorot_normal", name="FC1")(model_input)
            FC1 = Dropout(param_dict["dropout_float"], name="dropoutFC1")(FC1)

            model_output = Dense(1, activation="linear", name="outputs")(FC1)
        else:

            FC1 = Dense(param_dict['n_neuron'], activation=activation_str, kernel_initializer="glorot_normal", name="FC1")(model_input)
            FC1 = Dropout(param_dict["dropout_float"], name="dropoutFC1")(FC1)

            FC2 = Dense(int(param_dict['n_neuron']/2), activation=activation_str, kernel_initializer="glorot_normal", name="FC2")(FC1)
            FC2 = Dropout(param_dict["dropout_float"], name="dropoutFC2")(FC2)

            model_output = Dense(1, activation="linear", name="outputs")(FC2)

        # create model
        model = Model(inputs=base_model.inputs, outputs=model_output)

        # compile model
        optimizer = get_optimizer(optimizer_str=optimizer_str, learning_rate=float(param_dict["learning_rate"]))
        model.compile(loss='mean_squared_error', optimizer=optimizer,
                      metrics=[keras.metrics.MeanSquaredError()])
        print(model.summary())

        # fit model
        model_h5_str = fout_str + '.bayes_opt.h5'
        callback_list = [EarlyStopping(monitor='val_loss', mode='auto', patience = param_dict["earlyStop"]),
                         ModelCheckpoint((model_h5_str), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
                        ]
        history = model.fit(train_X_arr, train_y_arr,
                            epochs=param_dict["epoch"], shuffle=True, batch_size=param_dict["batch_size"],verbose=0,
                            validation_data=(valid_X_arr, valid_y_arr),callbacks=callback_list)
        
        # return score of validation
        #mse = np.min(history.history['val_mean_squared_error'])
        #print('RMSE={:}'.format(history.history['val_mean_squared_error']))
        #return -(mse) #because bo perform maximization
  
        # get score on test
        pred_arr = model.predict(test_X_arr).flatten()
        rmse_float = np.sqrt(skmts.mean_squared_error(test_y_arr, pred_arr))
        tf.keras.backend.clear_session() # avoid clutter from old model
        print('-rmse={:}'.format(-rmse_float))
        return -rmse_float # Bayesian is a maximization method, so minimize mse = maximize -(mse)


    # optimize hyperparameters
    #    init_points is the number of initial points to start with.
    #    n_iter is the number of iteration.
    #              This optimizer.maximize hold the state so whenever you execute it, it will continue from the last iteration.
    BO = BayesianOptimization(f=regressor, pbounds=pb, verbose=2, random_state=seed_int)
    BO.maximize(init_points=10, n_iter=5)
    best_param_dict = BO.max
    return best_param_dict

def create_classifier(train_df, valid_df, test_df, pretrained=None, pb=MODEL_PARAM_DICT, retrain=True, use_layer='hidden1', n_layer=1, seed_int=42, activation_str='tanh', optimizer_str='adamax', fout_str='./'):
    """
    return the best parameters
    """
    tf.keras.backend.clear_session() # avoid clutter from old model

    ########################################################################
    # balance ratio of  positive:negative for training data
    #     upsampling minority class to double-size,
    #     and undersampling majority class to the same size as the minority
    ########################################################################
    n_pos = np.count_nonzero(train_df.iloc[:,-1]==1)
    n_neg = np.count_nonzero(train_df.iloc[:,-1]==0)
    ratio = 3
    print("    Before, #1={:} | #0={:}".format(n_pos, n_neg))
    if n_pos/n_neg >= ratio:
        major_int = n_pos
        minor_int = n_neg
        pipe = make_pipeline( RandomOverSampler(sampling_strategy={0:minor_int*ratio}, random_state=seed_int),
                              RandomUnderSampler(sampling_strategy={1:minor_int*ratio}, random_state=seed_int))

        #pipe = make_pipeline( SMOTE(sampling_strategy={0:minor_int*3}, random_state=args.seed_int),
        #                      NearMiss(sampling_strategy={1:minor_int*3}))
    elif n_neg/n_pos >= ratio:
        major_int = n_neg
        minor_int = n_pos
        pipe = make_pipeline( RandomOverSampler(sampling_strategy={1:minor_int*ratio}, random_state=seed_int),
                              RandomUnderSampler(sampling_strategy={0:minor_int*ratio}, random_state=seed_int))
        #pipe = make_pipeline( SMOTE(sampling_strategy={1:minor_int*3}, random_state=args.seed_int),
        #                      NearMiss(sampling_strategy={0:minor_int*3}))
    #elif n_pos/n_neg < 3 or n_neg/n_pos < 3:
    else:
        pipe = make_pipeline( SMOTE(random_state=seed_int) )

    train_X_df, train_y_df = pipe.fit_resample(train_df.iloc[:, :-1], train_df.iloc[:, -1])
    print("    After, #1={:} | #0={:}".format(np.count_nonzero(train_y_df==1), np.count_nonzero(train_y_df==0)))
    train_X_arr, train_y_arr = train_X_df.values, train_y_df.values

    # get data
    #train_X_arr, train_y_arr = train_df.iloc[:, :-1].values, train_df.iloc[:,-1].values
    valid_X_arr, valid_y_arr = valid_df.iloc[:, :-1].values, valid_df.iloc[:,-1].values
    test_X_arr, test_y_arr = test_df.iloc[:, :-1].values, test_df.iloc[:,-1].values

    def classifier(batch_size=32, dropout_float=0.1, epoch=50, n_neuron=1024, learning_rate=0.1, earlyStop=10, n_layer=1):
        # get parameters
        param_dict = {}
        param_dict["batch_size"] = int(batch_size)
        param_dict["dropout_float"] = float(dropout_float)
        param_dict["epoch"] = int(epoch)
        param_dict["n_neuron"] = int(n_neuron)
        param_dict["learning_rate"] = float(learning_rate)
        param_dict["earlyStop"] = int(earlyStop)
        param_dict["n_layer"] = round(n_layer)

        # load pretrained model
        base_model = keras.models.load_model(pretrained)
        for layer in base_model.layers:
            layer.trainable = retrain #True #False # freeze weights
        

        model_input = base_model.get_layer(use_layer).output

        # add new hidden layers
        #print("n_layer={:}".format(param_dict["n_layer"]))
        if param_dict["n_layer"] == 0:
            model_output = Dense(1, activation="sigmoid", name="outputs")(model_input)

        elif param_dict["n_layer"] == 1:

            FC1 = Dense(param_dict['n_neuron'], activation=activation_str, kernel_initializer="glorot_normal", name="FC1")(model_input)
            FC1 = Dropout(param_dict["dropout_float"], name="dropoutFC1")(FC1)

            model_output = Dense(1, activation="sigmoid", name="outputs")(FC1)
        else:

            FC1 = Dense(param_dict['n_neuron'], activation=activation_str, kernel_initializer="glorot_normal", name="FC1")(model_input)
            FC1 = Dropout(param_dict["dropout_float"], name="dropoutFC1")(FC1)

            FC2 = Dense(int(param_dict['n_neuron']/2), activation=activation_str, kernel_initializer="glorot_normal", name="FC2")(FC1)
            FC2 = Dropout(param_dict["dropout_float"], name="dropoutFC2")(FC2)

            model_output = Dense(1, activation="sigmoid", name="outputs")(FC2)

        # create model
        model = Model(inputs=base_model.inputs, outputs=model_output)

        # compile model
        optimizer = get_classifier_optimizer(optimizer_str=optimizer_str, learning_rate=float(param_dict["learning_rate"]))
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[keras.metrics.AUC()])
        #print(model.summary())

        # fit model
        model_h5_str = fout_str + '.bayes_opt.h5'
        callback_list = [EarlyStopping(monitor='val_loss', mode='auto', patience = param_dict["earlyStop"]),
                         ModelCheckpoint((model_h5_str), verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
                        ]
        history = model.fit(train_X_arr, train_y_arr,
                            epochs=param_dict["epoch"], shuffle=True, batch_size=param_dict["batch_size"],verbose=0,
                            validation_data=(valid_X_arr, valid_y_arr),callbacks=callback_list)

        # get score of validation
        #key_auc = ''
        #for k in history.history.keys():
        #    if 'val_auc' in k:
        #        key_auc = k
        #auc = np.max(history.history[key_auc])
        #print("Valid AUC={:}".format(auc))
        #return auc

        # get score on test
        pred_arr = model.predict(test_X_arr).flatten()
        auc_float = skmts.roc_auc_score(test_y_arr, pred_arr)
        tf.keras.backend.clear_session() # avoid clutter from old model
        print("AUC={:}".format(auc_float))
        return auc_float
        
        # get score of val_loss
        #val_loss = np.min(history.history['val_loss'])
        #opt_score = 1.0 - val_loss 
        #print('1-val_loss={:}'.format(opt_score))
        #return opt_score


    # optimize hyperparameters
    #    init_points is the number of initial points to start with.
    #    n_iter is the number of iteration. 
    #              This optimizer.maximize hold the state so whenever you execute it, it will continue from the last iteration.
    BO = BayesianOptimization(f=classifier, pbounds=pb, verbose=2, random_state=seed_int)
    BO.maximize(init_points=30, n_iter=10) # init_points=30, n_iter=10
    best_param_dict = BO.max
    print('BO={:}'.format(best_param_dict))
    return best_param_dict


def parse_parameter():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters of a tumor model')
    parser.add_argument("-train", "--train_path",
                        help = "path to train data (i.e., .pkl)")
    parser.add_argument("-valid", "--valid_path",
                        help = "path to valid data (i.e., .pkl)")
    parser.add_argument("-test", "--test_path",
                        help = "path to test data (i.e., .pkl)")
    parser.add_argument("-f", "--feature_str",
                        default = "PID_REACTOME",
                        choices = ["PID", "REACTOME", "PID_REACTOME"],
                        help = "string representing pathway features to be used. default=REACTOME")
    parser.add_argument("-norm", "--normalization_str",
                        default = "standard",
                        choices = ["standard", "minmax"],
                        help = "normalization methods")
    parser.add_argument("-g", "--gpu_str",
                        default = '0',
                        type = str,
                        help = "gpu device ids for CUDA_VISIBLE_DEVICES")
    parser.add_argument("-s", "--seed_int",
                        default = 42,
                        type = int,
                        help = "seed for reproducibility. default=42")
    parser.add_argument("-task", "--task_str",
                        required = True,
                        choices = ["regression", "classification"],
                        help = "string representing job task")
    parser.add_argument("-pretrained", "--pretrained_str",
                        required = True,
                        help = "path to a pre-trained cell line model")
    parser.add_argument("-o", "--output_path",
                        required = True,
                        help = "path to ouput file")
    return parser.parse_args()


if __name__ == "__main__":
    # get args
    args = parse_parameter()
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
    train_df = pd.read_pickle(args.train_path)
    valid_df = pd.read_pickle(args.valid_path)
    test_df = pd.read_pickle(args.test_path)
    data_dict = {'train':train_df, 'valid':valid_df, 'test':test_df}
    for d, df in data_dict.items():
        if df.isnull().sum().sum() > 0:
            print('{:} contains #missing values (i.e., NAN)={:}'.format(
                    d, df.isnull().sum().sum()))

    # subsetting to include wanted pathway features source: PID, REACTOME, or both
    for d, df in data_dict.items():
        data_df = ut.subsetting_feature_column(df, feature_str=args.feature_str)
        data_dict[d] = data_df

    # scaling data
    norm_str = args.normalization_str
    col_list = ['CHEM', 'DGNet', 'EXP'] # these features will be scaled separately
    scaled_train_df, scaled_valid_df = ut.normalize_data(data_dict['train'], data_dict['valid'],
                                                        method=norm_str, feature_list=col_list)
    scaled_train_df, scaled_test_df = ut.normalize_data(data_dict['train'], data_dict['test'],
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



    if args.task_str == 'classification':
        # binarize response
        scaled_train_df['Response'] = scaled_train_df['Response'].replace(to_replace={'R':1, 'NR':0})
        scaled_valid_df['Response'] = scaled_valid_df['Response'].replace(to_replace={'R':1, 'NR':0})
        scaled_test_df['Response'] = scaled_test_df['Response'].replace(to_replace={'R':1, 'NR':0})


    print("train=\n{:}".format(scaled_train_df))
    print("valid=\n{:}".format(scaled_valid_df))
    print("test=\n{:}".format(scaled_test_df))



    # report the best record with smallest loss score
    best_auc = 0
    best_rmse = -1
    best_param_dict = {}

    # search best parameters
    act_list = ['tanh', 'relu', 'elu']
    opt_list = ['adam', 'adamax', 'adadelta', 'rmsprop']
    layer_list = ['hidden1', 'hidden2', 'hidden3', 'hidden4']
    retrain_list = [True, False]

    df_list = [] # collect scores
    for act in act_list:
        for opt in opt_list:
            for layer in layer_list:
                for retrain in retrain_list:
                    if args.task_str == 'classification':
                        max_dict = create_classifier(scaled_train_df, scaled_valid_df, scaled_test_df, pb=MODEL_PARAM_DICT, seed_int = args.seed_int,
                                    pretrained=args.pretrained_str, retrain=retrain, use_layer=layer, activation_str=act, optimizer_str=opt,
                                    fout_str=args.output_path)
                        # append to dict
                        param_dict = max_dict['params']
                        param_dict['target'] = max_dict['target']
                        param_dict['optimizer'] = opt
                        param_dict['activation'] = act
                        param_dict['feature'] = args.feature_str
                        param_dict['use_layer'] = layer
                        param_dict['retrain'] = retrain
                        # compare to the best
                        auc = max_dict['target']
                        if auc > best_auc:
                            best_auc = auc
                            best_param_dict = param_dict
                            print("the best so far={:}".format(best_param_dict))
                        # dict2df
                        col_name_str = os.path.basename(args.train_path)
                        df = pd.DataFrame(list(best_param_dict.items()),columns = ['param',col_name_str])
                        df_list.append(df.set_index(['param']))


                    else:
                        print("ACT={:} | OPT={:} | Layer={:} | non-trainable={:}".format(act, opt, layer, retrain))
                        max_dict = create_regression(scaled_train_df, scaled_valid_df, scaled_test_df, pb=MODEL_PARAM_DICT, seed_int = args.seed_int,
                                    pretrained=args.pretrained_str, retrain=retrain, use_layer=layer, activation_str=act, optimizer_str=opt,
                                    fout_str=args.output_path)
                        # append to dict
                        param_dict = max_dict['params']
                        param_dict['target'] = np.abs(max_dict['target'])
                        param_dict['optimizer'] = opt
                        param_dict['activation'] = act
                        param_dict['feature'] = args.feature_str
                        param_dict['use_layer'] = layer
                        param_dict['retrain'] = retrain
                        # compare to the best
                        #rmse = np.abs(max_dict['target'])
                        neg_rmse = max_dict['target']
                        if neg_rmse > best_rmse:
                            best_rmse = neg_rmse
                            best_param_dict = param_dict
                            print("the best so far={:}".format(best_param_dict))
                        # dict2df
                        col_name_str = os.path.basename(args.train_path)
                        df = pd.DataFrame(list(best_param_dict.items()),columns = ['param',col_name_str])
                        df_list.append(df.set_index(['param']))
    # merge
    all_df = pd.concat(df_list, axis=1)
    print('all results=\n{:}'.format(all_df))

    # dict2df
    col_name_str = os.path.basename(args.train_path)
    df = pd.DataFrame(list(best_param_dict.items()),columns = ['param',col_name_str])
    print('the best parameter set=\n{:}'.format(df))

    # save to file
    df.to_csv(args.output_path+'.bayes_opt.best_params.txt', header=True, index=False, sep="\t")
    all_df.to_csv(args.output_path+'.bayes_opt.all_results.txt', header=True, index=True, sep="\t")
    # finished
    print("[Finished in {:}]".format(ut.cal_time(datetime.now(), start)))
