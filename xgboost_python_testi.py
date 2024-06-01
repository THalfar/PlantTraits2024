import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import optuna
import pickle
from datetime import timedelta
import time
import os 
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import gc
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

feat = pd.read_csv('./data/test.csv')
FEATURE_COLS = feat.columns[1:].tolist()

study_name = '530_yritys'



pickle_file_path = f'./data/train_df_neljas.pickle'

with open(pickle_file_path, 'rb') as f:
    train_df = pickle.load(f)

train_df.drop(['523_ConvNeXtXLarge_4', '527_Convnextlarge', 'model_features_514_convnextlarge_maxavg_2'], axis=1, inplace=True)

def prepare_features(df, feature_columns, features):

    # print(f'In prepare_features columns: {feature_columns}')
    # print(f'In prepare_features features: {features}')
    # Yhdistää useita sarakkeita, oletetaan että jokainen arvo on listamuodossa tai pienenä NumPy-taulukkona
    if features == "on":
        data = [df[col].values for col in FEATURE_COLS]
    else:
        data = []
    
    if len(feature_columns) != 0:
        combined_features = np.hstack([np.vstack(df[col].values) for col in feature_columns])
        data.append(combined_features)
    
    all_features = np.column_stack(data).tolist()
    
    df['all_features'] = all_features
    return df


import warnings

# Ohita tietyn tyyppiset varoitukset
warnings.filterwarnings('ignore', category=UserWarning)


def get_features_array(features_series):
    # Muuntaa sarjan, joka sisältää taulukoita, yhdeksi 2D-taulukoksi
    return np.array(list(features_series))


def objective(trial, df, target):
    param = {        
        'objective': 'reg:squarederror',
        'device' : 'cuda',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log = True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log = True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log = True),        
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 142),        
        'boosting': 'dart'
        }
    

    # # model_features_514_convnextlarge_maxavg_2	model_features_511_convnextlarge_3	523_ConvNeXtXLarge_4	model_features_523_ConvNeXtXLarge_new	527_Convnextlarge	model_features_527_convnextl_0
    feature_names = ['model_features_511_convnextlarge_3', 'model_features_523_ConvNeXtXLarge_new', 'model_features_527_convnextl_0']
    choosed_features = []

    features = trial.suggest_categorical('features', ['on', 'off'])
    
    for col in feature_names:
        if trial.suggest_categorical(col, ['on', 'off']) == 'on':
            choosed_features.append(col)
    
    print(f'len choosed_features {len(choosed_features)}')

    if features == 'off':
        if len(choosed_features) == 0:
            print('No features selected')
            raise optuna.TrialPruned()

    df_this = prepare_features(df,choosed_features, features)
    
    
    num_total = df_this['all_features'].iloc[0]
    num_total = len(num_total)
    print(f'num_total {num_total}')
    
    folds = [4,3,2,1,0]
    mse_scores = []
    r2_scores = []

    num_boost_round = trial.suggest_int('n_estimators', 10, 1420, log=True) 

    for fold in folds:

    
        train_data = df_this[df_this['fold'] != fold]
        valid_data = df_this[df_this['fold'] == fold]

    
        X_train = get_features_array(train_data['all_features'])
        X_valid = get_features_array(valid_data['all_features'])
        # print(f'Shape X_train {X_train.shape}')

        y_train = train_data[target]
        y_valid = valid_data[target]

        # print(f'Y_train shape {y_train.shape}')

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        # print(f'Done creating DMatrix')

        
        
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        # print(f'Starting training')
        model = xgb.train(param, dtrain, num_boost_round=num_boost_round, evals=watchlist, verbose_eval=False)
        preds = model.predict(dvalid)
        mse = mean_squared_error(y_valid, preds)
        r2 = r2_score(y_valid, preds)

        trial.report(r2, fold)

        if trial.should_prune():
            print(f'Pruned fold {fold} with value {r2} and mse {mse}')
            raise optuna.TrialPruned()

        print(f'Fold {fold} MSE: {mse} R2: {r2}')
        mse_scores.append(mse)
        r2_scores.append(r2)

        del model, dtrain, dvalid, X_train, X_valid, y_train, y_valid, train_data, valid_data
        
        tf.keras.backend.clear_session()
        gc.collect()
    
    
    del df_this
    tf.keras.backend.clear_session()
    gc.collect()
    
    return np.mean(r2_scores)
    

def optimize_model(df, target):

    if os.path.exists(f'./NN_search/{study_name}_{target}_qmc_sampler.pickle'):
        with open(f'./NN_search/{study_name}_{target}_qmc_sampler.pickle', 'rb') as f:
            print(f'Loading QMC sampler from file {f}')
            qmc_sampler = pickle.load(f)
    else:
        print(f'Creating new QMC sampler')
        qmc_sampler = optuna.samplers.QMCSampler(warn_independent_sampling = False)

    if os.path.exists(f'./NN_search/{study_name}_{target}_tpe_sampler.pickle'):
        with open(f'./NN_search/{study_name}_{target}_tpe_sampler.pickle', 'rb') as f:
            print(f'Loading TPE sampler from file {f}')
            tpe_sampler = pickle.load(f)
    else:
        print(f'Creating new TPE sampler')
        tpe_sampler = optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True, warn_independent_sampling = False)

    if os.path.exists(f'./NN_search/{study_name}_{target}_pruner.pickle'):
        with open(f'./NN_search/{study_name}_{target}_pruner.pickle', 'rb') as f:
            print(f'Loading pruner from file {f}')
            pruner = pickle.load(f)
    else:
        print(f'Creating new pruner')
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

    start_time = time.time()
    study = optuna.create_study(direction='maximize',
                            study_name=study_name,
                            storage=f'sqlite:///530_yritys_{target}.db', # 530_yritys_{target}.db
                            load_if_exists=True                                    
                            )
    
    print(f'Starting optimization for {target} with qmc sampler')
    random_time = time.time()
    study.sampler = qmc_sampler
    study.optimize(lambda trial: objective(trial, df, target), n_trials=3)
    print(f'QCM optimization finished in {timedelta(seconds=time.time() - random_time)}')

    print(f'Saving QMC sampler to file ./NN_search/{study_name}_{target}_qmc_sampler.pickle')
    with open(f'./NN_search/{study_name}_{target}_qmc_sampler.pickle', 'wb') as f:
        pickle.dump(qmc_sampler, f)

    print(f'Starting optimization for {target} with TPE sampler')
    tpe_time = time.time()
    study.sampler = tpe_sampler
    study.optimize(lambda trial: objective(trial, df, target), n_trials=15)
    print(f'TPE optimization finished in {timedelta(seconds=time.time() - tpe_time)}')

    print(f'Saving TPE sampler to file ./NN_search/{study_name}_{target}_tpe_sampler.pickle')
    with open(f'./NN_search/{study_name}_{target}_tpe_sampler.pickle', 'wb') as f:
        pickle.dump(tpe_sampler, f)

    print(f'Saving pruner to file ./NN_search/{study_name}_{target}_pruner.pickle')
    with open(f'./NN_search/{study_name}_{target}_pruner.pickle', 'wb') as f:
        pickle.dump(pruner, f)

    print(f'Optimization finished in {timedelta(seconds=time.time() - start_time)}')

    
    

target_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']



time_search_start = time.time()
time_taken = 0

while time_taken < 3600 * 18:
    for target in target_columns:    
        print(f'\n\nOptimizing model for {target}\n\n')
        optimize_model(train_df, target)
        time_taken = time.time() - time_search_start
        print(f'Time taken: {timedelta(seconds=time_taken)}')   