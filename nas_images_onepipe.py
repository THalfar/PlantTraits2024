import pandas as pd 
import matplotlib.pyplot as plt
import os 
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
import numpy as np
import gc

pickle_file_path = './data/test_df.pickle'

with open(pickle_file_path, 'rb') as f:
    test_df = pickle.load(f)

pickle_file_path = './data/train_df.pickle'

with open(pickle_file_path, 'rb') as f:
    train_df = pickle.load(f)
    

study_name = '425_convnet_003_993_avg'

mean_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']



from sklearn.preprocessing import StandardScaler, RobustScaler

print(train_df['fold'].value_counts())

# scaler = StandardScaler() # TODO testaa robustscaler
scaler = RobustScaler()

sample_df = train_df.copy()
train_df = sample_df[sample_df.fold != 1]
valid_df = sample_df[sample_df.fold == 1]
print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")



X_train = np.stack(train_df['features'].values)
y_train = train_df[mean_columns]

X_valid = np.stack(valid_df['features'].values)
y_valid = valid_df[mean_columns]

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from optuna.integration import TFKerasPruningCallback
import optuna
from keras import regularizers, layers, optimizers, initializers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from datetime import timedelta
import time
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(f'Current GPU allocator: {os.getenv("TF_GPU_ALLOCATOR")}')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print(f'Setting memory growth for {gpu}')
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


np.seterr(over='ignore')

def r2_score_safe(y_true, y_pred):
    # Turvallinen R2 laskenta, joka palauttaa -inf, jos laskennassa ilmenee virheitä
    try:
        return r2_score(y_true, y_pred)
    except Exception as e: 
        # print(f'Error in r2_score_safe: {e}')        
        return float('-inf')

def mean_squared_error_safe(y_true, y_pred):
    # Turvallinen MSE laskenta, joka palauttaa inf, jos laskennassa ilmenee virheitä
    try:
        return mean_squared_error(y_true, y_pred)
    except Exception as e:
        # print(f'Error in mean_squared_error_safe: {e}')
        return float('-inf')

def mean_absolute_error_safe(y_true, y_pred):
    # Turvallinen MAE laskenta, joka palauttaa inf, jos laskennassa ilmenee virheitä
    try:
        return mean_absolute_error(y_true, y_pred)
    except Exception as e:
        # print(f'Erros in mean_absolute_error_safe: {e}')
        return float('-inf')

def r2_score_tf(y_true, y_pred):

    try: 
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)), axis=0)
        r2 = 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())
        r2 = tf.where(tf.math.is_nan(r2), tf.zeros_like(r2), r2) 
        return tf.reduce_mean(tf.maximum(r2, 0.0))
    except Exception as e:
        # print(f'Error in r2_score_tf: {e}')
        return float('-inf')
    

def create_model(trial):

    image_input = Input(shape=(X_train.shape[1],), name='image_input')
    

    img_dense = image_input

    max_units = 1000
    num_layers = trial.suggest_int('layers', 1, 3)

    for i in range(num_layers):

        init = trial.suggest_categorical(f'init_{i}', choices = ['glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform',  'random_normal', 'random_uniform'])
        activation = trial.suggest_categorical(f'act_{i}', choices = ['relu', 'tanh', 'selu', 'LeakyReLU', 'swish', 'elu', 'sigmoid'])
        drop = trial.suggest_float(f'D_{i}', 0.0, 0.9)
        norm = trial.suggest_categorical(f'N_{i}', choices = ['On', 'Off'])
        drop_on = trial.suggest_categorical(f'D_in_{i}', choices = ['On', 'Off'])
        units = trial.suggest_int(f'num_{i}', 32, max_units)

        img_dense = Dense(units, activation=activation, kernel_initializer = init)(img_dense)
        if norm == 'On':
            img_dense = layers.BatchNormalization()(img_dense)
        if drop_on == 'On':
            img_dense = Dropout(drop)(img_dense)
          
        max_units = min(max_units, units)
        
    output = Dense(6, activation='linear')(img_dense)
    model = Model(inputs=image_input, outputs=output)

    optimizer_options = ['adam', 'rmsprop', 'adamax', 'Ftrl']
    optimizer_selected = trial.suggest_categorical('optimizer', optimizer_options)

    lr = trial.suggest_float('lr', 1e-4, 1e-1, log = True)

    if optimizer_selected == 'adam':
        optimizer = optimizers.Adam(learning_rate = lr)
    elif optimizer_selected == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate = lr)        
    elif optimizer_selected == 'Ftrl':
        optimizer = optimizers.Ftrl(learning_rate = lr)
    else:
        optimizer = optimizers.Adamax(learning_rate = lr)
        
    loss_function = trial.suggest_categorical('loss', ['mse', 'mae', 'mape'])

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mse','mae', 'mape' , r2_score_tf])
    
    
    return model

 

def objective(trial):

    model = create_model(trial)

    y_train_transformed = y_train.copy()
    y_valid_transformed = y_valid.copy()


    log_base_options = {'none': None, 'log2': 2, 'log10': 10, 'log5': 5, 'log15': 15, 'sqrt': 'sqrt', 'cbrt': 'cbrt', 'log20': 20, 'log30' : 30}
    log_transforms = {}
    for target in mean_columns:
        log_base = trial.suggest_categorical(f'Log_{target}', list(log_base_options.keys()))
        log_transforms[target] = log_base_options[log_base]

    if trial.params['loss'] == 'mape':
        callbacks = [
                 ReduceLROnPlateau('val_mape', patience=2, factor=0.7, mode = 'min', verbose = 0)]
    elif trial.params['loss'] == 'mse':
        callbacks = [
                 ReduceLROnPlateau('val_mse', patience=2, factor=0.7, mode = 'min', verbose = 0)]
    else:
        callbacks = [
                 ReduceLROnPlateau('val_mae', patience=2, factor=0.7, mode = 'min', verbose = 0)]
        
    
    # callbacks = [
    #              ReduceLROnPlateau('val_r2_score_tf', patience=2, factor=0.7, mode = 'max', verbose = 0)]

    for target, log_base in log_transforms.items():
        if log_base is not None and log_base != 'sqrt' and log_base != 'cbrt':
            y_train_transformed[target] = np.log(y_train[target]) / np.log(log_base)
            y_valid_transformed[target] = np.log(y_valid[target]) / np.log(log_base)

        elif log_base == 'sqrt':
            y_train_transformed[target] = np.sqrt(y_train[target])
            y_valid_transformed[target] = np.sqrt(y_valid[target])

        elif log_base == 'cbrt':
            y_train_transformed[target] = np.cbrt(y_train[target])
            y_valid_transformed[target] = np.cbrt(y_valid[target])

        else:
            y_train_transformed[target] = y_train[target]
            y_valid_transformed[target] = y_valid[target]
    
    
    scaler_base_options = {'Std': StandardScaler(), 'None': None, 'minmax': MinMaxScaler()}
    scaler_transforms = {}
    for target in mean_columns:
        scaler_base = trial.suggest_categorical(f'Scaler_{target}', list(scaler_base_options.keys()))
        scaler_transforms[target] = scaler_base_options[scaler_base]
    
    for target, scaler in scaler_transforms.items():
        if scaler is not None:
            y_train_transformed[target] = scaler.fit_transform(y_train_transformed[target].values.reshape(-1, 1)).flatten()
            y_valid_transformed[target] = scaler.transform(y_valid_transformed[target].values.reshape(-1, 1)).flatten()

    new_best = None
    new_best_found = False
    for epoch in range(17):

        model.fit(X_train, y_train_transformed, validation_data=(X_valid, y_valid_transformed), batch_size=256, epochs=3, callbacks=callbacks, verbose = 0)
        preds_transformed = model.predict(X_valid, verbose = 0)        

        try:        
            for i, target in enumerate(mean_columns):
                scaler = scaler_transforms[target]
                if scaler is not None:
                    preds_transformed[:, i] = scaler.inverse_transform(preds_transformed[:, i].reshape(-1, 1)).flatten()                

            for i, target in enumerate(mean_columns):
                log_base = log_transforms[target]
                if log_base is not None and log_base != 'sqrt' and log_base != 'cbrt':
                    preds_transformed[:, i] = np.power(log_base, preds_transformed[:, i])                
                elif log_base == 'sqrt':   
                    preds_transformed[:, i] = np.square(preds_transformed[:, i])    
                elif log_base == 'cbrt':
                    preds_transformed[:, i] = np.power(preds_transformed[:, i], 3)

            r2_score_inv = r2_score_safe(y_valid, preds_transformed)                        

        except Exception as e:
            # print(f'Error in inverse transformation: {e}')
            # print(f'Trial number {trial.number} epoch {epoch}')
            r2_score_inv = float('-inf')
            
        if np.isnan(preds_transformed).any():
            # print(f'Nan values in predictions')
            # print(f'Trial number {trial.number} epoch {epoch}')
            r2_score_inv = float('-inf')                        

        if np.isinf(preds_transformed).any():
            # print(f'Inf values in predictions')
            # print(f'Trial number {trial.number} epoch {epoch}')
            r2_score_inv = float('-inf')

        trial.report(r2_score_inv, epoch)

        if r2_score_inv == float('-inf'):
            print('---')
            print(f'Trial {trial.number} failed at epoch {epoch} value : {r2_score_inv}')
            print('---')
            tf.keras.backend.clear_session()
            gc.collect()
            return r2_score_inv
        
        if new_best is None:
            new_best = r2_score_inv
        elif r2_score_inv > new_best:
            new_best = r2_score_inv
        
        if trial.should_prune():

            print('---')
            print(f'Trial {trial.number} pruned at epoch {epoch} with R2 {r2_score_inv:.5f}')
            print('---')

            if trial.number > 0:
                if r2_score_inv > study.best_value:

                    print("#" * 50)
                    print("*" * 50)
                    print(f'Old best R2 : {study.best_value:.5f}')
                    print(f'New best R2 : {r2_score_inv:.5f}')

                    r2 = r2_score_safe(y_valid, preds_transformed)
                    mse  = mean_squared_error(y_valid, preds_transformed)
                    mae = mean_absolute_error(y_valid, preds_transformed)
                    mape = mean_absolute_percentage_error(y_valid, preds_transformed)
                    
                    print(f'Best epoch all errors R2 : {r2:.5f}, MSE : {mse:.5f}, MAE : {mae:.5f}, MAPE : {mape:.5f}')
                    print(f'Best epoch : {epoch}')

                    best_filename = f'./NN_search/{study_name}_best_val_{r2_score_inv:.5f}_model.h5'
                    if os.path.exists(best_filename):
                        os.remove(best_filename)

                    print(f'Saving model to {best_filename}')
                    model.save(best_filename)
            
                    best_log_transforms_name = f'./NN_search/{study_name}_{r2_score_inv:.5f}_best_log_transforms.pickle'
                    print(f'Saving log transforms to {best_log_transforms_name}')
                    with open(best_log_transforms_name, 'wb') as f:
                        pickle.dump(log_transforms, f, protocol=pickle.HIGHEST_PROTOCOL)

                    scaler_transforms_name = f'./NN_search/{study_name}_{r2_score_inv:.5f}_best_scalers.pickle'
                    print(f'Saving scalers to {scaler_transforms_name}')
                    with open(scaler_transforms_name, 'wb') as f:
                        pickle.dump(scaler_transforms, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print("*" * 50)
                
            tf.keras.backend.clear_session()
            gc.collect()
                
            return new_best
        
        if trial.number > 0 and not new_best_found:
            if r2_score_inv > study.best_value:
                new_best_found = True
            
                print("#" * 50)                
                       
                print(f'Old best R2 : {study.best_value:.5f}')
                print(f'New best R2 : {r2_score_inv:.5f}')

                r2 = r2_score_safe(y_valid, preds_transformed)
                mse  = mean_squared_error(y_valid, preds_transformed)
                mae = mean_absolute_error(y_valid, preds_transformed)
                mape = mean_absolute_percentage_error(y_valid, preds_transformed)
                
                print(f'Best epoch all errors R2 : {r2:.5f}, MSE : {mse:.5f}, MAE : {mae:.5f}, MAPE : {mape:.5f}')
                print(f'Best epoch : {epoch}')

                best_filename = f'./NN_search/{study_name}_best_val_{r2_score_inv:.5f}_model.h5'
                if os.path.exists(best_filename):
                    os.remove(best_filename)

                print(f'Saving model to {best_filename}')
                model.save(best_filename)
        
                best_log_transforms_name = f'./NN_search/{study_name}_{r2_score_inv:.5f}_best_log_transforms.pickle'
                print(f'Saving log transforms to {best_log_transforms_name}')
                with open(best_log_transforms_name, 'wb') as f:
                    pickle.dump(log_transforms, f, protocol=pickle.HIGHEST_PROTOCOL)

                scaler_transforms_name = f'./NN_search/{study_name}_{r2_score_inv:.5f}_best_scalers.pickle'
                print(f'Saving scalers to {scaler_transforms_name}')
                with open(scaler_transforms_name, 'wb') as f:
                    pickle.dump(scaler_transforms, f, protocol=pickle.HIGHEST_PROTOCOL)

                print("#" * 50)                

    
                
        if new_best_found:
            if r2_score_inv > study.best_value:
                
                print("*" * 50)
                print(f'Getting better R2 : {r2_score_inv:.5f} with a new best in this run :) ' )
                print(f'Old best R2 : {study.best_value:.5f}')
                print(f'New best R2 : {r2_score_inv:.5f}')

                r2 = r2_score_safe(y_valid, preds_transformed)
                mse  = mean_squared_error(y_valid, preds_transformed)
                mae = mean_absolute_error(y_valid, preds_transformed)
                mape = mean_absolute_percentage_error(y_valid, preds_transformed)
                
                print(f'Best epoch all errors R2 : {r2:.5f}, MSE : {mse:.5f}, MAE : {mae:.5f}, MAPE : {mape:.5f}')
                print(f'Best epoch : {epoch}')

                best_filename = f'./NN_search/{study_name}_best_val_{r2_score_inv:.5f}_model.h5'
                if os.path.exists(best_filename):
                    os.remove(best_filename)

                print(f'Saving model to {best_filename}')
                model.save(best_filename)
        
                best_log_transforms_name = f'./NN_search/{study_name}_{r2_score_inv:.5f}_best_log_transforms.pickle'
                print(f'Saving log transforms to {best_log_transforms_name}')
                with open(best_log_transforms_name, 'wb') as f:
                    pickle.dump(log_transforms, f, protocol=pickle.HIGHEST_PROTOCOL)

                scaler_transforms_name = f'./NN_search/{study_name}_{r2_score_inv:.5f}_best_scalers.pickle'
                print(f'Saving scalers to {scaler_transforms_name}')
                with open(scaler_transforms_name, 'wb') as f:
                    pickle.dump(scaler_transforms, f, protocol=pickle.HIGHEST_PROTOCOL)

        
    if os.path.exists(f'./NN_search/{study_name}_search_model.h5'):
        os.remove(f'./NN_search/{study_name}_search_model.h5')

    tf.keras.backend.clear_session()
    gc.collect()
    
    return new_best


num_random_trials = 1
num_gene = 50
num_tpe_trial = 5


search_time_max = 3600 * 18

if os.path.exists(f'./NN_search/{study_name}_pruner.pickle'):
    with open(f'./NN_search/{study_name}_pruner.pickle', 'rb') as f:
        print(f'Loading pruner from file {f}')
        pruner = pickle.load(f)
else:
    print('Creating new pruner')
    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1, interval_steps=3)


study = optuna.create_study(direction='maximize',
                            study_name=study_name,
                            storage=f'sqlite:///425_convnet_003_993.db',
                            load_if_exists=True,
                            pruner=pruner
                            )

### enquing trials

study.enqueue_trial({'Log_X4': 'log15', 'Scaler_X4': 'minmax', 'Log_X11': 'cbrt', 'Scaler_X11': 'none', 'Log_X18': 'log10', 'Scaler_X18': 'none', 'Log_X50': 'log2', 'Scaler_X50': 'none', 'Log_X26': 'log10', 'Scaler_X26': 'Std', 'Log_X3112': 'log30', 'Scaler_X3112': 'none'})
study.enqueue_trial({'Log_X4': 'log15', 'Scaler_X4': 'minmax', 'Log_X11': 'log20', 'Scaler_X11': 'none', 'Log_X18': 'log10', 'Scaler_X18': 'none', 'Log_X50': 'log30', 'Scaler_X50': 'none', 'Log_X26': 'log10', 'Scaler_X26': 'Std', 'Log_X3112': 'log10', 'Scaler_X3112': 'none'})
study.enqueue_trial({'Log_X4': 'log2', 'Scaler_X4': 'none', 'Log_X11': 'cbrt', 'Scaler_X11': 'none', 'Log_X18': 'log10', 'Scaler_X18': 'none', 'Log_X50': 'log10', 'Scaler_X50': 'none', 'Log_X26': 'sqrt', 'Scaler_X26': 'Std', 'Log_X3112': 'log30', 'Scaler_X3112': 'none'})

study.enqueue_trial({'Log_X4': 'log15', 'Scaler_X4': 'minmax', 'Log_X11': 'cbrt', 'Scaler_X11': 'none', 'Log_X18': 'log10', 'Scaler_X18': 'none', 'Log_X50': 'log2', 'Scaler_X50': 'none', 'Log_X26': 'log10', 'Scaler_X26': 'Std', 'Log_X3112': 'log30', 'Scaler_X3112': 'none'})
study.enqueue_trial({'Log_X4': 'log15', 'Scaler_X4': 'minmax', 'Log_X11': 'log20', 'Scaler_X11': 'none', 'Log_X18': 'log10', 'Scaler_X18': 'none', 'Log_X50': 'log30', 'Scaler_X50': 'none', 'Log_X26': 'log10', 'Scaler_X26': 'Std', 'Log_X3112': 'log10', 'Scaler_X3112': 'none'})
study.enqueue_trial({'Log_X4': 'log2', 'Scaler_X4': 'none', 'Log_X11': 'cbrt', 'Scaler_X11': 'none', 'Log_X18': 'log10', 'Scaler_X18': 'none', 'Log_X50': 'log10', 'Scaler_X50': 'none', 'Log_X26': 'sqrt', 'Scaler_X26': 'Std', 'Log_X3112': 'log30', 'Scaler_X3112': 'none'})

study.enqueue_trial({'Log_X4': 'log15', 'Scaler_X4': 'minmax', 'Log_X11': 'cbrt', 'Scaler_X11': 'none', 'Log_X18': 'log10', 'Scaler_X18': 'none', 'Log_X50': 'log2', 'Scaler_X50': 'none', 'Log_X26': 'log10', 'Scaler_X26': 'Std', 'Log_X3112': 'log30', 'Scaler_X3112': 'none'})
study.enqueue_trial({'Log_X4': 'log15', 'Scaler_X4': 'minmax', 'Log_X11': 'log20', 'Scaler_X11': 'none', 'Log_X18': 'log10', 'Scaler_X18': 'none', 'Log_X50': 'log30', 'Scaler_X50': 'none', 'Log_X26': 'log10', 'Scaler_X26': 'Std', 'Log_X3112': 'log10', 'Scaler_X3112': 'none'})
study.enqueue_trial({'Log_X4': 'log2', 'Scaler_X4': 'none', 'Log_X11': 'cbrt', 'Scaler_X11': 'none', 'Log_X18': 'log10', 'Scaler_X18': 'none', 'Log_X50': 'log10', 'Scaler_X50': 'none', 'Log_X26': 'sqrt', 'Scaler_X26': 'Std', 'Log_X3112': 'log30', 'Scaler_X3112': 'none'})


###

search_time_taken = 0
search_start = time.time()
round = 0
trials_done = 0

import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

optuna.logging.get_logger("optuna").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

if os.path.exists(f'./NN_search/{study_name}_genesampler.pickle'):
    with open(f'./NN_search/{study_name}_genesampler.pickle', 'rb') as f:
        print(f'Loading gene sampler from file {f}')
        genemachine = pickle.load(f)
else:
    print('Creating new gene sampler')
    genemachine = optuna.samplers.NSGAIISampler(crossover = optuna.samplers.nsgaii.VSBXCrossover())

if os.path.exists(f'./NN_search/{study_name}_qmc_sampler.pickle'):
    with open(f'./NN_search/{study_name}_qmc_sampler.pickle', 'rb') as f:
        print(f'Loading QMC sampler from file {f}')
        qmcampler = pickle.load(f)
else:
    print(f'Creating new QMC sampler')
    qmcampler = optuna.samplers.QMCSampler(warn_independent_sampling = False)

if os.path.exists(f'./NN_search/{study_name}_tpe_sampler.pickle'):
    with open(f'./NN_search/{study_name}_tpe_sampler.pickle', 'rb') as f:
        print(f'Loading TPE sampler from file {f}')
        tpe_sampler = pickle.load(f)
else:
    print(f'Creating new TPE sampler')
    tpe_sampler = optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True, warn_independent_sampling = False)

if os.path.exists(f'./NN_search/{study_name}_pruner.pickle'):
    with open(f'./NN_search/{study_name}_pruner.pickle', 'rb') as f:
        print(f'Loading pruner from file {f}')
        study.pruner = pickle.load(f)


while search_time_taken < search_time_max:



    round_start = time.time()

    print('=' * 50)
    print(f'Starting study with {num_random_trials} random trials, round {round}')
    print(f'Search time so far taken : {timedelta(seconds=search_time_taken)}')
    print('=' * 50)
    study.sampler = qmcampler
    study.optimize(objective, n_trials=num_random_trials)
    print('-' * 50)
    print(f'Time taken for random trials: {timedelta(seconds= (time.time() - round_start))}')
    print(f'Time for one random trial: {timedelta(seconds= (time.time() - round_start) / num_random_trials)}')
    print('-' * 50)
    
    genetime = time.time()

    print(f'\nStarting gene {num_gene} trials...\n')
    study.sampler = genemachine
    study.optimize(objective, n_trials=num_gene)
    print('-' * 50)
    print(f'Time taken for gene trials: {timedelta(seconds= time.time() - genetime)}')
    print(f'Time for one gene trial: {timedelta(seconds= (time.time() - genetime) / num_gene)}')    
    print('-' * 50)
    time_tpe = time.time() 
    print(f'Starting TPE {num_tpe_trial} trials...')
    study.sampler = tpe_sampler
    study.optimize(objective, n_trials=num_tpe_trial)
    print('-' * 50)
    print(f'Time taken for TPE trials: {timedelta(seconds= time.time() - time_tpe)}')
    print(f'Time for one TPE trial: {timedelta(seconds= (time.time() - time_tpe) / num_tpe_trial)}')
    print('-' * 50)
    
    print(f'Time taken for one trial this round: {timedelta(seconds= (time.time() - round_start) / (num_random_trials + num_tpe_trial + num_gene))}')
    print(f'Time this round: {timedelta(seconds= time.time() - round_start)}')
    
    print('-' * 50)
    trials_done += num_random_trials + num_tpe_trial + num_gene
    print(f'Trials done so far: {trials_done}')
    search_time_taken = time.time() - search_start
    print(f'Time taken for one trials all rounds: {timedelta(seconds= search_time_taken / trials_done)}')
    round += 1

    with open(f'./NN_search/{study_name}_pruner.pickle', 'wb') as f:
        print(f'Saving pruner to file {f}')
        pickle.dump(study.pruner, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'./NN_search/{study_name}_genesampler.pickle', 'wb') as f:
        print(f'Saving gene sampler to file {f}')
        pickle.dump(genemachine, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'./NN_search/{study_name}_qmc_sampler.pickle', 'wb') as f:
        print(f'Saving QMC sampler to file {f}')
        pickle.dump(qmcampler, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'./NN_search/{study_name}_tpe_sampler.pickle', 'wb') as f:
        print(f'Saving TPE sampler to file {f}')
        pickle.dump(tpe_sampler, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'./NN_search/{study_name}_pruner.pickle', 'wb') as f:
        print(f'Saving pruner to file {f}')
        pickle.dump(study.pruner, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        

print(f'Search time total : {timedelta(seconds=time.time() - search_start)}')



        




