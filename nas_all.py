import pandas as pd 
import matplotlib.pyplot as plt
import os 
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
import numpy as np
import gc
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

pickle_file_path = './data/test_df.pickle'

with open(pickle_file_path, 'rb') as f:
    test_df = pickle.load(f)

pickle_file_path = './data/train_df.pickle'

with open(pickle_file_path, 'rb') as f:
    train_df = pickle.load(f)

features = pd.read_csv('./data/test.csv')
FEATURE_COLS = features.columns[1:].tolist()
    

study_name = '423_std_powerlog_3'

mean_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']


# selected_features_pickle_path = './data/selected_features_list.pickle'
# with open(selected_features_pickle_path, 'rb') as f:
#     FEATURE_COLS = pickle.load(f)


train_df_original = train_df.copy()

print(train_df['fold'].value_counts())


scaler = RobustScaler()

sample_df = train_df.copy()
train_df = sample_df[sample_df.fold != 3]
valid_df = sample_df[sample_df.fold == 3]
print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS].values)
valid_df[FEATURE_COLS] = scaler.transform(valid_df[FEATURE_COLS].values)

scaler_tabufeatures_name = f'./NN_search/scaler_tabufeatures_{study_name}_train.pickle'
print(f"Saving scaler to {scaler_tabufeatures_name}")
with open(f'{scaler_tabufeatures_name}', 'wb') as f:
    pickle.dump(scaler, f)

X_train_tab = train_df[FEATURE_COLS].values
X_train_feat = np.stack(train_df['features'].values)
y_train = train_df[mean_columns]

X_valid_tab = valid_df[FEATURE_COLS].values 
X_valid_feat = np.stack(valid_df['features'].values)
y_valid = valid_df[mean_columns]


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

    image_features_input = Input(shape=(X_train_feat.shape[1],), name='image_features_input')
    tabular_data_input = Input(shape=(X_train_tab.shape[1],), name='tabular_data_input')

    img_num_layers = trial.suggest_int('Imgage layers', 1, 3)
    max_img_units = 3000
    img_dense = image_features_input

    image_init = trial.suggest_categorical(f'Img_init', choices = ['glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform',  'random_normal', 'random_uniform'])
    activation_img = trial.suggest_categorical(f'Act_img', choices = ['relu', 'tanh', 'selu', 'LeakyReLU', 'swish', 'elu', 'sigmoid'])
    drop_img = trial.suggest_float(f'Drop_img', 0.0, 0.9)
    batch_norm_img = trial.suggest_categorical(f'Img_BatchN', choices = ['On', 'Off'])

    for i in range(img_num_layers):

        num_img_units = trial.suggest_int(f'Num_img_{i}', 32, max_img_units, log = True)
        img_dense = Dense(num_img_units, activation=activation_img, kernel_initializer = image_init)(img_dense)
        img_dense = Dropout(drop_img)(img_dense)

        max_img_units = min(max_img_units, num_img_units)

    if batch_norm_img == 'On':
        img_dense = layers.BatchNormalization()(img_dense)


    tab_num_layers = trial.suggest_int('Tabular layers', 1, 3)
    max_tab_units = 3000
    tab_dense = tabular_data_input
    tab_init = trial.suggest_categorical(f'Tab_init', choices = ['glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform',  'random_normal', 'random_uniform'])
    activation_tab = trial.suggest_categorical(f'Act_tab', choices = ['relu', 'tanh', 'selu', 'LeakyReLU', 'swish', 'elu', 'sigmoid'])
    drop_tab = trial.suggest_float(f'Drop_tab', 0.0, 0.9)
    batch_norm_tab = trial.suggest_categorical(f'Tab_BatchN', choices = ['On', 'Off'])

    for i in range(tab_num_layers):

        num_tab_units = trial.suggest_int(f'Num_tab_{i}', 16, max_tab_units, log = True)
        tab_dense = Dense(num_tab_units, activation=activation_tab, kernel_initializer = tab_init)(tab_dense)
        tab_dense = Dropout(drop_tab)(tab_dense)

        max_tab_units = min(max_tab_units, num_tab_units)

    if batch_norm_tab == 'On':
        tab_dense = layers.BatchNormalization()(tab_dense)

    concatenated = Concatenate()([img_dense, tab_dense])
    com_num_layers = trial.suggest_int('Concat layers', 1, 3)
    max_com_units = 3000
    con_init = trial.suggest_categorical(f'Con_init', choices = ['glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform', 'random_normal', 'random_uniform'])
    activation_common = trial.suggest_categorical(f'Act_con',  choices = ['relu', 'tanh', 'selu', 'LeakyReLU', 'swish', 'elu', 'sigmoid'])
    drop_common = trial.suggest_float(f'Drop_con', 0.0, 0.9)
    batch_norm_common = trial.suggest_categorical(f'Com_BatchN', ['On', 'Off'])

    for i in range(com_num_layers):

        num_common_units = trial.suggest_int(f'Num_con_{i}', 32, max_com_units, log = True)
        concatenated = Dense(num_common_units, activation=activation_common, kernel_initializer = con_init)(concatenated)
        concatenated = Dropout(drop_common)(concatenated)

        max_com_units = min(max_com_units, num_common_units)

    if batch_norm_common == 'On':
        concatenated = layers.BatchNormalization()(concatenated)

    output = Dense(6, activation='linear')(concatenated)
    model = Model(inputs=[image_features_input, tabular_data_input], outputs=output)

    optimizer_options = ['adam', 'rmsprop', 'adamax', 'Ftrl']
    optimizer_selected = trial.suggest_categorical('optimizer', optimizer_options)

    if optimizer_selected == 'adam':
        optimizer = optimizers.Adam()
    elif optimizer_selected == 'rmsprop':
        optimizer = optimizers.RMSprop()        
    elif optimizer_selected == 'Ftrl':
        optimizer = optimizers.Ftrl()
    else:
        optimizer = optimizers.Adamax()

    
        
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse','mae', 'mape' ,r2_score_tf])
    
    return model

 

def objective(trial):

    model = create_model(trial)

    y_train_transformed = y_train.copy()
    y_valid_transformed = y_valid.copy()


    log_base_options = {'none': None, 'log2': 2, 'log10': 10, 'log5': 5, 'sqrt': 'sqrt', 'cbrt': 'cbrt'}
    log_transforms = {}
    for target in mean_columns:
        log_base = trial.suggest_categorical(f'Log_{target}', list(log_base_options.keys()))
        log_transforms[target] = log_base_options[log_base]
    
    callbacks = [
                 ReduceLROnPlateau('val_mae', patience=2, factor=0.7, mode = 'min', verbose = 0)]

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
    
    std_scaler = StandardScaler()

    y_train_transformed = y_train_transformed[mean_columns].values
    y_valid_transformed = y_valid_transformed[mean_columns].values

    y_train_transformed = std_scaler.fit_transform(y_train_transformed)
    y_valid_transformed = std_scaler.transform(y_valid_transformed)


    new_best = None
    new_best_found = False
    for epoch in range(17):

        model.fit([X_train_feat, X_train_tab], y_train_transformed, validation_data=([X_valid_feat, X_valid_tab], y_valid_transformed), batch_size=256, epochs=3, callbacks=callbacks, verbose = 0)
        preds_transformed = model.predict([X_valid_feat, X_valid_tab], verbose = 0)        

    
        try:        
           
            preds_transformed = std_scaler.inverse_transform(preds_transformed)
            
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
                    
                    print(f'Best epoch all errors R2 : {r2:.5f}, MSE : {mse:.5f}, MAE : {mae:.5f}')
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
                        pickle.dump(std_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print("*" * 50)
                
            tf.keras.backend.clear_session()
            gc.collect()
                
            return new_best
        
        if trial.number > 0 and not new_best_found:
            if new_best > study.best_value:
                new_best_found = True
            
                print("#" * 50)                
                       
                print(f'Old best R2 : {study.best_value:.5f}')
                print(f'New best R2 : {r2_score_inv:.5f}')

                r2 = r2_score_safe(y_valid, preds_transformed)
                mse  = mean_squared_error(y_valid, preds_transformed)
                mae = mean_absolute_error(y_valid, preds_transformed)
                
                print(f'Best epoch all errors R2 : {r2:.5f}, MSE : {mse:.5f}, MAE : {mae:.5f}')
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
                    pickle.dump(std_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

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
                
                print(f'Best epoch all errors R2 : {r2:.5f}, MSE : {mse:.5f}, MAE : {mae:.5f}')
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
                    pickle.dump(std_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

        
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
                            storage=f'sqlite:///420_kaikkimukana.db',
                            load_if_exists=True,
                            pruner=pruner
                            )

search_time_taken = 0
search_start = time.time()
round = 0
trials_done = 0



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
    genemachine = optuna.samplers.NSGAIISampler(crossover = optuna.samplers.nsgaii.VSBXCrossover(), mutation_prob = 0.03)

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



        




