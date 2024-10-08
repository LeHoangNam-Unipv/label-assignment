"""
This file is for training a tsai model using sequence timeseries data
each tester is a timeseries data with length is NUMBER_OF_MIN_ROWS_SLIDE
so we have total 31 timeseries samples
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tsai.all import *
from fastai.callback.all import EarlyStoppingCallback, SaveModelCallback
import logging
# Import the Adam optimizer
from fastai.optimizer import Adam
from sklearn.model_selection import StratifiedShuffleSplit

# features = df[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values
# labels = df['label'].values

NUMBER_OF_IMPLEMENTATION = 10

# Define paths
base_dir = r"C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data"
#PROCESSED_FOLDER = ['slide1', 'slide2', 'slide3']
PROCESSED_FOLDER = 'slide2'
parent_directory = os.getcwd()


if(PROCESSED_FOLDER == 'slide1'):
    NUMBER_OF_MIN_ROWS_SLIDE = 9300
elif (PROCESSED_FOLDER == 'slide2'):
    NUMBER_OF_MIN_ROWS_SLIDE = 16800
elif (PROCESSED_FOLDER == 'slide3'):
    NUMBER_OF_MIN_ROWS_SLIDE = 34500
else:
    NUMBER_OF_MIN_ROWS_SLIDE = 9300

NUMBER_OF_EPOCH = 200
VALIDATION_SIZE = 0.2
BATCH_SIZE = 64

# Ensure your data is in a 3D shape (samples, sequence_length, features)
SEQUENCE_LENGTH = 300
NUMBER_OF_SAMPLE_PER_TESTER = NUMBER_OF_MIN_ROWS_SLIDE / SEQUENCE_LENGTH
NUMBER_OF_FEATURES = 8  # Number of features
NUMBER_OF_CLASSES = 2  # Number of classes (binary classification)

PATIENCE = 20  # Early stopping

now = datetime.now()
date_string = now.strftime("%d-%m-%Y-%H-%M-%S")


# Load processed data
def load_processed_data(processed_folders):
    data_frames = []
    sources_id = []
    folder = processed_folders
    folder_path = os.path.join(base_dir, folder)
    u_folder_path = os.path.join(folder_path, "U")
    n_folder_path = os.path.join(folder_path, "N")
    u_files = [f for f in os.listdir(u_folder_path) if f.endswith('_processed_gaze.csv')]
    n_files = [f for f in os.listdir(n_folder_path) if f.endswith('_processed_gaze.csv')]

    # Take equal number of classes
    min_files = min(len(u_files), len(n_files))
    u_files = random.sample(u_files, min_files)
    n_files = random.sample(n_files, min_files)

    u_files_testing = u_files.copy()
    n_files_testing = n_files.copy()

    # Add training data with class U
    random_U_for_training = random.sample(range(len(u_files)), int(len(u_files)*(1-VALIDATION_SIZE)))
    print(random_U_for_training)
    for file_index in random_U_for_training:
        file_path = os.path.join(u_folder_path, u_files[file_index])
        df = pd.read_csv(file_path, nrows=NUMBER_OF_MIN_ROWS_SLIDE)
        data_frames.append(df)
        # Extract user ID from file name (assume file name format 'User 0_all_gaze.csv')
        user_id = u_files[file_index].split(' ')[1].split('_')[0] # Adjust split index if necessary
        folder_id = folder.split('slide')[1]  # Adjust split index if necessary
        source_id = int(user_id+folder_id)
        sources_id.extend([source_id] * len(df))  # Extend user_ids with user_id repeated for each row
        # Remove file already took in train
        u_files_testing.remove(u_files[file_index])


    # Add training data with class I
    random_N_for_training = random.sample(range(len(n_files)), int(len(n_files) * (1 - VALIDATION_SIZE)))
    for file_index in random_N_for_training:
        file_path = os.path.join(n_folder_path, n_files[file_index])
        df = pd.read_csv(file_path, nrows=NUMBER_OF_MIN_ROWS_SLIDE)
        data_frames.append(df)
        # Extract user ID from file name (assume file name format 'User 0_all_gaze.csv')
        user_id = n_files[file_index].split(' ')[1].split('_')[0]  # Adjust split index if necessary
        folder_id = folder.split('slide')[1]  # Adjust split index if necessary
        source_id = int(user_id + folder_id)
        sources_id.extend([source_id] * len(df))  # Extend user_ids with user_id repeated for each row
        # Remove file already took in train
        n_files_testing.remove(n_files[file_index])

    # Add testing data with class U
    for file in u_files_testing:
        file_path = os.path.join(u_folder_path, file)
        df = pd.read_csv(file_path, nrows=NUMBER_OF_MIN_ROWS_SLIDE)
        data_frames.append(df)
        # Extract user ID from file name (assume file name format 'User 0_all_gaze.csv')
        user_id = file.split(' ')[1].split('_')[0]  # Adjust split index if necessary
        folder_id = folder.split('slide')[1]  # Adjust split index if necessary
        source_id = int(user_id + folder_id)
        sources_id.extend([source_id] * len(df))  # Extend user_ids with user_id repeated for each row

    # Add testing data with class I
    for file in n_files_testing:
        file_path = os.path.join(n_folder_path, file)
        df = pd.read_csv(file_path, nrows=NUMBER_OF_MIN_ROWS_SLIDE)
        data_frames.append(df)
        # Extract user ID from file name (assume file name format 'User 0_all_gaze.csv')
        user_id = file.split(' ')[1].split('_')[0]  # Adjust split index if necessary
        folder_id = folder.split('slide')[1]  # Adjust split index if necessary
        source_id = int(user_id + folder_id)
        sources_id.extend([source_id] * len(df))  # Extend user_ids with user_id repeated for each row

    return pd.concat(data_frames, ignore_index=True), np.array(sources_id)

def data_generation(processed_folders):
    df, sources_id = load_processed_data(processed_folders)

    # Filter row other than U and N
    df = df[df['label'].isin(['U', 'N'])]

    # Define features and target
    features = df[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values

    # Encode labels (e.g., 'N' -> 0, 'U' -> 1)
    label_mapping = {'N': 1, 'U': 0}
    labels = df['label'].map(label_mapping).values

    # Create sequences without mixing sources
    def create_sequences(data, labels, sources, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            if np.all(sources[i:i + sequence_length] == sources[i]):  # Ensure the sequence is from a single source
                X.append(data[i:i + sequence_length])
                y.append(labels[i + sequence_length - 1])  # Use the last label in the sequence
        return np.array(X), np.array(y)
    def create_countinous_sequences(data, labels, sources, sequence_length):
        X, y = [], []
        for i in range(0, len(data) - sequence_length + 1, sequence_length):
            if np.all(sources[i:i + sequence_length] == sources[i]):  # Ensure the sequence is from a single source
                X.append(data[i:i + sequence_length])
                y.append(labels[i + sequence_length - 1])  # Use the last label in the sequence
        return np.array(X), np.array(y)

    X, y = create_countinous_sequences(features, labels, sources_id, SEQUENCE_LENGTH)
    y = y.astype(np.int64)  # Ensure labels are integers
    return X, y

def build_models():
    # Define the architectures and their hyperparameters
    archs = [
        #(FCN, {}),
        #(ResNet, {}),
        #(xresnet1d34, {}),  # NOT ok
        #(ResCNN, {}),
        #(LSTM, {'n_layers':1, 'bidirectional': False, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        #(LSTM, {'n_layers':2, 'bidirectional': False, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        (LSTM, {'c_out':2, 'n_layers':3, 'hidden_size':32, 'bidirectional': False, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        #(LSTM, {'n_layers':1, 'bidirectional': True, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        #(LSTM, {'n_layers':2, 'bidirectional': True}),
        #(LSTM, {'n_layers':3, 'bidirectional': True}),
        #(LSTM_FCN, {}),
        #(LSTM_FCN, {'shuffle': False}),
        #(InceptionTime, {}),
        #(XceptionTime, {}),
        #(OmniScaleCNN, {}), # Not useful
        #(mWDN, {'levels': 4})  # NOT ok
    ]
    return archs

def train_model(archs, dls):
    results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'valid loss', 'precision', 'recall', 'valid accuracy'])
    for i, (arch, k) in enumerate(archs):
        model = create_model(arch, dls=dls, **k)
        print(model.__class__.__name__)
        model_saved_name = str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        # Initialize the EarlyStoppingCallback
        save_callback = SaveModelCallback(monitor='valid_loss', comp=None, fname=model_saved_name, every_epoch=False,
                                          at_end=False, with_opt=False, reset_on_fit=True)
        early_stopping  = EarlyStoppingCallback(patience = PATIENCE)

        loss = CrossEntropyLossFlat()
        learn = Learner(dls, model, metrics=[Precision(), Recall(), accuracy], loss_func=loss, opt_func=Adam)
        print(str(learn.opt_func))
        print(str(learn.loss_func))
        #start = time.time()
        #learn.fit_one_cycle(NUMBER_OF_EPOCH, cbs=[save_callback, early_stopping], lr_max=1e-3)
        learn.fit(NUMBER_OF_EPOCH, cbs=[save_callback, early_stopping])
        #elapsed = time.time() - start
        #vals = learn.recorder.values[-1]
        validation = learn.validate()
        results.loc[i] = [arch.__name__, k, count_parameters(model), validation[0], validation[1], validation[2], validation[3]]
        #results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
        #clear_output()
        #display(results)

        print(validation)
        logging.info(validation)

    return results

def implementation():
    X, y =  data_generation(PROCESSED_FOLDER)

    # Split the data into training and validation sets
    splits = get_splits(y, valid_size=VALIDATION_SIZE, shuffle=False, show_plot=False)
    # Create the TSDatasets
    tfms = [None, [Categorize()]]
    batch_tfms = TSStandardize(use_single_batch=True)
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    print(len(dsets.train))
    print(len(dsets.valid))
    # Create the TSDataLoaders
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[BATCH_SIZE, BATCH_SIZE * 2], batch_tfms=batch_tfms)
    archs = build_models()
    results = train_model(archs, dls)

    return results

def save_result_to_csv(file_path, results):
    f = os.path.join(parent_directory, "result", file_path)
    # Append new DataFrame to the existing CSV file
    append_to_csv(f, results)


# Create a function to append DataFrame to a CSV file
def append_to_csv(file_path, df):
    # Check if the file exists
    if not os.path.isfile(file_path):
        # File does not exist, write the DataFrame with the header
        df.to_csv(file_path, mode='w', index=False, header=True)
    else:
        # File exists, append the DataFrame without the header
        df.to_csv(file_path, mode='a', index=False, header=False)

def main():
    logging.basicConfig(filename=os.path.join(parent_directory, 'log', f'{date_string}.log'),
                        format='%(asctime)s %(message)s', encoding='utf-8', level=logging.INFO)
    logging.info('Started')
    try:
        for i in range(NUMBER_OF_IMPLEMENTATION):
            results = implementation()
            save_result_to_csv(str(date_string), results)
    except Exception as error:
        logging.error(error)
    logging.info('Finished')

main()