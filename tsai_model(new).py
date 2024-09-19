"""
This file is for training a tsai model using sequence timeseries data
each tester is divided into multiple timeseries with length is SEQUENCE_LENGTH
we have a lot more than 31 samples
To validate the final result, we need to predict small partial result of each tester, and then combine those result
"""

import os
import torch
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
from sklearn.metrics import accuracy_score
# features = df[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values
# labels = df['label'].values

NUMBER_OF_IMPLEMENTATION = 20

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
    for file_index in random_U_for_training:
        file_path = os.path.join(u_folder_path, u_files[file_index])
        df = pd.read_csv(file_path, nrows=NUMBER_OF_MIN_ROWS_SLIDE)
        data_frames.append(df)
        # Remove file already took in train
        u_files_testing.remove(u_files[file_index])

    # Add training data with class I
    random_N_for_training = random.sample(range(len(n_files)), int(len(n_files) * (1 - VALIDATION_SIZE)))
    logging.info(f"tester training: {random_U_for_training + random_N_for_training}")
    for file_index in random_N_for_training:
        file_path = os.path.join(n_folder_path, n_files[file_index])
        df = pd.read_csv(file_path, nrows=NUMBER_OF_MIN_ROWS_SLIDE)
        data_frames.append(df)
        # Remove file already took in train
        n_files_testing.remove(n_files[file_index])

    logging.info(f"tester validation: {u_files_testing + n_files_testing}")
    validation_dataframe = []
    # Add testing data with class U
    for file in u_files_testing:
        file_path = os.path.join(u_folder_path, file)
        df = pd.read_csv(file_path, nrows=NUMBER_OF_MIN_ROWS_SLIDE)
        data_frames.append(df)
        validation_dataframe.append(df)

    # Add testing data with class I
    for file in n_files_testing:
        file_path = os.path.join(n_folder_path, file)
        df = pd.read_csv(file_path, nrows=NUMBER_OF_MIN_ROWS_SLIDE)
        data_frames.append(df)
        validation_dataframe.append(df)

    def create_countinous_sequences(data):
        X, y = [], []
        for tester in range(len(data)):
            tester_dataframe = data[tester]
            # Define features and target
            features = tester_dataframe[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values
            # Encode labels (e.g., 'N' -> 0, 'U' -> 1)
            label_mapping = {'N': 1, 'U': 0}
            labels = tester_dataframe['label'].map(label_mapping).values
            for i in range(0, len(tester_dataframe) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
                X.append(features[i:i + SEQUENCE_LENGTH])
                y.append(labels[i + SEQUENCE_LENGTH - 1])  # Use the last label in the sequence
        return np.array(X), np.array(y)
    def create_countinous_sequences_for_each_tester(data):
        X, y = [], []
        tester_dataframe = data
        # Define features and target
        features = tester_dataframe[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values
        # Encode labels (e.g., 'N' -> 0, 'U' -> 1)
        label_mapping = {'N': 1, 'U': 0}
        labels = tester_dataframe['label'].map(label_mapping).values
        for i in range(0, len(tester_dataframe) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            X.append(features[i:i + SEQUENCE_LENGTH])
            y.append(labels[i + SEQUENCE_LENGTH - 1])  # Use the last label in the sequence
        return np.array(X), np.array(y)

    X, y = create_countinous_sequences(data_frames)


    # Create validation dataset with seperated tester.
    validation_data_X = []
    validation_data_y = []
    for tester_data in validation_dataframe:
        X_tester, y_tester = create_countinous_sequences_for_each_tester(tester_data)
        validation_data_X.append(X_tester)
        validation_data_y.append(y_tester[0])

    return X, y, validation_data_X, validation_data_y

def build_models():
    # Define the architectures and their hyperparameters
    archs = [
        (FCN, {}),
        (ResNet, {}),
        #(xresnet1d34, {}),  # NOT ok
        (ResCNN, {}),
        #(LSTM, {'n_layers':1, 'bidirectional': False, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        #(LSTM, {'n_layers':2, 'bidirectional': False, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        #(LSTM, {'c_out':2, 'n_layers':3, 'hidden_size':32, 'bidirectional': False, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        #(LSTM, {'n_layers':1, 'bidirectional': True, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        #(LSTM, {'n_layers':2, 'bidirectional': True}),
        (LSTM, {'c_out':2, 'n_layers':3, 'hidden_size':32, 'bidirectional': True, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        (LSTM_FCN, {}),
        #(LSTM_FCN, {'shuffle': False}),
        (InceptionTime, {}),
        (XceptionTime, {}),
        #(OmniScaleCNN, {}), # Not useful
        #(mWDN, {'levels': 4})  # NOT ok
    ]
    return archs

def train_model(archs, dls, validation_data_X, validation_data_y):
    results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'model_saved_name', 'valid loss', 'precision', 'recall', 'valid accuracy', 'true_validation_labels', 'predicted_validation_with_confidence','accuracy_by_tester'])
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
        learn.fit(NUMBER_OF_EPOCH, cbs=[save_callback, early_stopping])
        #start = time.time()
        #learn.fit_one_cycle(NUMBER_OF_EPOCH, cbs=[save_callback, early_stopping], lr_max=1e-3)

        #elapsed = time.time() - start
        #vals = learn.recorder.values[-1]

        #results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
        #clear_output()
        #display(results)


        def return_predicted_label_and_coffidence(pred_labels):

            # Count the number of 1s
            count_1 = np.sum(pred_labels)
            # Count the number of 0s
            count_0 = len(pred_labels) - count_1
            # Determine which value is higher and calculate the probability
            if count_1 > count_0:
                higher_value = 1
                higher_count = count_1
            else:
                higher_value = 0
                higher_count = count_0

            probability_higher_value = higher_count / len(pred_labels)
            return higher_value, probability_higher_value

        #print(f"validation_data_y: {str(validation_data_y)}")

        predicted_labels = []
        predicted_label_with_confidence = []
        # Validation
        for tester_index in range(len(validation_data_X)):
            tester_features = validation_data_X[tester_index]
            true_label = validation_data_y[tester_index]
            #print(true_label)
            #print(tester_features.shape)
            preds = []
            for sample in tester_features:

                # Convert to tensor if necessary
                sample_reshaped = sample.reshape(1, 300, 8)
                #sample_tensor = torch.tensor(sample_reshaped, dtype=torch.float32)
                #batch_tfms = TSStandardize(use_single_batch=True)
                # Create a TSDataset
                #sample_ds = TSDataset(sample_tensor)
                # Create a DataLoader from the TSDataset
                #dls = TSDataLoader(sample_ds, bs=1, batch_tfms=batch_tfms)
                # Perform prediction

                test_probas, test_targets, test_preds = learn.get_X_preds(sample_reshaped)

                #print(f"prediction: {test_preds}")
                preds.append(int(test_preds[1]))
            #print(preds)
            # Convert predictions to class labels if needed
            #pred_labels = np.argmax(preds, axis=1)
            pred_label, coffidence = return_predicted_label_and_coffidence(preds)
            predicted_labels.append(int(pred_label))
            predicted_label_with_confidence.append([pred_label, coffidence])

        # Calculate accuracy
        accuracy_by_tester = accuracy_score(validation_data_y, predicted_labels)
        #print(str(predicted_label_with_confidence))

        validation = learn.validate()
        print(validation)
        logging.info(validation)

        results.loc[i] = [arch.__name__, k, count_parameters(model), model_saved_name, validation[0], validation[1],
                          validation[2], validation[3], str(validation_data_y), str(predicted_label_with_confidence), accuracy_by_tester]

    return results


def implementation():
    X, y, validation_data_X, validation_data_y = load_processed_data(PROCESSED_FOLDER)

    # Split the data into training and validation sets
    splits = get_splits(y, valid_size=VALIDATION_SIZE, shuffle=False, show_plot=False)
    # Create the TSDatasets
    tfms = [None, [Categorize()]]
    batch_tfms = TSStandardize(use_single_batch=True)
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    #dsets_test = TSDatasets(X_test, y_test, tfms=tfms)

    # Create the TSDataLoaders
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[BATCH_SIZE, BATCH_SIZE * 2], batch_tfms=batch_tfms)

    archs = build_models()
    results = train_model(archs, dls, validation_data_X, validation_data_y)

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
    logging.info(f"Slide: {PROCESSED_FOLDER}")
    logging.info(f"NUMBER_OF_EPOCH: {NUMBER_OF_EPOCH}")
    logging.info(f"VALIDATION_SIZE: {VALIDATION_SIZE}")
    logging.info(f"SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    logging.info(f"PATIENCE: {PATIENCE}")
    logging.info('Started')
    try:
        for i in range(NUMBER_OF_IMPLEMENTATION):
            results = implementation()
            save_result_to_csv(f"{str(date_string)}.csv", results)
    except Exception as error:
        logging.error(error)
    logging.info('Finished')

main()