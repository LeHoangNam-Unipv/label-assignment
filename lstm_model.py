from datetime import datetime
import os
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import timeseries_dataset_from_array
import re
import random  # Import the random module
import logging

parent_directory = os.getcwd()

NUMBER_OF_TESTERS = 31
NUMBER_OF_EPOCH = 5
TRAIN_SIZE = 0.7

# Prepare sequences
sequence_length = 300  # Define the sequence length
batch_size = 32  # Define batch size

features_shape = 8

# Define paths
base_dir = r"C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data"
#processed_folders = ['processed_P2', 'processed_P3', 'processed_P4']
processed_folders = ['processed_P3', 'processed_P4']



def extract_numbers(s):
    # Find all sequences of digits in the string
    return re.findall(r'\d+', s)

# Load the processed data
def load_processed_data(processed_folders):
    data_frames = []
    sources_id = []
    for folder in processed_folders:
        folder_path = os.path.join(base_dir, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('_processed_gaze.csv')]
        random.shuffle(files)  # Shuffle the list of files
        for file in files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            data_frames.append(df)
            # Extract user ID from file name (assume file name format 'User 0_all_gaze.csv')
            user_id = file.split(' ')[1].split('_')[0] # Adjust split index if necessary
            folder_id = folder.split('P')[1]  # Adjust split index if necessary
            source_id = int(user_id+folder_id)
            sources_id.extend([source_id] * len(df))  # Extend user_ids with user_id repeated for each row
    return pd.concat(data_frames, ignore_index=True), sources_id

def build_LSTM_model():
    # Build the LSTM model for binary classification (N vs U)
    model = Sequential()
    model.add(InputLayer(input_shape=[sequence_length, features_shape]))
    model.add(LSTM(64, return_sequences=True))
    #model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    #model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification, sigmoid activation

    model.summary()
    return model

def train_model(model, train_generator, test_generator):
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(train_generator, epochs=NUMBER_OF_EPOCH, validation_data=test_generator, shuffle=True, callbacks=[early_stopping])
    logging.info(history)


def train_test_data_generator(processed_folders):
    df, sources_id = load_processed_data(processed_folders)

    # Extract features and labels
    features = df[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values
    labels = df['label'].values

    # Filter out instances with label 'I'
    filtered_indices = np.where(labels != 'I')[0]
    features = features[filtered_indices]
    labels = labels[filtered_indices]
    user_ids = np.array(sources_id)[filtered_indices]  # Apply the same filter to user_ids

    # Encode labels (N -> 0, U -> 1)
    label_mapping = {'N': 0, 'U': 1}
    labels = np.vectorize(label_mapping.get)(labels)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Create source information based on user IDs
    src = np.expand_dims(user_ids, 1)

    # Append source information to X for filtration
    features_with_src = np.append(src, features, 1)

    # Split data into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(features_with_src, labels, test_size=1-TRAIN_SIZE,
                                                                                shuffle=False)
    # Create TimeseriesGenerator for training and testing
    train_generator = TimeseriesGenerator(features_train, labels_train, length=sequence_length, batch_size=batch_size)
    test_generator = TimeseriesGenerator(features_test, labels_test, length=sequence_length, batch_size=batch_size)

    #generator = TimeseriesGenerator(features, labels, length=sequence_length, batch_size=batch_size)
    # Split data into training and testing sets
    #train_generator = TimeseriesGenerator(features, labels, length=sequence_length, batch_size=batch_size, shuffle=True, end_index=TRAIN_SIZE)
    #test_generator = TimeseriesGenerator(features, labels, length=sequence_length, batch_size=batch_size, start_index=TRAIN_SIZE)

    return train_generator, test_generator


def train_test_data_timeseries(processed_folders):
    df, user_ids = load_processed_data(processed_folders)

    # Extract features and labels
    features = df[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values
    labels = df['label'].values

    # Filter out instances with label 'I'
    filtered_indices = np.where(labels != 'I')[0]
    features = features[filtered_indices]
    labels = labels[filtered_indices]
    user_ids = np.array(user_ids)[filtered_indices]  # Apply the same filter to user_ids

    # Encode labels (N -> 0, U -> 1)
    label_mapping = {'N': 0, 'U': 1}
    labels = np.vectorize(label_mapping.get)(labels)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Create source information based on user IDs
    src = np.expand_dims(user_ids, 1)

    # Append source information to X for filtration
    features_with_src = np.append(src, features, 1)
    print(features_with_src.shape)

    # Split data into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(features_with_src, labels,
                                                                                test_size=1 - TRAIN_SIZE,
                                                                                shuffle=False)
    # Create TimeseriesGenerator for training and testing
    train_generator = timeseries_dataset_from_array(data=features_train, targets=labels_train, sequence_length=sequence_length, batch_size=1, shuffle=False)
    test_generator = timeseries_dataset_from_array(data=features_test, targets=labels_test, sequence_length=sequence_length, batch_size=1, shuffle=False)

    # Filtering by and removing src info
    def single_source(x, y):
        source = x[:, :, 0]
        return tf.reduce_all(source == source[0])

    def drop_source(x, y):
        x_ = x[:, :, 1:]
        return x_, y

    train_generator = train_generator.filter(single_source)
    train_generator = train_generator.map(drop_source)
    train_generator = train_generator.unbatch().batch(batch_size)


    test_generator = test_generator.filter(single_source)
    test_generator = test_generator.map(drop_source)
    test_generator = test_generator.unbatch().batch(batch_size)


    return train_generator, test_generator

def evaluation_model(model, test_data):
    # Evaluate the model
    loss, accuracy = model.evaluate(test_data)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Make predictions
    predictions = model.predict(test_data)
    predicted_labels = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions (0 or 1)

    # Compare with true labels
    true_labels = np.array([test_data[i][1] for i in range(len(test_data))]).flatten()

    # Print some sample predictions
    for i in range(10):
        print(f'Predicted: {predicted_labels[i]}, True: {true_labels[i]}')


def implementation():
    #train_generator, test_generator = train_test_data_generator(processed_folders)
    train_generator, test_generator =train_test_data_timeseries(processed_folders)
    model = build_LSTM_model()
    train_model(model, train_generator, test_generator)


def main():
    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    logging.basicConfig(filename=os.path.join(parent_directory,'log',f'{date_string}.log'),
                        format='%(asctime)s %(message)s', encoding='utf-8', level=logging.INFO)
    logging.info('Started')
    try:
        implementation()
    except Exception as error:
        logging.error(error)
    logging.info('Finished')

main()