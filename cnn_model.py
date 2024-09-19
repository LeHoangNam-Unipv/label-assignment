"""
This file is for training a CNN model
each tester is divided into muiltiple 2D samples with length = SEQUENCE_LENGHT
so we have a lot more than 31 samples
"""

import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
BATCH_SIZE = 32
NUMBER_OF_FEATURES = 8  # Number of features
NUMBER_OF_CLASSES = 2  # Number of classes (binary classification)
SEQUENCE_LENGTH = 150

def load_dataset(processed_folders):
    data_U = []
    data_N = []

    def process_file(file_path):
        # Read the entire CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None, skiprows=1, nrows=NUMBER_OF_MIN_ROWS_SLIDE, usecols=range(NUMBER_OF_FEATURES))

        # Process the DataFrame in chunks of 150 rows
        for start in range(0, df.shape[0] - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            chunk = df.iloc[start:start + SEQUENCE_LENGTH].values
            yield chunk

    # Loop through each tester's CSV in the "U" and "N" folders
    for folder in ['U', 'N']:
        folder_path = os.path.join(processed_folders, folder)

        for tester_file in os.listdir(folder_path):
            if tester_file.endswith('.csv'):
                file_path = os.path.join(folder_path, tester_file)

                # Load the CSV file (assuming it has n rows and 8 columns)
                #data = pd.read_csv(file_path, header=None, skiprows=1, nrows=NUMBER_OF_MIN_ROWS_SLIDE, usecols=range(NUMBER_OF_FEATURES))
                for chunk in process_file(file_path):
                    if folder == 'U':
                        data_U.append(chunk)
                    else:
                        data_N.append(chunk)

    # Convert lists to NumPy arrays for easier manipulation
    data_U = np.array(data_U)
    data_N = np.array(data_N)

    # Normalize data separately for each tester
    def normalize_data(data_list):
        scaled_data = []
        for data in data_list:
            scaler = StandardScaler()
            data_flattened = data.reshape(-1, data.shape[-1])
            data_scaled = scaler.fit_transform(data_flattened).reshape(data.shape)
            scaled_data.append(data_scaled)
        return np.array(scaled_data)

    data_U = normalize_data(data_U)
    data_N = normalize_data(data_N)

    # Shuffle data before balancing
    np.random.shuffle(data_U)
    np.random.shuffle(data_N)

    # Ensure equal number of samples for both classes
    min_samples = min(len(data_U), len(data_N))
    data_U = data_U[:min_samples]
    data_N = data_N[:min_samples]

    # Create labels for both classes
    labels_U = np.zeros((min_samples, 1))  # "U" class as 0
    labels_N = np.ones((min_samples, 1))  # "N" class as 1

    # Split each class into training and testing sets separately (80/20 split)
    X_U_train, X_U_test, y_U_train, y_U_test = train_test_split(data_U, labels_U, test_size=0.2, random_state=42)
    X_N_train, X_N_test, y_N_train, y_N_test = train_test_split(data_N, labels_N, test_size=0.2, random_state=42)

    # Combine both classes' training and testing sets
    X_train = np.vstack((X_U_train, X_N_train))
    y_train = np.vstack((y_U_train, y_N_train))

    X_test = np.vstack((X_U_test, X_N_test))
    y_test = np.vstack((y_U_test, y_N_test))

    # Shuffle training and testing data to mix the classes
    train_indices = np.arange(X_train.shape[0])
    np.random.shuffle(train_indices)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    test_indices = np.arange(X_test.shape[0])
    np.random.shuffle(test_indices)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    return X_train, y_train, X_test, y_test

def build_cnn_model():
    # Define the model
    model = models.Sequential()

    # 1st Convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SEQUENCE_LENGTH, NUMBER_OF_FEATURES, 1)))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.25)),
    # 2nd Convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.25)),
    # Flatten the output to feed into a Dense layer
    model.add(layers.Flatten())

    # Fully connected layer (Dense layer)
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.25)),
    # Output layer (adjust units based on the number of classes for classification)
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Summary of the model
    model.summary()

    return model

def implement():
    processed_folders = os.path.join(base_dir, PROCESSED_FOLDER)
    X_train, y_train, X_test, y_test = load_dataset(processed_folders)

    # Reshape data back to 4D (samples x height x width x channels)
    X_train_reshaped = X_train.reshape(X_train.shape[0], SEQUENCE_LENGTH, NUMBER_OF_FEATURES, 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], SEQUENCE_LENGTH, NUMBER_OF_FEATURES, 1)

    # Reshape y_train and y_test if needed
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    print(y_test.shape)

    model = build_cnn_model()

    # Create the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        patience=10,  # Number of epochs to wait for improvement
        restore_best_weights=True  # Restore weights from the epoch with the best validation loss
    )

    # Fit the model
    history = model.fit(X_train_reshaped, y_train, epochs=NUMBER_OF_EPOCH, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[early_stopping])

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test)

    # Print the testing accuracy
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Get predicted probabilities
    y_pred_prob  = model.predict(X_test_reshaped)
    # Convert probabilities to binary labels (0 or 1)
    y_pred = (y_pred_prob > 0.5).astype(int)
    # Print the predicted labels
    print("Predicted labels for X_test:", y_pred.flatten())


def main():
    implement()

main()
