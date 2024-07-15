import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tsai.all import *
# features = df[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values
# labels = df['label'].values

# Define paths
base_dir = r"C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data"
#processed_folders = ['processed_P2', 'processed_P3', 'processed_P4']
PROCESSED_FOLDER = ['processed_P3']

NUMBER_OF_EPOCH = 10
VALIDATION_SIZE = 0.3
BATCH_SIZE = 64

# Ensure your data is in a 3D shape (samples, sequence_length, features)
SEQUENCE_LENGTH = 150
NUMBER_OF_FEATURES = 8  # Number of features
NUMBER_OF_CLASSES = 2  # Number of classes (binary classification)
# Load processed data
# Collect all file paths
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
    return pd.concat(data_frames, ignore_index=True), np.array(sources_id)

def data_generation(processed_folders):
    df, sources_id = load_processed_data(processed_folders)
    # Filter out rows with label 'I'
    df = df[df['label'] != 'I']

    # features = df[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values
    # labels = df['label'].values

    # Define features and target
    features = df[['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD', 'FPOGX-diff', 'FPOGY-diff']].values

    # Encode labels (e.g., 'N' -> 0, 'U' -> 1)
    label_mapping = {'N': 0, 'U': 1}
    labels = df['label'].map(label_mapping).values

    # Create sequences without mixing sources
    def create_sequences(data, labels, sources, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            if np.all(sources[i:i + sequence_length] == sources[i]):  # Ensure the sequence is from a single source
                X.append(data[i:i + sequence_length])
                y.append(labels[i + sequence_length - 1])  # Use the last label in the sequence
        return np.array(X), np.array(y)
    def create_countinous_sequence(data, labels, sources, sequence_length):
        X, y = [], []
        for i in range(0, len(data) - sequence_length + 1, sequence_length):
            if np.all(sources[i:i + sequence_length] == sources[i]):  # Ensure the sequence is from a single source
                X.append(data[i:i + sequence_length])
                y.append(labels[i + sequence_length - 1])  # Use the last label in the sequence
        return np.array(X), np.array(y)

    X, y = create_countinous_sequence(features, labels, sources_id, SEQUENCE_LENGTH)
    y = y.astype(np.int64)  # Ensure labels are integers

    return X, y

def build_models():
    # Define the architectures and their hyperparameters
    archs = [
        (FCN, {}),
        (ResNet, {}),
        #(xresnet1d34, {}),  # NOT ok
        (ResCNN, {}),
        (LSTM, {'n_layers':1, 'bidirectional': False}),
        (LSTM, {'n_layers':2, 'bidirectional': False}),
        (LSTM, {'n_layers':3, 'bidirectional': False}),
        (LSTM, {'n_layers':1, 'bidirectional': True}),
        (LSTM, {'n_layers':2, 'bidirectional': True}),
        (LSTM, {'n_layers':3, 'bidirectional': True}),
        (LSTM_FCN, {}),
        (LSTM_FCN, {'shuffle': False}),
        (InceptionTime, {}),
        (XceptionTime, {}),
        (OmniScaleCNN, {}),
        #(mWDN, {'levels': 4})  # NOT ok
    ]
    return archs

def train_model(archs, dls):
    results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
    for i, (arch, k) in enumerate(archs):
        model = create_model(arch, dls=dls, **k)
        print(model.__class__.__name__)

        learn = Learner(dls, model, metrics=accuracy)
        start = time.time()
        learn.fit_one_cycle(NUMBER_OF_EPOCH, lr_max=1e-3)
        elapsed = time.time() - start
        vals = learn.recorder.values[-1]
        results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed)]
        results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
        #clear_output()
        display(results)

def main():
    X, y = data_generation(PROCESSED_FOLDER)
    print(X.shape)
    print(y.shape)
    # Split the data into training and validation sets
    #splits = get_splits(y, valid_size=VALIDATION_SIZE, stratify=True, shuffle=True)
    splits = get_splits(y, valid_size=VALIDATION_SIZE, shuffle=False)
    # Create the TSDatasets
    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    print(len(dsets.train))
    print(len(dsets.valid))
    # Create the TSDataLoaders
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[BATCH_SIZE, BATCH_SIZE * 2])

    archs = build_models()
    train_model(archs, dls)

main()