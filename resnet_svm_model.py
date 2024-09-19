import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn import svm
import os

# Define paths
base_dir = r"C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data"
#PROCESSED_FOLDER = ['slide1', 'slide2', 'slide3']
PROCESSED_FOLDER = 'slide2'

parent_directory = os.getcwd()

NUMB_OF_ITERATION = 20
ACCURACY = []

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
NUMBER_OF_FEATURES = 8  # Number of features
NUMBER_OF_CLASSES = 2  # Number of classes (binary classification)
SEQUENCE_LENGTH = 300
PATIENCE = 10


def model_resnet_1d(input_shape):
    def conv1d_bn(x, filters, kernel_size=3, strides=1, padding='same'):
        x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    def residual_block_1d(x, filters):
        shortcut = x
        x = conv1d_bn(x, filters, kernel_size=3)
        x = conv1d_bn(x, filters, kernel_size=3)
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x
    def build_resnet_1d(input_shape):
        inputs = tf.keras.Input(shape=input_shape)

        # Initial Conv Layer
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)

        # Add residual blocks
        x = residual_block_1d(x, 64)
        x = residual_block_1d(x, 64)

        # Global Average Pooling and Dense layers for classification
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Flatten()(x)

        model = models.Model(inputs, outputs)
        return model

    return build_resnet_1d(input_shape)

def model_resnet18_1d(input_shape):
    def conv1d_bn(x, filters, kernel_size, strides=1, padding='same', name=None):
        x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def identity_block_1d(input_tensor, filters, kernel_size=3):
        # Save the input value
        x = conv1d_bn(input_tensor, filters, kernel_size)
        x = conv1d_bn(x, filters, kernel_size)

        # Add shortcut connection
        shortcut = layers.BatchNormalization()(input_tensor)
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def conv_block_1d(input_tensor, filters, kernel_size=3, strides=2):
        # Shortcut path
        shortcut = conv1d_bn(input_tensor, filters, kernel_size=1, strides=strides)
        shortcut = layers.BatchNormalization()(shortcut)
        # Main path
        x = conv1d_bn(input_tensor, filters, kernel_size, strides=strides)
        x = conv1d_bn(x, filters, kernel_size)

        # Add shortcut connection
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def build_resnet18_1d(input_shape):
        inputs = tf.keras.Input(shape=input_shape)

        # Initial conv layer
        x = conv1d_bn(inputs, 64, 7, strides=2)
        x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

        # Residual blocks
        x = conv_block_1d(x, 64, 3)
        x = identity_block_1d(x, 64, 3)
        #x = identity_block_1d(x, 64, 3)

        x = conv_block_1d(x, 128, 3)
        x = identity_block_1d(x, 128, 3)
        #x = identity_block_1d(x, 128, 3)

        x = conv_block_1d(x, 256, 3)
        x = identity_block_1d(x, 256, 3)
        #x = identity_block_1d(x, 256, 3)

        x = conv_block_1d(x, 512, 3)
        x = identity_block_1d(x, 512, 3)
        #x = identity_block_1d(x, 512, 3)

        # Global Average Pooling and Dense layer for classification
        x = layers.GlobalAveragePooling1D()(x)
        # Fully connected layer
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        # Output layer for binary classification
        outputs = layers.Dense(1, activation='sigmoid')(x)
        #outputs = layers.Flatten()(x)

        model = models.Model(inputs, outputs)
        return model

    return build_resnet18_1d(input_shape)


# Data preprocessing (using the same method to load and normalize data as before)
def load_and_preprocess_data(root_dir):
    X_train_U, X_train_N = [], []
    y_train_U, y_train_N = [], []
    validation_X_U, validation_X_N = [], []
    validation_y_U, validation_y_N = [], []

    def process_file(file_path):
        # Read the entire CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None, skiprows=1, nrows=NUMBER_OF_MIN_ROWS_SLIDE, usecols=range(NUMBER_OF_FEATURES), dtype='float64')

        #leftpupil_diameter = df.iloc[:, 4]  # 5th column is index 4 (zero-based indexing)
        #pupil_diameter_scaled = leftpupil_diameter.diff().fillna(0)  # Take the difference over rows to capture changes
        #df.iloc[:, 4] = pupil_diameter_scaled

        #rightpupil_diameter = df.iloc[:, 5]  # 5th column is index 5 (zero-based indexing)
        #pupil_diameter_scaled = rightpupil_diameter.diff().fillna(0)  # Take the difference over rows to capture changes
        #df.iloc[:, 5] = pupil_diameter_scaled

        # Process the DataFrame in chunks of SEQUENCE_LENGTH rows
        for start in range(0, df.shape[0] - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            chunk = df.iloc[start:start + SEQUENCE_LENGTH].values
            yield chunk

    # Load data for each folder 'U' and 'N'
    for folder in ['U', 'N']:
        folder_path = os.path.join(root_dir, folder)

        # Get list of files in the folder
        tester_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        # Split files at the tester level into training and testing
        train_files, test_files = train_test_split(tester_files, test_size=0.2, random_state=42)

        # Process training files
        for tester_file in train_files:
            file_path = os.path.join(folder_path, tester_file)
            for chunk in process_file(file_path):
                if folder == 'U':
                    X_train_U.append(chunk)
                    y_train_U.append(0)  # Label for 'U'
                else:
                    X_train_N.append(chunk)
                    y_train_N.append(1)  # Label for 'N'

        # Process testing files (for validation data, split by tester)
        for tester_file in test_files:
            tester_data = []  # Collect chunks for this tester
            tester_labels = []  # Collect labels for this tester
            file_path = os.path.join(folder_path, tester_file)
            for chunk in process_file(file_path):
                tester_data.append(chunk)
                if folder == 'U':
                    tester_labels.append(0)  # Label for 'U'
                else:
                    tester_labels.append(1)  # Label for 'N'

            # Add the tester's data and labels to validation arrays
            if folder == 'U':
                validation_X_U.append(np.array(tester_data))
                validation_y_U.append(tester_labels[0])
            else:
                validation_X_N.append(np.array(tester_data))
                validation_y_N.append(tester_labels[0])

    # Convert lists to numpy arrays for training data
    X_train_U = np.array(X_train_U)
    X_train_N = np.array(X_train_N)
    y_train_U = np.array(y_train_U)
    y_train_N = np.array(y_train_N)

    # Class balancing for training data
    min_train_samples = min(len(X_train_U), len(X_train_N))

    X_train_U = X_train_U[:min_train_samples]
    y_train_U = y_train_U[:min_train_samples]
    X_train_N = X_train_N[:min_train_samples]
    y_train_N = y_train_N[:min_train_samples]

    # Stack training data and labels
    X_train = np.vstack((X_train_U, X_train_N))
    y_train = np.concatenate((y_train_U, y_train_N))  # Flatten labels into a 1D array


    # Class balancing for validation data
    min_test_samples = min(len(validation_X_U), len(validation_X_N))
    validation_X_U = validation_X_U[:min_test_samples]
    validation_y_U = validation_y_U[:min_test_samples]
    validation_X_N = validation_X_N[:min_test_samples]
    validation_y_N = validation_y_N[:min_test_samples]

    # Stack validation data and labels
    validation_X = validation_X_U + validation_X_N
    validation_y = validation_y_U + validation_y_N

    return X_train, y_train, validation_X, validation_y


def implement():
    # Load and preprocess data
    processed_file_path = os.path.join(base_dir, PROCESSED_FOLDER)
    X_train, y_train, validation_X, validation_y = load_and_preprocess_data(processed_file_path)

    # Apply StandardScaler separately to X_train and X_test
    scaler = RobustScaler()
    #X_train = scaler.fit_transform(X_train.reshape(-1, NUMBER_OF_FEATURES)).reshape(X_train.shape)

    # Reshape data for ResNet
    X_train_reshaped = X_train.reshape((X_train.shape[0], SEQUENCE_LENGTH, NUMBER_OF_FEATURES))

    # Build and compile the ResNet model
    input_shape = (SEQUENCE_LENGTH, NUMBER_OF_FEATURES)  # Shape of your input data (1 channel)
    #resnet_model = model_resnet(input_shape)
    resnet_model = model_resnet18_1d(input_shape)
    resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Create EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    # Fit the model with early stopping
    history = resnet_model.fit(
        X_train_reshaped, y_train,
        epochs=NUMBER_OF_EPOCH,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    # X_train and X_test are your input data
    X_train_features = resnet_model.predict(X_train)

    X_test_features = []
    for tester in validation_X:
        X_test_feature = resnet_model.predict(tester)
        X_test_features.append(X_test_feature)


    # Now use SVM on extracted features
    svm_classifier = svm.SVC()

    # Train SVM on ResNet features
    svm_classifier.fit(X_train_features, y_train)

    # Evaluation
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

    # print(f"validation_data_y: {str(validation_data_y)}")

    predicted_labels = []
    predicted_label_with_confidence = []
    # Validation
    for tester_index in range(len(X_test_features)):
        tester_data = X_test_features[tester_index]
        true_label = X_test_features[tester_index]
        # print(true_label)
        # print(tester_features.shape)
        preds = []
        # Get predicted probabilities from model
        predictions = []
        for sample in tester_data:

            sample = sample.reshape(1, -1)
            pred = svm_classifier.predict(sample)
            predictions.append(pred[0])  # X_test is your test data

        #print(str(predictions))

        # predicted_classes is now a 2D array with shape (num_samples, 1)
        # If you need it as a 1D array:
        #predicted_classes = predicted_classes.flatten()
        #print(preds)
        # Convert predictions to class labels if needed
        # pred_labels = np.argmax(preds, axis=1)
        pred_label, coffidence = return_predicted_label_and_coffidence(predictions)
        predicted_labels.append(int(pred_label))
        predicted_label_with_confidence.append([pred_label, coffidence])

    # Calculate accuracy
    accuracy_by_tester = accuracy_score(validation_y, predicted_labels)
    #print(str(validation_y))
    #print(str(predicted_labels))
    #print(accuracy_by_tester)
    print(accuracy_by_tester)
    ACCURACY.append(accuracy_by_tester)
    #y_pred = resnet_model.predict(X_test_reshaped)


def implement_resnet_keras():
    # Load and preprocess data
    processed_file_path = os.path.join(base_dir, PROCESSED_FOLDER)
    X_train, y_train, X_test, y_test = load_and_preprocess_data(processed_file_path)

    # Reshape data for ResNet
    print(X_train.shape)
    print(X_train[0].shape)
    X_train_reshaped = X_train.reshape((X_train.shape[0], 32, 32, 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 32, 32, 1))

    # Load the pre-trained ResNet50 model, excluding the top (output) layer
    # Adjust the input shape to your own data (224x224x1 for grayscale data)
    base_model = ResNet50(
        include_top=False,  # Exclude the top (fully connected) layers
        weights=None,  # You can also use 'imagenet' weights if your input shape is 3 channels
        input_shape=(32, 32, 1)  # Set this to the shape of your input data (grayscale in your case)
    )

    # Add your own fully connected layers for binary classification
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)  # Binary classification, so use sigmoid

    # Create the model
    model = models.Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Now you can train the model as usual, using your X_train and y_train
    # Assuming X_train and X_test are already resized to (224, 224, 1)

    # Create early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Fit the model
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=NUMBER_OF_EPOCH,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test)

    # Print the test accuracy
    print(f'Test accuracy: {test_accuracy:.4f}')

def main():
    for i in range(NUMB_OF_ITERATION):
        implement()
    #implement_resnet_keras()
    print(ACCURACY)
    print(f"Average accuracy after {NUMB_OF_ITERATION}: {np.mean(ACCURACY)}")

main()