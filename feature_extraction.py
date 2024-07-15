import os
import pandas as pd
import numpy as np

# Define the columns to extract and new feature names
columns_to_extract = ['LPCX', 'LPCY', 'RPCX', 'RPCY', 'LPD', 'RPD']
new_feature_names = ['FPOGX-diff', 'FPOGY-diff']


def calculate_new_features(df):
    """Calculate new features based on the difference from the previous row."""
    df['FPOGX-diff'] = df['FPOGX'].diff().fillna(0)
    df['FPOGY-diff'] = df['FPOGY'].diff().fillna(0)
    return df


def process_gaze_csv(file_path, output_path, labels_df, tester_id, question_idx):
    """Process a gaze CSV file to extract specific columns, calculate new features, and add labels."""
    df = pd.read_csv(file_path)

    # Check if required columns are present
    required_columns = columns_to_extract + ['FPOGX', 'FPOGY']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in {file_path}")

    # Extract the specified columns
    extracted_df = df[columns_to_extract].copy()

    # Calculate new features
    feature_df = calculate_new_features(df)

    # Use .loc to avoid SettingWithCopyWarning
    extracted_df.loc[:, new_feature_names[0]] = feature_df['FPOGX-diff']
    extracted_df.loc[:, new_feature_names[1]] = feature_df['FPOGY-diff']

    # Add the label from labels_df
    label = labels_df.at[tester_id, f'Question_{question_idx + 1}']
    extracted_df['label'] = label

    # Save the processed dataframe to a new CSV file
    extracted_df.to_csv(output_path, index=False)


def read_labels(file_path):
    """Read the l_i_j.csv file and return the DataFrame."""
    labels_df = pd.read_csv(file_path, index_col='Tester_ID')
    # Convert the index to integers
    labels_df.index = labels_df.index.astype(int)
    return labels_df


def main():
    base_dir = r'C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data'  # Replace with the path to your base directory
    question_folders = ['P2', 'P3', 'P4']

    # Read the l_i_j.csv file
    labels_file_path = r"C:\Users\lehoa\pythonProject\l_i_j.csv"
    labels_df = read_labels(labels_file_path)

    for question_idx, folder in enumerate(question_folders):
        folder_path = os.path.join(base_dir, folder)
        output_folder_path = os.path.join(base_dir, f'processed_{folder}')
        os.makedirs(output_folder_path, exist_ok=True)

        for tester_id in range(31):  # NUM_TESTER = 31
            input_file_path = os.path.join(folder_path, f'User {tester_id}_all_gaze.csv')
            output_file_path = os.path.join(output_folder_path, f'User {tester_id}_processed_gaze.csv')
            process_gaze_csv(input_file_path, output_file_path, labels_df, tester_id, question_idx)
            print(f'Processed {input_file_path} and saved to {output_file_path}')

    # Print counts of each label
    label_counts = labels_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    print("Label N count:", int(label_counts.get('N', 0)))
    print("Label U count:", int(label_counts.get('U', 0)))
    print("Label I count:", int(label_counts.get('I', 0)))


if __name__ == "__main__":
    main()
