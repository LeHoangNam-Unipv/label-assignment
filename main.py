import os
import csv
import pandas as pd

# Global variables
time_taken = [[] for _ in range(31)]
average_reading_time = [0.0, 0.0, 0.0]  # For 3 questions
d_i_j = [[] for _ in range(31)]  # Initialize d_i_j as a 2D list
d_i = [0.0 for _ in range(31)]  # Initialize d_i as a list for each tester
k = 2  # Global variable k set to 2
l_i_j = [[0, 0, 0] for _ in range(31)]  # Initialize l_i_j as a 2D list with default value 0 for each tester and question
a_i_j = [[None, None, None] for _ in range(31)]  # Initialize a_i_j as a 2D list with default value None for each tester and question


def count_rows_in_csv(file_path):
    """Count the number of rows in a CSV file, excluding the header."""
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header
        return sum(1 for row in csvreader)

def calculate_time_taken(row_count):
    """Calculate time taken to read a question based on row count."""
    return row_count / 150

def calculate_average_reading_time():
    """Calculate the average reading time for each question across all participants."""
    global average_reading_time
    num_testers = len(time_taken)
    num_questions = len(time_taken[0]) if num_testers > 0 else 0

    for question_idx in range(num_questions):
        total_time = sum(time_taken[tester_id][question_idx] for tester_id in range(num_testers))
        average_reading_time[question_idx] = total_time / num_testers if num_testers > 0 else 0

def calculate_d_i_j():
    """Calculate d_i_j for each tester and question based on the given formula."""
    global d_i_j
    num_testers = len(time_taken)
    num_questions = len(time_taken[0]) if num_testers > 0 else 0

    for tester_id in range(num_testers):
        for question_idx in range(num_questions):
            if time_taken[tester_id][question_idx] != 0:
                d_i_j_value = (time_taken[tester_id][question_idx] - average_reading_time[question_idx]) / time_taken[tester_id][question_idx]
            else:
                d_i_j_value = 0  # Handle division by zero if time_taken is zero
            d_i_j[tester_id].append(d_i_j_value)

def calculate_d_i():
    """Calculate d_i for each tester as the absolute average of d_i_j."""
    global d_i
    num_testers = len(d_i_j)
    num_questions = len(d_i_j[0]) if num_testers > 0 else 0

    for tester_id in range(num_testers):
        total_d_i_j = sum(d_i_j[tester_id])
        d_i[tester_id] = total_d_i_j / num_questions if num_questions > 0 else 0

def calculate_l_i_j():
    """Calculate l_i_j based on the conditions provided."""
    global l_i_j
    num_testers = len(time_taken)
    num_questions = len(time_taken[0]) if num_testers > 0 else 0

    for tester_id in range(num_testers):
        for question_idx in range(num_questions):
            t_i_j = time_taken[tester_id][question_idx]
            t_j = average_reading_time[question_idx]
            a_i_j_val = a_i_j[tester_id][question_idx]
            d_i_val = d_i[tester_id]

            if a_i_j_val == 0 and t_i_j < (t_j * (1 + d_i_val) / k):
                print(f"t_i_j: {t_i_j}")
                print(f"(t_j * (1 + d_i_val) / k): {t_j * (1 + d_i_val) / k}")
                l_i_j[tester_id][question_idx] = 2
            elif a_i_j_val == 1 and t_i_j > (t_j * (1 + d_i_val)):
                l_i_j[tester_id][question_idx] = 1
            elif a_i_j_val == 0 and t_i_j >= (t_j * (1 + d_i_val) / k):
                l_i_j[tester_id][question_idx] = 1
            else:
                l_i_j[tester_id][question_idx] = 0

def count_labels():
    """Count occurrences of labels 0, 1, and 2 in l_i_j."""
    global l_i_j
    label_counts = {0: 0, 1: 0, 2: 0}

    for labels in l_i_j:
        for label in labels:
            label_counts[label] += 1

    return label_counts

def save_l_i_j_to_csv(file_path):
    """Save l_i_j to a CSV file."""
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Tester_ID', 'Question_1', 'Question_2', 'Question_3'])  # Write header
        for tester_id, labels in enumerate(l_i_j):
            writer.writerow([tester_id] + labels)

def main():
    global time_taken, a_i_j
    base_dir = r'C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data'  # Replace with the path to your base directory
    question_folders = ['P2', 'P3', 'P4']
    num_testers = 31
    row_counts = [[] for _ in range(num_testers)]  # Initialize a 2D list for row counts

    for tester_id in range(num_testers):
        for question_folder in question_folders:
            file_name = f"User {tester_id}_all_gaze.csv"
            folder_path = os.path.join(base_dir, question_folder)
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                row_count = count_rows_in_csv(file_path)
                row_counts[tester_id].append(row_count)
                time_taken[tester_id].append(calculate_time_taken(row_count))
            else:
                print(f"File not found: {file_path}")
                row_counts[tester_id].append(0)  # If file does not exist, append 0
                time_taken[tester_id].append(0)  # If file does not exist, append 0

    calculate_average_reading_time()
    calculate_d_i_j()
    calculate_d_i()

    answers_dir = r'C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data\result'  # Replace with the path to the directory containing the answers CSV file
    answers_file = 'test result.xlsx'  # Name of the answers excel file
    # Read answers Excel file and store answers to a_i_j
    answers_df = pd.read_excel(os.path.join(answers_dir, answers_file))

    for index, row in answers_df.iterrows():
        user_id = int(str(row['User ID']).split('_')[1])  # Extract user ID from 'user_0' format
        a_i_j[user_id][0] = int(row['Q1']) if str(row['Q1']) in ['0', '1'] else None
        a_i_j[user_id][1] = int(row['Q2']) if str(row['Q2']) in ['0', '1'] else None
        a_i_j[user_id][2] = int(row['Q3']) if str(row['Q3']) in ['0', '1'] else None

    calculate_l_i_j()

    # Save l_i_j to a CSV file
    save_l_i_j_to_csv("l_i_j.csv")

if __name__ == "__main__":
    main()