import os
import csv
import pandas as pd
import numpy as np


base_dir = r'C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data'  # Replace with the path to your base directory
question_folders = ['P2', 'P3', 'P4']
answers_file = r'result\test result.xlsx'  # Name of the answers excel file

NUM_TESTER = 31
NUM_QUESTION = 3
PERCENTILE_LOWERBOUND = 25
PERCENTILE_UPPERBOUND = 75
VALID_IQR_THRESHOLD = 1.5
def count_rows_in_csv(file_path):
    """Count the number of rows in a CSV file, excluding the header."""
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header
        return sum(1 for row in csvreader)

def calculate_t_i_j(row_count):
    """Calculate time taken to read a question based on row count."""
    return row_count / 150

def calculate_t_j(t_i_j):
    """Calculate the average reading time for each question across all participants."""
    t_j = np.zeros(NUM_QUESTION, dtype=float)
    for question_idx in range(NUM_QUESTION):
        t_j[question_idx] = np.median(t_i_j[:, question_idx])
    return t_j

def read_csv_gaze_to_extract_reading_time(base_dir, question_folders):
    row_counts = np.zeros((NUM_TESTER, NUM_QUESTION), dtype=int)  # Initialize a 2D list for row counts
    t_i_j = np.zeros((NUM_TESTER, NUM_QUESTION), dtype=float)  # t_i_j

    for tester_id in range(NUM_TESTER):
        q = 0
        for question_folder in question_folders:
            file_name = f"User {tester_id}_all_gaze.csv"
            folder_path = os.path.join(base_dir, question_folder)
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                row_count = count_rows_in_csv(file_path)
                row_counts[tester_id,q] = row_count
                t_i_j[tester_id,q] = calculate_t_i_j(row_count)
            else:
                print(f"File not found: {file_path}")
                row_counts[tester_id,q] = 0  # If file does not exist, append 0
                t_i_j[tester_id,q] = 0  # If file does not exist, append 0
            q += 1
    return t_i_j

def read_answer_file(answers_dir, answers_file):
    a_i_j = np.zeros((NUM_TESTER, NUM_QUESTION), dtype=int)  # Read answers Excel file and store answers to a_i_j

    answers_df = pd.read_excel(os.path.join(answers_dir, answers_file))

    for index, row in answers_df.iterrows():
        user_id = int(str(row['User ID']).split('_')[1])  # Extract user ID from 'user_0' format
        a_i_j[user_id,0] = int(row['Q1'])
        a_i_j[user_id,1] = int(row['Q2'])
        a_i_j[user_id,2] = int(row['Q3'])
    return a_i_j

def save_l_i_j_to_csv(file_path, l_i_j):
    """Save l_i_j to a CSV file."""
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Tester_ID', 'Question_1', 'Question_2', 'Question_3'])  # Write header
        for tester_id, label in enumerate(l_i_j):
            writer.writerow([str(tester_id)] + list(label))

def calculate_iqr_j(t_i_j):
    iqr_j = np.zeros(NUM_QUESTION, dtype=float)
    for question_idx in range(NUM_QUESTION):
        q1_j = np.percentile(t_i_j[:, question_idx], PERCENTILE_LOWERBOUND)
        q3_j = np.percentile(t_i_j[:, question_idx], PERCENTILE_UPPERBOUND)
        iqr_j[question_idx] = q3_j - q1_j
    return iqr_j

def find_invalid_testers(a_i_j, t_i_j, lf_j, lf_cj):
    """Find invalid testers based on the condition a_i_j = 0 and t_i_j < LFj."""
    invalid_testers_with_questions = []
    for tester_id in range(NUM_TESTER):
        for question_idx in range(NUM_QUESTION):
            if a_i_j[tester_id, question_idx] == 0 and t_i_j[tester_id, question_idx] < lf_j[question_idx]:
                invalid_testers_with_questions.append((tester_id, question_idx))
            elif a_i_j[tester_id, question_idx] == 1 and t_i_j[tester_id, question_idx] > lf_cj[question_idx]:
                invalid_testers_with_questions.append((tester_id, question_idx))
    return invalid_testers_with_questions


def calculate_iqr_cj(t_i_j, a_i_j, invalid_testers_with_questions):
    """
    Calculate the interquartile range (IQR) for each question considering only valid answers (a_i_j = 1)
    and excluding invalid testers.
    """
    iqr_cj = np.zeros(NUM_QUESTION, dtype=float)

    # Create a set of (tester_id, question_idx) pairs for quick lookup of invalid questions
    invalid_questions_set = set(invalid_testers_with_questions)

    for question_idx in range(NUM_QUESTION):
        valid_times = []
        for tester_id in range(NUM_TESTER):
            if a_i_j[tester_id, question_idx] == 1 and (tester_id, question_idx) not in invalid_questions_set:
                valid_times.append(t_i_j[tester_id, question_idx])

        q1_cj = np.percentile(valid_times, PERCENTILE_LOWERBOUND)
        q3_cj = np.percentile(valid_times, PERCENTILE_UPPERBOUND)
        iqr_cj[question_idx] = q3_cj - q1_cj

    return iqr_cj

def calculate_uf_cj(t_i_j, a_i_j, invalid_testers_with_questions):
    """
    Calculate the interquartile range (IQR) for each question considering only correct answers (a_i_j = 1)
    and excluding invalid testers.
    """
    iqr_cj = np.zeros(NUM_QUESTION, dtype=float)
    uf_cj = np.zeros(NUM_QUESTION, dtype=float)

    # Create a set of (tester_id, question_idx) pairs for quick lookup of invalid questions
    invalid_questions_set = set(invalid_testers_with_questions)

    for question_idx in range(NUM_QUESTION):
        valid_times = []
        for tester_id in range(NUM_TESTER):
            if a_i_j[tester_id, question_idx] == 1 and (tester_id, question_idx) not in invalid_questions_set:
                valid_times.append(t_i_j[tester_id, question_idx])

        q1_cj = np.percentile(valid_times, PERCENTILE_LOWERBOUND)
        q3_cj = np.percentile(valid_times, PERCENTILE_UPPERBOUND)
        iqr_cj[question_idx] = q3_cj - q1_cj
        uf_cj[question_idx] = q3_cj + (VALID_IQR_THRESHOLD * iqr_cj[question_idx])

    return uf_cj

def calculate_lf_j(t_i_j):
    iqr_j = np.zeros(NUM_QUESTION, dtype=float)
    lf_j = np.zeros(NUM_QUESTION, dtype=float)
    for question_idx in range(NUM_QUESTION):
        q1_j = np.percentile(t_i_j[:, question_idx], PERCENTILE_LOWERBOUND)
        q3_j = np.percentile(t_i_j[:, question_idx], PERCENTILE_UPPERBOUND)
        iqr_j[question_idx] = q3_j - q1_j
        lf_j[question_idx] = q3_j - (VALID_IQR_THRESHOLD * iqr_j[question_idx])
    return lf_j

def calculate_lf_cj(a_i_j, t_i_j):
    iqr_cj = np.zeros(NUM_QUESTION, dtype=float)
    lf_cj = np.zeros(NUM_QUESTION, dtype=float)
    for question_idx in range(NUM_QUESTION):
        valid_times = []
        for tester_id in range(NUM_TESTER):
            if a_i_j[tester_id, question_idx] == 1:
                valid_times.append(t_i_j[tester_id, question_idx])

        q1_cj = np.percentile(valid_times, PERCENTILE_LOWERBOUND)
        q3_cj = np.percentile(valid_times, PERCENTILE_UPPERBOUND)
        iqr_cj[question_idx] = q3_cj - q1_cj
        lf_cj[question_idx] = q3_cj + (VALID_IQR_THRESHOLD * iqr_cj[question_idx])
    return lf_cj

def calculate_t_cj(t_i_j, a_i_j, invalid_testers_with_questions):
    """
    Calculate the median reading time for each question considering only valid answers (a_i_j = 1)
    and excluding invalid testers.
    """
    t_cj = np.zeros(NUM_QUESTION, dtype=float)

    # Create a set of (tester_id, question_idx) pairs for quick lookup of invalid questions
    invalid_questions_set = set(invalid_testers_with_questions)

    for question_idx in range(NUM_QUESTION):
        valid_times = []
        for tester_id in range(NUM_TESTER):
            if a_i_j[tester_id, question_idx] == 1 and (tester_id, question_idx) not in invalid_questions_set:
                valid_times.append(t_i_j[tester_id, question_idx])

        t_cj[question_idx] = np.median(valid_times)

    return t_cj


def calculate_l_i_j(a_i_j, t_i_j, invalid_testers_with_questions, uf_cj):
    """
    Calculate l_i_j based on the provided formula.
    """
    l_i_j = np.empty((NUM_TESTER, NUM_QUESTION), dtype=str)

    # Create a set of (tester_id, question_idx) pairs for quick lookup of invalid questions
    invalid_questions_set = set(invalid_testers_with_questions)

    for tester_id in range(NUM_TESTER):
        for question_idx in range(NUM_QUESTION):
            if (tester_id, question_idx) in invalid_questions_set:
                l_i_j[tester_id,question_idx] = "I"
            else:
                if a_i_j[tester_id,question_idx] == 1 and t_i_j[tester_id,question_idx] > uf_cj[question_idx]:
                    l_i_j[tester_id,question_idx] = "U"
                elif a_i_j[tester_id,question_idx] == 0:
                    l_i_j[tester_id,question_idx] = "U"
                else:
                    l_i_j[tester_id,question_idx] = "N"  # Otherwise

    return l_i_j


def main():

    t_i_j = read_csv_gaze_to_extract_reading_time(base_dir, question_folders)
    print(t_i_j)
    a_i_j = read_answer_file(base_dir, answers_file)
    print(a_i_j)
    t_j = calculate_t_j(t_i_j)

    lf_j = calculate_lf_j(t_i_j)
    print(lf_j)

    lf_cj = calculate_lf_cj(a_i_j,t_i_j)

    invalid_testers = find_invalid_testers(a_i_j, t_i_j, lf_j, lf_cj)
    print(invalid_testers)
    t_cj = calculate_t_cj(t_i_j, a_i_j, invalid_testers)

    uf_cj = calculate_uf_cj(t_i_j, a_i_j, invalid_testers)
    print(uf_cj)
    l_i_j = calculate_l_i_j(a_i_j, t_i_j, invalid_testers, uf_cj)

    # Save l_i_j to a CSV file
    save_l_i_j_to_csv("l_i_j.csv", l_i_j)

main()