"""
Data Preprocessing Script for Behavioral Experiment Analysis

This script processes participant response data from behavioral experiments,
filtering out outliers based on:
1. Response rate (proportion of no-responses)
2. Accuracy in trials with a 10% or 90% proportion of dots

Valid data files are copied to a 'processed_data' folder.
"""

import os
import shutil
import glob
import numpy as np
import pandas as pd


def get_next_rows(df, values):
    """
    Get rows following the rows where 'dot_ratio' matches given values.
    
    Args:
        df: DataFrame to search
        values: List of values to match in 'dot_ratio' column
    
    Returns:
        DataFrame containing the next rows after matches
    """
    indices = df[df['dot_ratio'].isin(values)].index + 1
    # Exclude indices that exceed the DataFrame range
    indices = indices[indices < len(df)]
    return df.loc[indices]


def load_and_prepare_data(file_path):
    """
    Load CSV file and prepare data for analysis.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        tuple: (DataFrame, dot_response_key, diamond_response_key)
    """
    df = pd.read_csv(file_path, header=0)
    df = df[['dot_ratio', 'key_resp.keys', 'right']]
    
    # Determine response keys based on participant assignment
    if df['right'].str.contains('subject chose dot').any():
        dot_res = 'right'
        diamond_res = 'left'
    elif df['right'].str.contains('subject chose diamond').any():
        dot_res = 'left'
        diamond_res = 'right'
    else:
        dot_res = None
        diamond_res = None
    
    # Remove practice trials and invalid row
    df = df.iloc[6:]
    df = df.drop(216, errors='ignore')
    
    return df, dot_res, diamond_res


def calculate_accuracy_skewed_trials(df, dot_res, diamond_res):
    """
    Calculate accuracy for skewed rate trials (0.1 and 0.9).

    Args:
        df: DataFrame with trial data
        dot_res: Response key for circle choice
        diamond_res: Response key for diamond choice
    
    Returns:
        float: Accuracy ratio (correct/total)
    """
    correct = 0
    amount = 0
    
    for i in range(df.shape[0]):
        dot_ratio = df.iat[i, 0]
        response = df.iat[i, 1]
        
        # Check correctness based on trial type and participant assignment
        if dot_ratio == 0.1:
            if (dot_res == 'left' and response == 'right') or \
               (dot_res == 'right' and response == 'left'):
                correct += 1
            amount += 1
        elif dot_ratio == 0.9:
            if (dot_res == 'left' and response == 'left') or \
               (dot_res == 'right' and response == 'right'):
                correct += 1
            amount += 1
    
    return correct / amount if amount > 0 else 0


def main():
    """Main execution function."""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    participants = glob.glob(os.path.join(script_dir, "data/*.csv"))
    
    # Step 1: Calculate no-response rates and identify outliers
    print("Step 1: Filtering by no-response rate...")
    empty_ratios = []
    
    for file_path in participants:
        df, _, _ = load_and_prepare_data(file_path)
        empty_ratio = df['key_resp.keys'].isnull().sum() / df.shape[0]
        empty_ratios.append(empty_ratio)
    
    empty_arr = np.array(empty_ratios)
    q3, q1 = np.percentile(empty_arr, [75, 25])
    iqr = q3 - q1
    outlier_bottom = q1 - 1.5 * iqr
    outlier_top = q3 + 1.5 * iqr
    print(f'Response rate outlier threshold: {outlier_top:.3f}, {outlier_bottom:.3f}')
    
    # Filter out participants with too many missing responses
    delete_indices = []
    for i, file_path in enumerate(participants):
        df, _, _ = load_and_prepare_data(file_path)
        empty_ratio = df['key_resp.keys'].isnull().sum() / df.shape[0]
        if empty_ratio > outlier_top:
            delete_indices.append(i)
    
    participants_valid_response = np.delete(participants, delete_indices)
    print(f'Removed {len(delete_indices)} participants due to high no-response rate')
    
    # Step 2: Calculate accuracy for trials with highly skewed rates and identify outliers
    print("\nStep 2: Filtering by accuracy on trials with highly skewed rates (0.1, 0.9)...")
    accuracy_scores = []
    
    for file_path in participants_valid_response:
        df, dot_res, diamond_res = load_and_prepare_data(file_path)
        accuracy = calculate_accuracy_skewed_trials(df, dot_res, diamond_res)
        accuracy_scores.append(accuracy)
    
    accuracy_arr = np.array(accuracy_scores)
    q3, q1 = np.percentile(accuracy_arr, [75, 25])
    iqr = q3 - q1
    outlier_threshold = q1 - 1.5 * iqr
    print(f'Accuracy outlier threshold: {outlier_threshold:.3f}')
    
    # Filter out participants with low accuracy
    delete_indices = []
    for i, file_path in enumerate(participants_valid_response):
        df, dot_res, diamond_res = load_and_prepare_data(file_path)
        accuracy = calculate_accuracy_skewed_trials(df, dot_res, diamond_res)
        if accuracy < outlier_threshold:
            delete_indices.append(i)
    
    participants_final = np.delete(participants_valid_response, delete_indices)
    print(f'Removed {len(delete_indices)} participants due to low accuracy')
    print(f'Final valid participants: {len(participants_final)}')
    
    # Step 3: Copy valid files to processed_data folder
    print("\nStep 3: Copying valid data files...")
    dest_folder = os.path.join(script_dir, "processed_data")
    
    # Remove existing folder if present
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    
    # Create new folder
    os.makedirs(dest_folder, exist_ok=True)
    
    # Copy CSV files
    for file_path in participants_final:
        try:
            if os.path.isfile(file_path):
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(dest_folder, file_name)
                shutil.copy(file_path, dest_path)
                print(f"Copied: {file_name} → {dest_folder}")
            else:
                print(f"File does not exist: {file_path}")
        except Exception as e:
            print(f"Error copying {file_path}: {e}")
    
    print(f"\nProcessing complete! Valid data saved to: {dest_folder}")


if __name__ == "__main__":
    main()