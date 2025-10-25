"""
Analysis of Dot Selection Tendency

This script analyzes the overall tendency for selecting dots across all trials
for each participant, regardless of the dot_ratio. A one-sample t-test is performed
to test whether the mean proportion differs from chance level (0.5).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


def response_count(df, response):
    """
    Calculate the proportion of trials where a specific response was made.
    
    Args:
        df: DataFrame containing response data
        response: Target response key (e.g., 'right' or 'left')
    
    Returns:
        float: Proportion of target response (0-1)
    """
    choice_count = 0
    count = 0
    
    for i in range(df.shape[0]):
        if df.iat[i, 1] == response:
            choice_count += 1
        count += 1
    
    try:
        return choice_count / count
    except ZeroDivisionError:
        return 0


def load_and_process_data(file_path):
    """
    Load participant data and determine response key assignment.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        tuple: (DataFrame, white_response_key, black_response_key)
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

    # Remove practice trials (rows 0-5) and invalid row (216)
    df = df[['dot_ratio', 'key_resp.keys']]
    df = df.drop(index=list(range(0, 6)) + [216], errors='ignore')

    return df, dot_res, diamond_res


def calculate_dot_preference(file_path):
    """
    Calculate the proportion of dot selections for a participant.

    Args:
        file_path: Path to participant's CSV file
    
    Returns:
        float: Proportion of dot selections
    """
    df, dot_res, diamond_res = load_and_process_data(file_path)
    dot_proportion = response_count(df, dot_res)
    return dot_proportion


def perform_statistical_test(dot_proportions):
    """
    Perform one-sample t-test against chance level (0.5).
    
    Args:
        dot_proportions: Array of dot selection proportions
    """
    print("\n=== Statistical Analysis ===")
    print(f"Number of participants: {len(dot_proportions)}")
    print(f"Mean proportion: {np.mean(dot_proportions):.3f}")
    print(f"SD: {np.std(dot_proportions):.3f}")
    print(f"SEM: {np.std(dot_proportions) / np.sqrt(len(dot_proportions)):.3f}")
    
    t_value, p_value = ttest_1samp(dot_proportions, 0.5)
    print(f"\nOne-sample t-test (test value = 0.5):")
    print(f"  df: {len(dot_proportions) - 1}")
    print(f"  t-value: {t_value:.3f}")
    print(f"  p-value: {p_value:.4f}")


def plot_dot_preference(dot_proportions):
    """
    Create visualization of dot preference analysis.

    Args:
        dot_proportions: Array of dot selection proportions
    """
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    plt.figure(figsize=(3, 6))
    
    # Plot mean with error bars (SEM)
    plt.errorbar(
        0.5, np.mean(dot_proportions),
        yerr=np.std(dot_proportions) / np.sqrt(len(dot_proportions)),
        fmt='o',
        markersize=10,
        markerfacecolor='blue',
        markeredgecolor='blue',
        ecolor='blue',
        elinewidth=3,
        capsize=6
    )
    
    # Axis configuration
    plt.xticks([])
    plt.tick_params(axis='both', labelsize=10)
    plt.ylabel('Proportion of Dots Selected (0-1)', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0.4, 0.6)
    
    # Reference line at chance level (0.5)
    plt.hlines(0.5, 0, 3, colors='gray', linestyles=':', linewidth=1)
    
    plt.tight_layout()
    plt.show()
    
def get_dot_proportions():
    """
    Calculates and returns a list of dot selection proportions for all participants.
    """
    # Get script directory and participant files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    participants = glob.glob(os.path.join(script_dir, "processed_data/*.csv"))
    
    # Calculate white dot selection proportions
    dot_proportions = []
    for file_path in participants:
        dot_prop = calculate_dot_preference(file_path)
        dot_proportions.append(dot_prop)

    return np.array(dot_proportions) # Returns a numpy array


def main():
    """Main execution function."""
    # Get script directory and participant files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    participants = glob.glob(os.path.join(script_dir, "processed_data/*.csv"))
    
    print(f"Analyzing white dot preference for {len(participants)} participants...")
    
    # Calculate white dot selection proportion for each participant
    dot_proportions = []
    for file_path in participants:
        dot_prop = calculate_dot_preference(file_path)
        dot_proportions.append(dot_prop)

    dot_proportions = np.array(dot_proportions)

    # Statistical analysis
    perform_statistical_test(dot_proportions)

    # Visualization
    print("\nGenerating visualization...")
    plot_dot_preference(dot_proportions)


if __name__ == "__main__":
    main()