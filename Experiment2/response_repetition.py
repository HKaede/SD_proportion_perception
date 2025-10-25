"""
Response Repetition Analysis

This script analyzes the tendency for participants to repeat their previous responses
(right or left key press) regardless of stimulus assignment. Statistical tests and 
visualization are performed to evaluate response repetition patterns.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


def get_next_rows(df, values):
    """
    Get rows following the rows where 'key_resp.keys' matches given values.
    
    Args:
        df: DataFrame to search
        values: List of response values to match
    
    Returns:
        DataFrame containing the next rows after matches
    """
    indices = df[df['key_resp.keys'].isin(values)].index + 1
    # Exclude indices that exceed the DataFrame range
    indices = indices[indices < len(df)]
    return df.loc[indices]


def response_count(df, response):
    """
    Calculate the proportion of repeating the previous response.
    
    Args:
        df: DataFrame containing response data
        response: Target response key (e.g., 'right' or 'left')
    
    Returns:
        float: Proportion of repeating the previous response (0-1)
    """
    repeat_count = 0
    amount = 0
    
    for i in range(df.shape[0]):
        if df.iat[i, 1] == response:
            repeat_count += 1
        amount += 1
    
    try:
        return repeat_count / amount
    except ZeroDivisionError:
        return 0


def analyze_repetition_rate(file_path):
    """
    Analyze response repetition rates for a single participant.
    
    Args:
        file_path: Path to participant's CSV file
    
    Returns:
        tuple: (right_repetition_rate, left_repetition_rate)
    """
    df = pd.read_csv(file_path, header=0)
    df = df[['dot_ratio', 'key_resp.keys']]
    
    # Get trials following 'right' responses
    df_right = get_next_rows(df, ['right'])
    df_right = df_right.dropna()
    
    # Get trials following 'left' responses
    df_left = get_next_rows(df, ['left'])
    df_left = df_left.dropna()
    
    # Calculate repetition rates
    right_repeat = response_count(df_right, 'right')
    left_repeat = response_count(df_left, 'left')
    
    return right_repeat, left_repeat


def perform_statistical_tests(right_repeats, left_repeats):
    """
    Perform one-sample t-tests against chance level (0.5).
    
    Args:
        right_repeats: Array of right response repetition rates
        left_repeats: Array of left response repetition rates
    """
    print("\n=== Statistical Analysis ===")
    
    print("\nRight Response Repetition:")
    t_value, p_value = ttest_1samp(right_repeats, 0.5)
    print(f"  Mean: {np.mean(right_repeats):.3f} ± {np.std(right_repeats):.3f}")
    print(f"  df: {len(right_repeats) - 1}")
    print(f"  t-value: {t_value:.3f}")
    print(f"  p-value: {p_value:.4f}")
    
    print("\nLeft Response Repetition:")
    t_value, p_value = ttest_1samp(left_repeats, 0.5)
    print(f"  Mean: {np.mean(left_repeats):.3f} ± {np.std(left_repeats):.3f}")
    print(f"  df: {len(left_repeats) - 1}")
    print(f"  t-value: {t_value:.3f}")
    print(f"  p-value: {p_value:.4f}")


def plot_repetition_patterns(right_repeats, left_repeats):
    """
    Create visualization of response repetition analysis.
    
    Args:
        right_repeats: Array of right response repetition rates
        left_repeats: Array of left response repetition rates
    """
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    plt.figure(figsize=(3, 6))
    
    # Plot right response repetition with error bars
    plt.errorbar(
        1, np.mean(right_repeats),
        yerr=np.std(right_repeats)/ np.sqrt(len(right_repeats)),
        fmt='o',
        markersize=10,
        markerfacecolor='green',      # 塗りなし
        markeredgecolor='green',
        ecolor='green',
        elinewidth=3,
        capsize=6
    )
    
    # Plot left response repetition with error bars
    plt.errorbar(
        2, np.mean(left_repeats),
        yerr=np.std(left_repeats)/ np.sqrt(len(left_repeats)),
        fmt='o',
        markersize=10,
        markerfacecolor='royalblue',      # 塗りなし
        markeredgecolor='royalblue',
        ecolor='royalblue',
        elinewidth=3,
        capsize=6
    )
    
    # Axis configuration
    plt.xticks([1, 2], ['Right', 'Left'], fontsize=10)
    plt.tick_params(axis='both', labelsize=10)
    plt.ylabel('Proportion of Repeated Responses (0-1)', fontsize=12)
    plt.xlabel('Previous Response', fontsize=12)
    plt.ylim(0.4, 0.6)
    plt.xlim(0, 3)
    
    # Reference line at chance level (0.5)
    plt.hlines(0.5, 0, 3, colors='gray', linestyles=':', linewidth=1)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # Get script directory and participant files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    participants = glob.glob(os.path.join(script_dir, "processed_data/*.csv"))
    
    print(f"Analyzing response repetition for {len(participants)} participants...")
    
    # Arrays to store repetition rates
    right_repeats = []
    left_repeats = []
    
    # Process each participant
    for file_path in participants:
        right_repeat, left_repeat = analyze_repetition_rate(file_path)
        right_repeats.append(right_repeat)
        left_repeats.append(left_repeat)
    
    right_repeats = np.array(right_repeats)
    left_repeats = np.array(left_repeats)
    
    # Statistical analysis
    perform_statistical_tests(right_repeats, left_repeats)
    
    # Visualization
    print("\nGenerating visualization...")
    plot_repetition_patterns(right_repeats, left_repeats)


if __name__ == "__main__":
    main()
