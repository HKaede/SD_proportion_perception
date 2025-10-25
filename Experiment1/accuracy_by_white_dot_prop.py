"""
Accuracy Analysis by White Dot Ratio

This script analyzes participants’ accuracy across 
different white dot ratios (10%, 40%, 45%, 50%, 55%, 60%, 90%) and 
visualizes how accuracy varies as a function of stimulus strength.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define white ratios to analyze
WHITE_RATIOS = [0.1, 0.4, 0.45, 0.55, 0.6, 0.9]


def load_and_identify_response_keys(file_path):
    """
    Load participant data and identify response key assignment.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        tuple: (DataFrame, white_response_key, black_response_key)
    """
    df = pd.read_csv(file_path, header=0)
    df = df[['white_ratio', 'key_resp.keys', 'right']]
    
    # Determine response keys based on participant assignment
    if df['right'].str.contains('subject chose white').any():
        white_res = 'right'
        black_res = 'left'
    elif df['right'].str.contains('subject chose black').any():
        white_res = 'left'
        black_res = 'right'
    else:
        white_res = None
        black_res = None
    
    # Keep only necessary columns
    df = df[['white_ratio', 'key_resp.keys']]
    
    return df, white_res, black_res


def calculate_accuracy_by_ratio(df, white_res, black_res, ratio):
    """
    Calculate accuracy for a specific white dot ratio.
    
    Accuracy is defined as choosing the majority color:
    - For ratio < 0.5: choosing black is correct
    - For ratio > 0.5: choosing white is correct
    
    Args:
        df: DataFrame with trial data
        white_res: Response key for white choice
        black_res: Response key for black choice
        ratio: White dot ratio to analyze
    
    Returns:
        float: Accuracy (proportion correct)
    """
    correct = 0
    amount = 0
    
    for i in range(df.shape[0]):
        if df.iat[i, 0] == ratio:
            response = df.iat[i, 1]
            
            # Determine correct response based on majority color
            if ratio < 0.5:
                # Black is majority
                if response == black_res:
                    correct += 1
            else:
                # White is majority
                if response == white_res:
                    correct += 1
            
            amount += 1
    
    return correct / amount if amount > 0 else 0


def analyze_all_participants(participants):
    """
    Analyze accuracy across all participants for each white ratio.
    
    Args:
        participants: List of file paths to participant data
    
    Returns:
        dict: Dictionary mapping ratio (as percentage) to accuracy array
    """
    accuracy_by_ratio = {int(r * 100): [] for r in WHITE_RATIOS}
    
    for file_path in participants:
        df, white_res, black_res = load_and_identify_response_keys(file_path)
        
        for ratio in WHITE_RATIOS:
            accuracy = calculate_accuracy_by_ratio(df, white_res, black_res, ratio)
            accuracy_by_ratio[int(ratio * 100)].append(accuracy)
    
    # Convert lists to numpy arrays
    for key in accuracy_by_ratio:
        accuracy_by_ratio[key] = np.array(accuracy_by_ratio[key])
    
    return accuracy_by_ratio


def plot_mean_accuracy(accuracy_by_ratio):
    """
    Create a line plot of mean accuracy as a function of white dot ratio.

    Args:
        accuracy_by_ratio: Dictionary mapping ratio percentage to accuracy arrays
    """
    # Prepare data for plotting
    x = []
    y = []
    yerr = []
    
    for ratio_pct, accuracies in sorted(accuracy_by_ratio.items()):
        x.append(ratio_pct)
        y.append(np.mean(accuracies))
        yerr.append(np.std(accuracies) / np.sqrt(len(accuracies)))  # Standard error
    
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Create plot
    plt.figure(figsize=(6.4, 4.8))
    plt.errorbar(
        x, y, yerr=yerr,
        fmt='-',
        capsize=5,
        color='red',
        ecolor='red',
        elinewidth=2,
        marker='o',
        markersize=8
    )
    
    plt.xlabel('Proportion of White Dots (%)', fontsize=12)
    plt.ylabel('Accuracy (0-1)', fontsize=12)
    plt.tick_params(axis='both', labelsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.show()


def print_summary_statistics(accuracy_by_ratio):
    """
    Print summary statistics for each white ratio.
    
    Args:
        accuracy_by_ratio: Dictionary mapping ratio percentage to accuracy arrays
    """
    print("\n=== Accuracy Summary by White Dot Ratio ===\n")
    print(f"{'Ratio':<10} {'Mean':<10} {'SD':<10} {'SEM':<10} {'N':<10}")
    print("-" * 50)
    
    for ratio_pct, accuracies in sorted(accuracy_by_ratio.items()):
        mean_acc = np.mean(accuracies)
        sd_acc = np.std(accuracies)
        sem_acc = sd_acc / np.sqrt(len(accuracies))
        n = len(accuracies)
        
        print(f"{ratio_pct}%{'':<7} {mean_acc:.3f}{'':<6} {sd_acc:.3f}{'':<6} "
              f"{sem_acc:.3f}{'':<6} {n:<10}")


def main():
    """Main execution function."""
    # Get script directory and participant files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    participants = glob.glob(os.path.join(script_dir, "processed_data/*.csv"))
    
    print(f"Analyzing accuracy for {len(participants)} participants...")
    print(f"White dot ratios: {[int(r*100) for r in WHITE_RATIOS]}%")
    
    # Analyze all participants
    accuracy_by_ratio = analyze_all_participants(participants)
    
    # Print summary statistics
    print_summary_statistics(accuracy_by_ratio)
    
    # Visualization
    print("\nGenerating mean accuracy plot...")
    plot_mean_accuracy(accuracy_by_ratio)


if __name__ == "__main__":
    main()