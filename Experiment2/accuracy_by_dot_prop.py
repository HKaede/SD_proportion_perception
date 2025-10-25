"""
Accuracy Analysis by Dot Ratio

This script analyzes participants’ accuracy across 
different dot ratios (10%, 40%, 45%, 50%, 55%, 60%, 90%) and 
visualizes how accuracy varies as a function of stimulus strength.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define dot ratios to analyze
DOT_RATIOS = [0.1, 0.4, 0.45, 0.55, 0.6, 0.9]


def load_and_identify_response_keys(file_path):
    """
    Load participant data and identify response key assignment.
    
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

    # Keep only necessary columns
    df = df[['dot_ratio', 'key_resp.keys']]
    
    return df, dot_res, diamond_res


def calculate_accuracy_by_ratio(df, dot_res, diamond_res, ratio):
    """
    Calculate accuracy for a specific dot ratio.

    Accuracy is defined as choosing the majority color:
    - For ratio < 0.5: choosing diamond is correct
    - For ratio > 0.5: choosing dot is correct

    Args:
        df: DataFrame with trial data
        dot_res: Response key for dot choice
        diamond_res: Response key for diamond choice
        ratio: Dot ratio to analyze
    
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
                # Diamond is majority
                if response == diamond_res:
                    correct += 1
            else:
                # Dot is majority
                if response == dot_res:
                    correct += 1
            
            amount += 1
    
    return correct / amount if amount > 0 else 0


def analyze_all_participants(participants):
    """
    Analyze accuracy across all participants for each ratio.
    
    Args:
        participants: List of file paths to participant data
    
    Returns:
        dict: Dictionary mapping ratio (as percentage) to accuracy array
    """
    accuracy_by_ratio = {int(r * 100): [] for r in DOT_RATIOS}
    
    for file_path in participants:
        df, dot_res, diamond_res = load_and_identify_response_keys(file_path)
        
        for ratio in DOT_RATIOS:
            accuracy = calculate_accuracy_by_ratio(df, dot_res, diamond_res, ratio)
            accuracy_by_ratio[int(ratio * 100)].append(accuracy)
    
    # Convert lists to numpy arrays
    for key in accuracy_by_ratio:
        accuracy_by_ratio[key] = np.array(accuracy_by_ratio[key])
    
    return accuracy_by_ratio


def plot_mean_accuracy(accuracy_by_ratio):
    """
    Create a line plot of mean accuracy as a function of dot ratio.

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
        color='blue',
        ecolor='blue',
        elinewidth=2,
        marker='o',
        markersize=8
    )
    
    plt.xlabel('Proportion of Dots (%)', fontsize=12)
    plt.ylabel('Accuracy (0-1)', fontsize=12)
    plt.tick_params(axis='both', labelsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.show()


def print_summary_statistics(accuracy_by_ratio):
    """
    Print summary statistics for each ratio.
    
    Args:
        accuracy_by_ratio: Dictionary mapping ratio percentage to accuracy arrays
    """
    print("\n=== Accuracy Summary by Dot Ratio ===\n")
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
    print(f"Dot ratios: {[int(r*100) for r in DOT_RATIOS]}%")

    # Analyze all participants
    accuracy_by_ratio = analyze_all_participants(participants)
    
    # Print summary statistics
    print_summary_statistics(accuracy_by_ratio)
    
    # Visualization
    print("\nGenerating mean accuracy plot...")
    plot_mean_accuracy(accuracy_by_ratio)


if __name__ == "__main__":
    main()