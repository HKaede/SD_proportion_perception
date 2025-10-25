"""
Accuracy Analysis by Previous Trial Condition

This script analyzes how the previous trial's stimulus (dot ratio) affects
accuracy on the current trial. A repeated measures ANOVA is performed to test
whether accuracy differs based on the previous trial's stimulus strength.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.anova as anova


# Previous trial ratios to analyze
PREVIOUS_RATIOS = [0.1, 0.4, 0.6, 0.9]
CONDITION_LABELS = ['Prev10%', 'Prev40%', 'Prev60%', 'Prev90%']


def get_next_rows(df, values):
    """
    Get rows following the rows where 'dot_ratio' matches given values.
    
    Args:
        df: DataFrame to search
        values: List of dot_ratio values to match
    
    Returns:
        DataFrame containing the next rows after matches
    """
    indices = df[df['dot_ratio'].isin(values)].index + 1
    # Exclude indices that exceed the DataFrame range
    indices = indices[indices < len(df)]
    return df.loc[indices]


def calculate_accuracy(df, dot_res, diamond_res):
    """
    Calculate accuracy for a set of trials.
    
    Accuracy is defined as choosing the majority color:
    - For ratio < 0.5: choosing diamond is correct
    - For ratio > 0.5: choosing dot is correct
    - For ratio = 0.5: trial is skipped (ambiguous)
    
    Args:
        df: DataFrame with trial data
        dot_res: Response key for dot choice
        diamond_res: Response key for diamond choice

    Returns:
        float: Accuracy (proportion correct)
    """
    correct = 0
    amount = 0
    
    for i in range(df.shape[0]):
        ratio = df.iat[i, 0]
        response = df.iat[i, 1]
        
        # Skip ambiguous trials (50/50)
        if ratio == 0.5:
            continue
        
        # Check if response is correct
        if ratio < 0.5 and response == diamond_res:
            correct += 1
        elif ratio > 0.5 and response == dot_res:
            correct += 1
        
        amount += 1
    
    return correct / amount if amount > 0 else 0


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


def analyze_sequential_effects(file_path):
    """
    Analyze accuracy following different previous trial ratios.
    
    Args:
        file_path: Path to participant's CSV file
    
    Returns:
        dict: Dictionary mapping previous ratio to accuracy
    """
    df, dot_res, diamond_res = load_and_identify_response_keys(file_path)
    
    accuracies = {}
    
    for prev_ratio in PREVIOUS_RATIOS:
        # Get trials following the specified previous ratio
        df_following = get_next_rows(df, [prev_ratio])
        df_following = df_following.dropna()
        
        # Calculate accuracy for those trials
        accuracy = calculate_accuracy(df_following, dot_res, diamond_res)
        accuracies[prev_ratio] = accuracy
    
    return accuracies


def perform_repeated_measures_anova(accuracy_by_condition):
    """
    Perform repeated measures ANOVA on accuracy data.
    
    Args:
        accuracy_by_condition: Dictionary mapping condition to accuracy arrays
    
    Returns:
        ANOVA results object
    """
    n_participants = len(accuracy_by_condition[PREVIOUS_RATIOS[0]])
    n_conditions = len(PREVIOUS_RATIOS)
    
    # Prepare data for ANOVA
    subject_ids = np.tile(np.arange(n_participants), n_conditions)
    condition_ids = np.repeat(np.arange(n_conditions), n_participants)
    
    # Concatenate all accuracy values
    all_accuracies = np.concatenate([
        accuracy_by_condition[ratio] for ratio in PREVIOUS_RATIOS
    ])
    
    # Create DataFrame for ANOVA
    df_anova = pd.DataFrame({
        'Subject': subject_ids,
        'Condition': condition_ids,
        'Accuracy': all_accuracies
    })
    
    # Perform repeated measures ANOVA
    aov = anova.AnovaRM(df_anova, 'Accuracy', 'Subject', ['Condition'])
    result = aov.fit()
    
    return result


def print_anova_results(result):
    """
    Print formatted ANOVA results.
    
    Args:
        result: ANOVA results object
    """
    print("\n=== Repeated Measures ANOVA Results ===")
    print(result)
    
    # Extract and print F-statistic in APA format
    anv = result.anova_table
    print("\n=== F-Statistics (APA Format) ===")
    for index, row in anv.iterrows():
        if not pd.isnull(row['F Value']):
            dfn = int(row['Num DF'])  # Numerator degrees of freedom
            dfd = int(row['Den DF'])  # Denominator degrees of freedom
            f_value = row['F Value']
            p_value = row['Pr > F']
            print(f"{index}: F({dfn}, {dfd}) = {f_value:.2f}, p = {p_value:.4f}")


def plot_sequential_effects(accuracy_by_condition):
    """
    Create visualization of sequential effects on accuracy.
    
    Args:
        accuracy_by_condition: Dictionary mapping condition to accuracy arrays
    """
    colors = ['green', 'yellowgreen', 'deepskyblue', 'royalblue']
    
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    plt.figure(figsize=(6.4, 4.8))
    
    for i, (ratio, label) in enumerate(zip(PREVIOUS_RATIOS, CONDITION_LABELS), 1):
        accuracies = accuracy_by_condition[ratio]
        mean_acc = np.mean(accuracies)
        sem_acc = np.std(accuracies) / np.sqrt(len(accuracies))
        
        plt.errorbar(
            i, mean_acc,
            yerr=sem_acc,
            fmt='o',
            markersize=10,
            markerfacecolor=colors[i-1],
            markeredgecolor=colors[i-1],
            ecolor=colors[i-1],
            elinewidth=3,
            capsize=6
        )
    
    # Axis configuration
    plt.xticks(range(1, len(CONDITION_LABELS) + 1), CONDITION_LABELS, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Accuracy (0-1)', fontsize=12)
    plt.xlabel('Previous Trial Condition', fontsize=12)
    plt.tick_params(axis='both', labelsize=10)
    plt.ylim(0.6, 0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def print_descriptive_statistics(accuracy_by_condition):
    """
    Print descriptive statistics for each condition.
    
    Args:
        accuracy_by_condition: Dictionary mapping condition to accuracy arrays
    """
    print("\n=== Descriptive Statistics ===")
    print(f"{'Condition':<15} {'Mean':<10} {'SD':<10} {'SEM':<10}")
    print("-" * 45)
    
    for ratio, label in zip(PREVIOUS_RATIOS, CONDITION_LABELS):
        accuracies = accuracy_by_condition[ratio]
        mean_acc = np.mean(accuracies)
        sd_acc = np.std(accuracies)
        sem_acc = sd_acc / np.sqrt(len(accuracies))
        
        print(f"{label:<15} {mean_acc:.3f}{'':<6} {sd_acc:.3f}{'':<6} {sem_acc:.3f}")


def main():
    """Main execution function."""
    # Get script directory and participant files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    participants = glob.glob(os.path.join(script_dir, "processed_data/*.csv"))
    
    print(f"Analyzing sequential effects for {len(participants)} participants...")
    print(f"Previous trial conditions: {[int(r*100) for r in PREVIOUS_RATIOS]}%")
    
    # Initialize storage for accuracy by condition
    accuracy_by_condition = {ratio: [] for ratio in PREVIOUS_RATIOS}
    
    # Analyze each participant
    for file_path in participants:
        accuracies = analyze_sequential_effects(file_path)
        for ratio in PREVIOUS_RATIOS:
            accuracy_by_condition[ratio].append(accuracies[ratio])
    
    # Convert to numpy arrays
    for ratio in PREVIOUS_RATIOS:
        accuracy_by_condition[ratio] = np.array(accuracy_by_condition[ratio])
    
    # Print descriptive statistics
    print_descriptive_statistics(accuracy_by_condition)
    
    # Perform repeated measures ANOVA
    print("\nPerforming repeated measures ANOVA...")
    anova_result = perform_repeated_measures_anova(accuracy_by_condition)
    print_anova_results(anova_result)
    
    # Visualization
    print("\nGenerating visualization...")
    plot_sequential_effects(accuracy_by_condition)


if __name__ == "__main__":
    main()