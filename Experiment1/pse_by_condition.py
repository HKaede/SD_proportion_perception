"""
Point of Subjective Equality (PSE) Analysis with Line Plot

This script analyzes how the previous trial's stimulus affects the Point of 
Subjective Equality (PSE). Results are visualized as a line plot showing how
PSE shifts across different previous trial conditions. Statistical tests include
one-sample t-tests, repeated measures ANOVA, and Tukey HSD post-hoc comparisons.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.special import erf
from scipy.stats import ttest_1samp
import statsmodels.stats.anova as anova
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from selected_white_dot_ratio import get_white_proportions


# White ratios used in the experiment
WHITE_RATIOS = np.array([0.1, 0.4, 0.45, 0.5, 0.55, 0.6, 0.9])
PREVIOUS_RATIOS = [0.1, 0.4, 0.6, 0.9]
CONDITION_LABELS = ['Prev10%', 'Prev40%', 'Prev60%', 'Prev90%']


def get_next_rows(df, values):
    """
    Get rows following the rows where 'white_ratio' matches given values.
    
    Args:
        df: DataFrame to search
        values: List of white_ratio values to match
    
    Returns:
        DataFrame containing the next rows after matches
    """
    indices = df[df['white_ratio'].isin(values)].index + 1
    # Exclude indices that exceed the DataFrame range
    indices = indices[indices < len(df)]
    return df.loc[indices]


def gaussian_cdf(x, mu, sigma):
    """
    Cumulative distribution function of Gaussian distribution.
    
    Args:
        x: Input values
        mu: Mean (PSE)
        sigma: Standard deviation (psychometric slope)
    
    Returns:
        CDF values
    """
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def objective_function(variables, x, y):
    """
    Objective function for fitting psychometric curve.
    
    Args:
        variables: [mu, sigma] parameters to optimize
        x: White ratios
        y: Proportion of white choices
    
    Returns:
        Root mean squared error
    """
    mu = variables[0]
    sigma = variables[1]
    y_est = gaussian_cdf(x, mu, sigma)
    d = y - y_est
    return np.sqrt(np.dot(d, d))


def fit_psychometric_curve(x, y):
    """
    Fit Gaussian CDF to psychometric data to extract PSE.
    
    Args:
        x: White ratios
        y: Proportion of white choices
    
    Returns:
        tuple: (PSE, sigma) - Point of Subjective Equality and slope
    """
    # Try two different initial values to avoid local minima
    res1 = scipy.optimize.minimize(
        objective_function, [0.3, 1], args=(x, y),
        method="L-BFGS-B",
        bounds=[(0.3, 0.7), (None, None)]
    )
    res2 = scipy.optimize.minimize(
        objective_function, [0.7, 1], args=(x, y),
        method="L-BFGS-B",
        bounds=[(0.3, 0.7), (None, None)]
    )
    
    # Choose the result with lower error
    res = res1 if res1.fun < res2.fun else res2
    
    return res.x[0], res.x[1]


def count_white_choices(df, white_res):
    """
    Count white choices for each white ratio.
    
    Args:
        df: DataFrame with trial data
        white_res: Response key for white choice
    
    Returns:
        tuple: (white_choice_counts, total_counts) for each ratio
    """
    white_counts = np.zeros(len(WHITE_RATIOS))
    total_counts = np.zeros(len(WHITE_RATIOS))
    
    ratio_to_index = {ratio: i for i, ratio in enumerate(WHITE_RATIOS)}
    
    for i in range(df.shape[0]):
        ratio = df.iat[i, 0]
        response = df.iat[i, 1]
        
        if ratio in ratio_to_index:
            idx = ratio_to_index[ratio]
            if response == white_res:
                white_counts[idx] += 1
            total_counts[idx] += 1
    
    return white_counts, total_counts


def load_and_identify_response_keys(file_path):
    """
    Load participant data and identify response key assignment.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        tuple: (DataFrame, white_response_key)
    """
    df = pd.read_csv(file_path, header=0)
    df = df[['white_ratio', 'key_resp.keys', 'right']]
    
    # Determine response keys based on participant assignment
    if df['right'].str.contains('subject chose white').any():
        white_res = 'right'
    elif df['right'].str.contains('subject chose black').any():
        white_res = 'left'
    else:
        white_res = None
    
    # Keep only necessary columns
    df = df[['white_ratio', 'key_resp.keys']]
    
    return df, white_res


def analyze_pse_for_participant(file_path):
    """
    Analyze PSE values following different previous trial conditions.
    
    Args:
        file_path: Path to participant's CSV file
    
    Returns:
        dict: Dictionary mapping previous ratio to PSE value
    """
    df, white_res = load_and_identify_response_keys(file_path)
    
    pse_values = {}
    
    for prev_ratio in PREVIOUS_RATIOS:
        # Get trials following the specified previous ratio
        df_following = get_next_rows(df, [prev_ratio])
        df_following = df_following.dropna()
        
        # Count white choices for each ratio
        white_counts, total_counts = count_white_choices(df_following, white_res)
        
        # Calculate proportion of white choices
        proportions = np.divide(
            white_counts, total_counts,
            out=np.zeros_like(white_counts, dtype=np.float64),
            where=total_counts != 0
        )
        
        # Fit psychometric curve to extract PSE
        pse, sigma = fit_psychometric_curve(WHITE_RATIOS, proportions)
        pse_values[prev_ratio] = pse
    
    return pse_values

def get_mean_white_dot_selection_proportion():
    """
    Retrieves the white dot selection proportions and calculates their mean.
    """
    # Call the function inside selected_white_dot_ratio.py to get the list of proportions (np.array)
    white_proportions_array = get_white_proportions()
    
    if white_proportions_array.size == 0:
        print("Data files were not found, or no data was processed.")
        return
    
    # Calculate the mean
    mean_proportion = np.mean(white_proportions_array)
    
    return mean_proportion


def perform_statistical_tests(pse_by_condition):
    """
    Perform one-sample t-tests against the mean proportion of “more white dots” responses (no bias).
    
    Args:
        pse_by_condition: Dictionary mapping condition to PSE arrays
    """
    print(f"\n=== One-Sample t-Tests (test value = {get_mean_white_dot_selection_proportion()}) ===")

    for ratio, label in zip(PREVIOUS_RATIOS, CONDITION_LABELS):
        pse_values = pse_by_condition[ratio]
        t_stat, p_val = ttest_1samp(pse_values, get_mean_white_dot_selection_proportion())
        df = len(pse_values) - 1
        mean_pse = np.mean(pse_values)
        print(f"{label}: t({df}) = {t_stat:.3f}, p = {p_val:.4f}, M = {mean_pse:.3f}")


def perform_repeated_measures_anova(pse_by_condition):
    """
    Perform repeated measures ANOVA on PSE values.
    
    Args:
        pse_by_condition: Dictionary mapping condition to PSE arrays
    
    Returns:
        tuple: (ANOVA results, DataFrame for post-hoc tests)
    """
    n_participants = len(pse_by_condition[PREVIOUS_RATIOS[0]])
    
    # Prepare data for ANOVA
    subject_ids = np.tile(np.arange(n_participants), len(PREVIOUS_RATIOS))
    condition_labels = np.concatenate([
        [label] * n_participants for label in CONDITION_LABELS
    ])
    all_pse = np.concatenate([
        pse_by_condition[ratio] for ratio in PREVIOUS_RATIOS
    ])
    
    # Create DataFrame for ANOVA
    df_anova = pd.DataFrame({
        'Subject': subject_ids,
        'Condition': condition_labels,
        'PSE': all_pse
    })
    
    # Perform repeated measures ANOVA
    aov = anova.AnovaRM(df_anova, 'PSE', 'Subject', ['Condition'])
    result = aov.fit()
    
    return result, df_anova


def print_anova_results(result):
    """
    Print formatted ANOVA results.
    
    Args:
        result: ANOVA results object
    """
    print("\n=== Repeated Measures ANOVA ===")
    print(result)
    
    # Extract and print F-statistic in APA format
    anv = result.anova_table
    print("\n=== F-Statistics (APA Format) ===")
    for index, row in anv.iterrows():
        if not pd.isnull(row['F Value']):
            dfn = int(row['Num DF'])
            dfd = int(row['Den DF'])
            f_value = row['F Value']
            p_value = row['Pr > F']
            print(f"{index}: F({dfn}, {dfd}) = {f_value:.2f}, p = {p_value:.4f}")


def perform_posthoc_test(df_anova):
    """
    Perform Tukey HSD post-hoc test.
    
    Args:
        df_anova: DataFrame prepared for ANOVA
    """
    print("\n=== Post-Hoc Test (Tukey HSD) ===")
    tukey_result = pairwise_tukeyhsd(df_anova['PSE'], df_anova['Condition'])
    print(tukey_result)


def plot_pse_line(pse_by_condition):
    """
    Create line plot visualization of PSE values across conditions.
    
    Args:
        pse_by_condition: Dictionary mapping condition to PSE arrays
    """
    colors = ['orangered', 'tomato', 'darkorange', 'orange']
    
    # Convert to percentages and calculate statistics
    means = [np.mean(pse_by_condition[ratio]) * 100 for ratio in PREVIOUS_RATIOS]
    sems = [np.std(pse_by_condition[ratio]) / np.sqrt(len(pse_by_condition[ratio])) * 100 
            for ratio in PREVIOUS_RATIOS]
    
    x_positions = [1, 2, 3, 4]
    
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica' 
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    # Plot error bars and points
    for i, (x, y, y_err, color) in enumerate(zip(x_positions, means, sems, colors)):
        ax.errorbar(
            x, y,
            yerr=y_err,
            fmt='o',
            capsize=5,
            color=color,
            ecolor=color,
            elinewidth=2,
            markersize=8
        )
    
    # Connect points with line segments
    ax.plot(
        x_positions,
        means,
        color='gray',
        linewidth=2.5,
        zorder=1 
    )
    
    # Add statistical comparison line
    x1, x2 = 1, 4
    y_line = 52.5
    h = 1
    ax.plot([x1, x1, x2, x2], [y_line, y_line+h, y_line+h, y_line], 
            lw=1.5, c='k')
    ax.text((x1+x2)*.5, y_line+h, "**", ha='center', va='bottom', 
            color='k', fontsize=20)
    
    # Formatting
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(CONDITION_LABELS, fontsize=10)
    ax.set_ylabel('PSE (%)', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_ylim(40, 60)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # Get script directory and participant files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    participants = glob.glob(os.path.join(script_dir, "processed_data/*.csv"))
    
    print(f"Analyzing PSE for {len(participants)} participants...")
    print(f"Previous trial conditions: {[int(r*100) for r in PREVIOUS_RATIOS]}%")
    
    # Initialize storage for PSE by condition
    pse_by_condition = {ratio: [] for ratio in PREVIOUS_RATIOS}
    
    # Analyze each participant
    for file_path in participants:
        pse_values = analyze_pse_for_participant(file_path)
        for ratio in PREVIOUS_RATIOS:
            pse_by_condition[ratio].append(pse_values[ratio])
    
    # Convert to numpy arrays
    for ratio in PREVIOUS_RATIOS:
        pse_by_condition[ratio] = np.array(pse_by_condition[ratio])
    
    # Statistical tests
    perform_statistical_tests(pse_by_condition)
    
    # Repeated measures ANOVA
    print("\nPerforming repeated measures ANOVA...")
    anova_result, df_anova = perform_repeated_measures_anova(pse_by_condition)
    print_anova_results(anova_result)
    
    # Post-hoc test
    perform_posthoc_test(df_anova)
    
    # Visualization
    print("\nGenerating line plot...")
    plot_pse_line(pse_by_condition)


if __name__ == "__main__":
    main()