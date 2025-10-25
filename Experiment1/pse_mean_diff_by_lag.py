"""
Temporal Decay of Serial Dependence

This script analyzes how the influence of extreme previous trials (10% vs 90% white)
on the serial dependence decays over time. PSE differences are
calculated for trials occurring 1, 2, and 3 positions after extreme stimuli to
examine the temporal dynamics of serial dependence.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
from scipy.special import erf
from scipy.stats import ttest_1samp


# White ratios used in the experiment
WHITE_RATIOS = np.array([0.1, 0.4, 0.45, 0.5, 0.55, 0.6, 0.9])
EXTREME_RATIOS = [0.1, 0.9]  # Low and high extremes for comparison
LAG_LABELS = ['1-Back', '2-Back', '3-Back']


def get_rows_at_lag(df, values, lag):
    """
    Get rows at a specific lag after rows where 'white_ratio' matches given values.
    
    Args:
        df: DataFrame to search
        values: List of white_ratio values to match
        lag: Number of trials ahead (1 = next trial, 2 = trial after next, etc.)
    
    Returns:
        DataFrame containing rows at the specified lag after matches
    """
    indices = df[df['white_ratio'].isin(values)].index + lag
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


def analyze_pse_at_lag(file_path, lag):
    """
    Analyze PSE values at a specific lag after extreme stimuli.
    
    Args:
        file_path: Path to participant's CSV file
        lag: Number of trials after extreme stimulus (1, 2, or 3)
    
    Returns:
        dict: Dictionary mapping extreme ratio (0.1, 0.9) to PSE value
    """
    df, white_res = load_and_identify_response_keys(file_path)
    
    pse_values = {}
    
    for extreme_ratio in EXTREME_RATIOS:
        # Get trials at specified lag after the extreme ratio
        df_lagged = get_rows_at_lag(df, [extreme_ratio], lag)
        df_lagged = df_lagged.dropna()
        
        # Count white choices for each ratio
        white_counts, total_counts = count_white_choices(df_lagged, white_res)
        
        # Calculate proportion of white choices
        proportions = np.divide(
            white_counts, total_counts,
            out=np.zeros_like(white_counts, dtype=np.float64),
            where=total_counts != 0
        )
        
        # Fit psychometric curve to extract PSE
        pse, sigma = fit_psychometric_curve(WHITE_RATIOS, proportions)
        pse_values[extreme_ratio] = pse
    
    return pse_values


def calculate_delta_pse_for_all_lags(participants):
    """
    Calculate PSE differences (10% - 90%) for all participants at each lag.
    
    Args:
        participants: List of file paths to participant data
    
    Returns:
        dict: Dictionary mapping lag to array of delta PSE values
    """
    delta_pse_by_lag = {lag: [] for lag in range(1, 4)}
    
    for lag in range(1, 4):
        for file_path in participants:
            pse_values = analyze_pse_at_lag(file_path, lag)
            # Calculate difference: PSE after 10% - PSE after 90%
            delta_pse = pse_values[0.1] - pse_values[0.9]
            delta_pse_by_lag[lag].append(delta_pse)
        
        delta_pse_by_lag[lag] = np.array(delta_pse_by_lag[lag]) * 100  # Convert to percentage
    
    return delta_pse_by_lag


def perform_statistical_tests(delta_pse_by_lag):
    """
    Perform one-sample t-tests against 0 (no bias).
    
    Args:
        delta_pse_by_lag: Dictionary mapping lag to delta PSE arrays
    """
    print("\n=== One-Sample t-Tests (test value = 0) ===")
    
    for lag, label in zip(range(1, 4), LAG_LABELS):
        delta_pse = delta_pse_by_lag[lag]
        t_value, p_value = ttest_1samp(delta_pse, 0)
        df = len(delta_pse) - 1
        mean_delta = np.mean(delta_pse)
        sem_delta = scipy.stats.sem(delta_pse)
        
        print(f"\n{label}:")
        print(f"  Mean: {mean_delta:.3f}%")
        print(f"  SEM: {sem_delta:.3f}%")
        print(f"  t({df}) = {t_value:.3f}, p = {p_value:.4f}")


def plot_temporal_decay(delta_pse_by_lag):
    """
    Create bar plot showing temporal decay of PSE difference.
    
    Args:
        delta_pse_by_lag: Dictionary mapping lag to delta PSE arrays
    """
    colors = ['orangered', 'tomato', 'darkorange']
    
    # Prepare data
    x_positions = np.array([0.3, 0.7, 1.1])
    means = [np.mean(delta_pse_by_lag[lag]) for lag in range(1, 4)]
    sems = [scipy.stats.sem(delta_pse_by_lag[lag]) for lag in range(1, 4)]
    
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    ax.bar(x_positions, means, width=0.2, color=colors, 
           yerr=sems, capsize=5, alpha=0.8)
    
    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Add significance marker for 1-Back
    ax.text(0.3, 0.5, "**", ha='center', va='bottom', color='k', fontsize=20)
    
    # Formatting
    ax.set_ylabel('Average PSE Difference (%)', fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(LAG_LABELS)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-4, 8)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # Get script directory and participant files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    participants = glob.glob(os.path.join(script_dir, "processed_data/*.csv"))
    
    print(f"Analyzing temporal decay of sequential effects for {len(participants)} participants...")
    print(f"Comparing trials after {[int(r*100) for r in EXTREME_RATIOS]}% white")
    print(f"Lags analyzed: {LAG_LABELS}")
    
    # Calculate delta PSE for all lags
    print("\nCalculating PSE differences across lags...")
    delta_pse_by_lag = calculate_delta_pse_for_all_lags(participants)
    
    # Statistical tests
    perform_statistical_tests(delta_pse_by_lag)
    
    # Visualization
    print("\nGenerating temporal decay plot...")
    plot_temporal_decay(delta_pse_by_lag)


if __name__ == "__main__":
    main()