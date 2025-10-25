"""
Master script to sequentially run all behavioral experiment analysis files.

This script executes:
1. preprocess_outlier_removal.py (Data Cleaning and Preprocessing)
2. response_repetition.py (Response Repetition Analysis)
3. selected_white_dot_ratio.py (Analysis of White Dot Selection Tendency)
4. accuracy_by_white_dot_prop.py (Accuracy Analysis by White Dot Ratio)
5. accuracy_by_condition.py (Accuracy Analysis by Previous Trial Condition)
6. pse_by_condition.py (Point of Subjective Equality (PSE) Analysis with Line Plot)
7. pse_mean_diff_by_lag.py (Temporal Decay of Serial Dependence)


Ensure all original analysis scripts and the 'data' folder are in the same directory.
"""

import sys
import os
import subprocess

# Add the current directory to the Python path to allow importing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_script_by_import(module_name):
    """
    Imports a module and executes its main() function.
    """
    print(f"\n=======================================================")
    print(f"| Starting Analysis: {module_name}.py")
    print(f"=======================================================")
    
    try:
        # Dynamically import the module
        module = __import__(module_name)
        
        # Execute the main function of the imported module
        if hasattr(module, 'main') and callable(module.main):
            module.main()
        else:
            print(f"ERROR: '{module_name}.py' does not have a 'main()' function.")
            
    except ImportError:
        print(f"ERROR: Could not import module '{module_name}'. Check the file name and path.")
    except Exception as e:
        print(f"An error occurred during execution of '{module_name}.py': {e}")
        

def run_script_by_subprocess(script_path):
    """
    Executes a script using a subprocess. This is necessary for 
    'preprocess_outlier_removal.py' to handle its file operations correctly 
    and to prevent side effects in the main script's environment.
    """
    print(f"\n=======================================================")
    print(f"| Starting Preprocessing: {os.path.basename(script_path)}")
    print(f"=======================================================")
    
    try:
        # Use subprocess.run to execute the script
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=False, 
            text=True,
            check=True  # Raise an error if the script fails
        )
        print("Preprocessing completed successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Preprocessing script failed with return code {e.returncode}.")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"ERROR: Python interpreter not found or script path is incorrect: {script_path}")
    except Exception as e:
        print(f"An unexpected error occurred during subprocess execution: {e}")


def main():
    """Defines the order of execution for all analysis scripts."""
    
    # List of all analysis modules (excluding '.py' extension)
    # The order is crucial: Preprocessing must run first to create 'processed_data'.
    
    print("Starting master analysis script...")
    
    # 1. Preprocessing (Must be run first)
    preprocess_script_name = 'preprocess_outlier_removal.py'
    preprocess_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), preprocess_script_name)
    run_script_by_subprocess(preprocess_script_path)
    
    # After preprocessing, the 'processed_data' folder exists, and other scripts can run.
    
    # 2. Sequential Analysis Scripts
    analysis_modules = [
        'response_repetition',
        'selected_white_dot_ratio',
        'accuracy_by_white_dot_prop',
        'accuracy_by_condition',
        'pse_by_condition',
        'pse_mean_diff_by_lag',
    ]
    
    for module in analysis_modules:
        # Note: Importing and running main() is typically faster than subprocess.
        # But ensure file paths within imported scripts work relative to their location,
        # which they seem to in this case (using os.path.dirname(os.path.abspath(__file__))).
        run_script_by_import(module)
        
    print("\n=======================================================")
    print("| All analyses complete.")
    print("=======================================================")


if __name__ == "__main__":
    main()