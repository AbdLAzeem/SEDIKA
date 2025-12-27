import pandas as pd
import os

def load_and_analyze_data(filepath):
    """
    Loads the dataset and prints basic information and summary statistics.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    try:
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        
        print("\n" + "="*40)
        print("BASIC INFORMATION")
        print("="*40)
        print(f"Shape: {df.shape}")
        print("\nInfo:")
        print(df.info())
        
        print("\n" + "="*40)
        print("MISSING VALUES")
        print("="*40)
        print(df.isnull().sum())

        print("\n" + "="*40)
        print("SUMMARY STATISTICS")
        print("="*40)
        print(df.describe())

        print("\n" + "="*40)
        print("DATA CLEANING")
        print("="*40)
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows found: {duplicates}")
        
        if duplicates > 0:
            print("Removing duplicates...")
            df_cleaned = df.drop_duplicates()
            print(f"New Shape: {df_cleaned.shape}")
            print("Duplicates removed.")
            return df_cleaned
        else:
            print("No duplicates to remove.")
            return df

        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Using absolute path as per user environment
    FILE_PATH = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/RT_IOT2022.csv"
    load_and_analyze_data(FILE_PATH)
