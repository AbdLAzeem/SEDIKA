import pandas as pd
import os
from paths import PROCESSED_DIR

def check_stats():
    train_path = os.path.join(PROCESSED_DIR, "train_data.pkl")
    val_path = os.path.join(PROCESSED_DIR, "val_data.pkl")
    test_path = os.path.join(PROCESSED_DIR, "test_data.pkl")
    
    print("Loading Processed Data to generate statistics...")
    try:
        df_train = pd.read_pickle(train_path)
        df_val = pd.read_pickle(val_path)
        df_test = pd.read_pickle(test_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("\n" + "="*50)
    print("TRAINING SET (Processed & Balanced)")
    print("="*50)
    print("\n--- Basic Info ---")
    df_train.info()
    print("\n--- Summary Statistics ---")
    print(df_train.describe())

    print("\n" + "="*50)
    print("VALIDATION SET (Processed & Imbalanced)")
    print("="*50)
    print("\n--- Basic Info ---")
    df_val.info()
    print("\n--- Summary Statistics ---")
    print(df_val.describe())

    print("\n" + "="*50)
    print("TESTING SET (Processed & Imbalanced)")
    print("="*50)
    print("\n--- Basic Info ---")
    df_test.info()
    print("\n--- Summary Statistics ---")
    print(df_test.describe())

if __name__ == "__main__":
    check_stats()
