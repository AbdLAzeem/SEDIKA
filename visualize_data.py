import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Append current directory to path to import load_data
sys.path.append(os.getcwd())
try:
    from load_data import load_and_analyze_data
except ImportError:
    pass

def load_clean_data(filepath):
    """Loads and returns cleaned data."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if df.duplicated().sum() > 0:
            df = df.drop_duplicates()
        return df
    return None

def plot_class_distribution(df, output_dir):
    """Plots and saves the distribution of the target variable."""
    plt.figure(figsize=(12, 6))
    if 'Attack_type' in df.columns:
        counts = df['Attack_type'].value_counts()
        sns.barplot(x=counts.index, y=counts.values, palette='viridis')
        plt.title('Distribution of Attack Types')
        plt.xlabel('Attack Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
        plt.close()
        print("Saved class_distribution.png")

def plot_correlation_heatmap(df, output_dir):
    """Plots and saves a correlation heatmap for numerical features."""
    plt.figure(figsize=(14, 12))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    # Drop constant columns if any
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
    
    # Calculate correlation
    corr = numeric_df.corr()
    
    # Select features with high correlation to the target or just general heatmap?
    # Since we have many features (80+), a full heatmap is messy. 
    # Let's show the heatmap of features that have at least some high correlation with others, 
    # or just the top 20 most variable features for readability.
    # Alternatively, simply plotting the full matrix but without annotations if it's large.
    
    # For better visualization, let's pick columns that have a correlation > 0.5 with at least one other column (excluding self)
    # broad filter
    mask = pd.np.triu(pd.np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Heatmap (Numerical Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Saved correlation_heatmap.png")

def calculate_feature_importance(df, output_dir):
    """Calculates feature importance using Random Forest and plots it."""
    print("Calculating feature importance...")
    
    # Preprocessing
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['Attack_type'])
    
    # Drop non-numeric for X (simplification)
    X = df.select_dtypes(include=['float64', 'int64'])
    
    # Handle any NaN if present (though we checked they are 0)
    X = X.fillna(0)
    
    # Use a small forest for speed
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot Top 20 Features
    top_n = 20
    plt.figure(figsize=(12, 8))
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), [X.columns[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    print("Saved feature_importance.png")
    
    # Save text list
    with open(os.path.join(output_dir, 'feature_importance_list.txt'), 'w') as f:
        f.write("Feature Importance Ranking:\n")
        for i in range(len(importances)):
            f.write(f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.6f}\n")

if __name__ == "__main__":
    file_path = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/RT_IOT2022.csv"
    output_dir = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/plots"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = load_clean_data(file_path)
    if df is not None:
        plot_class_distribution(df, output_dir)
        plot_correlation_heatmap(df, output_dir)
        calculate_feature_importance(df, output_dir)
    else:
        print("Failed to load data.")
