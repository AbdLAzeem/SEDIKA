import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier

# Configure Plots
plt.style.use('ggplot')
OUTPUT_DIR = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/plots"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    base_dir = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/processed_data"
    print("Loading datasets...")
    train = pd.read_pickle(os.path.join(base_dir, "train_data.pkl"))
    val = pd.read_pickle(os.path.join(base_dir, "val_data.pkl"))
    return train, val

def plot_correlation(df, name="Train"):
    print(f"Generating Correlation Matrix for {name}...")
    plt.figure(figsize=(12, 10))
    # Select only top 15 features for readability if too many
    cols = df.columns[:15] 
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f'Correlation Matrix ({name} - Top 15 Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"correlation_matrix_{name.lower()}.png"))
    plt.close()

def plot_target_distribution(train_df, val_df):
    print("Generating Target Distribution Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Train
    sns.countplot(x='target', data=train_df, ax=axes[0])
    axes[0].set_title('Train Target Distribution (Balanced)')
    
    # Val
    sns.countplot(x='target', data=val_df, ax=axes[1])
    axes[1].set_title('Validation Target Distribution (Imbalanced)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution_comparison.png"))
    plt.close()

def feature_importance_analysis(X, y):
    print("Calculating Feature Importance...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    
    # Return top features
    return X.columns[indices]

def bivariate_analysis(df, top_features, target_col='target'):
    print("Generating Bivariate Analysis Plots...")
    # Plot Top 2 features vs Target
    top2 = top_features[:2]
    
    for col in top2:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f'{col} vs Target')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"bivariate_{col}_vs_target.png"))
        plt.close()

def run_analysis():
    train_df, val_df = load_data()
    
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    
    # 1. Target Distribution
    plot_target_distribution(train_df, val_df)
    
    # 2. Correlation
    plot_correlation(train_df, "Train")
    
    # 3. Feature Importance
    top_features = feature_importance_analysis(X_train, y_train)
    
    # 4. Bivariate Analysis (Top features)
    bivariate_analysis(train_df, top_features)
    
    print(f"Analysis complete. Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()
