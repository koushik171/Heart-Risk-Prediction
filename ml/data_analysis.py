import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data():
    """Load the heart disease dataset"""
    df = pd.read_excel('../public/data/heart_risk_dataset.xlsx')
    return df

def basic_analysis(df):
    """Perform basic data analysis"""
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    if 'target' in df.columns:
        print(f"\nTarget Distribution:")
        print(df['target'].value_counts())
        print(f"Heart Disease Rate: {df['target'].mean():.2%}")

def correlation_analysis(df):
    """Analyze correlations between features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()

def feature_distribution(df):
    """Plot feature distributions"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.show()

def target_analysis(df):
    """Analyze target variable relationships"""
    if 'target' not in df.columns:
        print("No target column found")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'target']
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df.boxplot(column=col, by='target', ax=axes[i])
            axes[i].set_title(f'{col} by Heart Disease')
            axes[i].set_xlabel('Heart Disease')
            axes[i].set_ylabel(col)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('target_analysis.png')
    plt.show()

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    print("Performing basic analysis...")
    basic_analysis(df)
    
    print("\nGenerating visualizations...")
    correlation_analysis(df)
    feature_distribution(df)
    target_analysis(df)
    
    print("Analysis complete. Check generated PNG files.")