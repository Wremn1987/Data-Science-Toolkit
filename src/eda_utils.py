# src/eda_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_dataframe_info(df):
    """Prints basic information about the DataFrame, including shape, column types, and missing values."""
    print("
--- DataFrame Info ---")
    print(f"Shape: {df.shape}")
    print("
Column Info:")
    df.info()
    print("
Missing Values:
", df.isnull().sum())
    print("
Duplicate Rows:
", df.duplicated().sum())

def get_descriptive_stats(df, include_categorical=False):
    """Prints descriptive statistics for numerical and optionally categorical columns."""
    print("
--- Descriptive Statistics (Numerical) ---")
    print(df.describe())
    if include_categorical:
        print("
--- Descriptive Statistics (Categorical) ---")
        print(df.describe(include='object'))

def plot_numerical_distributions(df, columns=None, bins=30):
    """
    Plots histograms and box plots for numerical columns.
    Requires matplotlib and seaborn to be installed.
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    
    for col in columns:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True, bins=bins)
        plt.title(f'Distribution of {col}' )
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[col].dropna())
        plt.title(f'Box Plot of {col}' )
        
        plt.tight_layout()
        plt.show()

def plot_categorical_distributions(df, columns=None):
    """
    Plots count plots for categorical columns.
    Requires matplotlib and seaborn to be installed.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns
    
    for col in columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Count Plot of {col}' )
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

def plot_correlation_heatmap(df, numerical_columns=None):
    """
    Plots a correlation heatmap for numerical columns.
    Requires matplotlib and seaborn to be installed.
    """
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=np.number).columns
    
    corr_matrix = df[numerical_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == '__main__':
    # Sample DataFrame
    data = {
        'Numerical_1': np.random.rand(100) * 100,
        'Numerical_2': np.random.randint(0, 50, 100),
        'Categorical_1': np.random.choice(['A', 'B', 'C'], 100),
        'Categorical_2': np.random.choice(['X', 'Y'], 100),
        'Target': np.random.rand(100) > 0.5
    }
    df = pd.DataFrame(data)
    df.loc[5, 'Numerical_1'] = np.nan # Add a missing value
    df.loc[10, 'Numerical_2'] = 150 # Add an outlier
    df.loc[15, 'Categorical_1'] = np.nan # Add a missing categorical

    get_dataframe_info(df)
    get_descriptive_stats(df, include_categorical=True)
    # plot_numerical_distributions(df, columns=['Numerical_1', 'Numerical_2']) # Commented out for non-interactive environment
    # plot_categorical_distributions(df, columns=['Categorical_1', 'Categorical_2']) # Commented out for non-interactive environment
    # plot_correlation_heatmap(df, numerical_columns=['Numerical_1', 'Numerical_2']) # Commented out for non-interactive environment
    print("Plots are commented out for non-interactive environment. Uncomment to view locally.")
