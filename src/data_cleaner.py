# src/data_cleaner.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handles missing values in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): Imputation strategy ('mean', 'median', 'most_frequent', 'constant').
        columns (list, optional): List of columns to apply imputation. If None, applies to all numeric columns for 'mean'/'median', or all categorical for 'most_frequent'.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df_copy = df.copy()
    if columns is None:
        if strategy in ['mean', 'median']:
            columns = df_copy.select_dtypes(include=np.number).columns
        elif strategy == 'most_frequent':
            columns = df_copy.select_dtypes(include='object').columns
        else:
            columns = df_copy.columns # For 'constant' strategy

    for col in columns:
        if df_copy[col].isnull().any():
            if strategy == 'constant':
                fill_value = 0 if df_copy[col].dtype in [np.number] else 'missing'
                imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            else:
                imputer = SimpleImputer(strategy=strategy)
            df_copy[col] = imputer.fit_transform(df_copy[[col]])
    return df_copy

def remove_outliers_iqr(df, column, k=1.5):
    """
    Removes outliers from a specified column using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to process.
        k (float): Multiplier for the IQR to define outlier bounds.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def scale_features(df, columns, scaler_type='standard'):
    """
    Scales numerical features using StandardScaler or MinMaxScaler.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numerical columns to scale.
        scaler_type (str): Type of scaler ('standard' or 'minmax').

    Returns:
        pd.DataFrame: DataFrame with scaled features.
    """
    df_copy = df.copy()
    if scaler_type == 'standard':
        scaler = StandardScaler()
    # elif scaler_type == 'minmax':
    #     scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy

def encode_categorical_features(df, columns, encoder_type='onehot'):
    """
    Encodes categorical features using OneHotEncoder.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of categorical columns to encode.
        encoder_type (str): Type of encoder ('onehot').

    Returns:
        pd.DataFrame: DataFrame with encoded features.
    """
    df_copy = df.copy()
    if encoder_type == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = encoder.fit_transform(df_copy[columns])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(columns))
        df_copy = pd.concat([df_copy.drop(columns=columns), encoded_df], axis=1)
    else:
        raise ValueError("encoder_type must be 'onehot'")
    return df_copy

if __name__ == '__main__':
    # Sample DataFrame
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, 100, 50],
        'C': ['X', 'Y', 'X', 'Z', np.nan],
        'D': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:
", df)

    # Handle missing values
    df_cleaned = handle_missing_values(df, strategy='mean', columns=['A'])
    df_cleaned = handle_missing_values(df_cleaned, strategy='most_frequent', columns=['C'])
    print("
DataFrame after handling missing values:
", df_cleaned)

    # Remove outliers
    df_no_outliers = remove_outliers_iqr(df_cleaned, 'B', k=1.5)
    print("
DataFrame after removing outliers from B:
", df_no_outliers)

    # Scale features
    df_scaled = scale_features(df_no_outliers, columns=['A', 'B'], scaler_type='standard')
    print("
DataFrame after scaling features A and B:
", df_scaled)

    # Encode categorical features
    df_encoded = encode_categorical_features(df_scaled, columns=['C'], encoder_type='onehot')
    print("
DataFrame after one-hot encoding C:
", df_encoded)
