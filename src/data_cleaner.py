import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    A utility class for performing common data cleaning operations on Pandas DataFrames.
    Includes methods for handling missing values, removing duplicates, and type conversion.
    """
    def __init__(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")
        self.df = dataframe.copy()
        logger.info("DataCleaner initialized with a DataFrame.")

    def handle_missing_values(self, strategy=\'mean\', columns=None, fill_value=None):
        """
        Handles missing values (NaN) in the DataFrame.
        
        Args:
            strategy (str): How to fill missing values. Options: \'mean\', \'median\', \'mode\', \'ffill\', \'bfill\', \'drop\', \'constant\'.
            columns (list): List of columns to apply the strategy. If None, applies to all suitable columns.
            fill_value: Value to use if strategy is \'constant\'.
        """
        target_columns = columns if columns is not None else self.df.columns
        logger.info(f"Handling missing values with strategy: {strategy} for columns: {target_columns}")

        for col in target_columns:
            if self.df[col].isnull().any():
                if strategy == \'mean\':
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    else:
                        logger.warning(f"Cannot apply \'mean\' strategy to non-numeric column: {col}")
                elif strategy == \'median\':
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        logger.warning(f"Cannot apply \'median\' strategy to non-numeric column: {col}")
                elif strategy == \'mode\':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == \'ffill\':
                    self.df[col].fillna(method=\'ffill\', inplace=True)
                elif strategy == \'bfill\':
                    self.df[col].fillna(method=\'bfill\', inplace=True)
                elif strategy == \'drop\':
                    self.df.dropna(subset=[col], inplace=True)
                elif strategy == \'constant\':
                    if fill_value is not None:
                        self.df[col].fillna(fill_value, inplace=True)
                    else:
                        logger.warning(f"\'constant\' strategy requires a fill_value for column: {col}")
                else:
                    logger.warning(f"Unknown strategy: {strategy} for column: {col}")
            else:
                logger.info(f"No missing values in column: {col}")
        return self.df

    def remove_duplicates(self, subset=None, keep=\'first\'):
        """
        Removes duplicate rows from the DataFrame.
        
        Args:
            subset (list): List of column names to consider for identifying duplicates.
            keep (str): Which duplicates to mark as False. Options: \'first\', \'last\', False.
        """
        initial_rows = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        removed_rows = initial_rows - len(self.df)
        logger.info(f"Removed {removed_rows} duplicate rows.")
        return self.df

    def convert_column_type(self, column, new_type):
        """
        Converts a specified column to a new data type.
        
        Args:
            column (str): The name of the column to convert.
            new_type (type): The target data type (e.g., int, float, str, datetime).
        """
        if column not in self.df.columns:
            logger.error(f"Column \'{column}\' not found in DataFrame.")
            return self.df
        
        try:
            if new_type == \'datetime\':
                self.df[column] = pd.to_datetime(self.df[column])
            else:
                self.df[column] = self.df[column].astype(new_type)
            logger.info(f"Column \'{column}\' converted to type {new_type.__name__}.")
        except Exception as e:
            logger.error(f"Failed to convert column \'{column}\' to {new_type.__name__}: {e}")
        return self.df

    def get_dataframe(self):
        """
        Returns the cleaned DataFrame.
        """
        return self.df

if __name__ == "__main__":
    logger.info("Running example for DataCleaner...")
    
    # Create a sample DataFrame with missing values and duplicates
    data = {
        \'A\': [1, 2, np.nan, 4, 2, 5],
        \'B\': [\'foo\', \'bar\', \'foo\', \'baz\', \'bar\', np.nan],
        \'C\': [10.1, 20.2, 10.1, 40.4, 20.2, 50.5],
        \'D\': [\'2023-01-01\', \'2023-01-02\', \'2023-01-01\', \'2023-01-04\', \'2023-01-02\', \'2023-01-05\']
    }
    sample_df = pd.DataFrame(data)
    logger.info("Original DataFrame:\n" + str(sample_df))

    cleaner = DataCleaner(sample_df)
    
    # Handle missing values
    cleaner.handle_missing_values(strategy=\'mean\', columns=[\'A\'])
    cleaner.handle_missing_values(strategy=\'mode\', columns=[\'B\'])
    cleaner.handle_missing_values(strategy=\'constant\', columns=[\'C\'], fill_value=0.0)
    logger.info("\nDataFrame after handling missing values:\n" + str(cleaner.get_dataframe()))

    # Remove duplicates
    cleaner.remove_duplicates(subset=[\'A\', \'B\'], keep=\'first\')
    logger.info("\nDataFrame after removing duplicates:\n" + str(cleaner.get_dataframe()))

    # Convert column type
    cleaner.convert_column_type(\'A\', int)
    cleaner.convert_column_type(\'D\', \'datetime\')
    logger.info("\nDataFrame after type conversion:\n" + str(cleaner.get_dataframe()))
    logger.info("DataCleaner example completed.")
