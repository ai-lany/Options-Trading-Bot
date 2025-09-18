"""
Utility functions for processing and managing stock data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataProcessor:
    """Class for processing and converting stock data to numpy arrays."""
    
    def __init__(self):
        self.data_cache = {}
    
    def dataframe_to_numpy(self, df: pd.DataFrame, columns: List[str] = None) -> np.ndarray:
        """
        Convert pandas DataFrame to numpy array.
        
        Args:
            df: Input DataFrame with stock data
            columns: List of columns to include. If None, includes all numeric columns.
            
        Returns:
            numpy.ndarray: Converted data array
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to dataframe_to_numpy")
            return np.array([])
        
        if columns:
            # Select only specified columns that exist in the DataFrame
            available_columns = [col for col in columns if col in df.columns]
            if not available_columns:
                logger.error(f"None of the specified columns {columns} found in DataFrame")
                return np.array([])
            df_subset = df[available_columns]
        else:
            # Select only numeric columns
            df_subset = df.select_dtypes(include=[np.number])
        
        return df_subset.values
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock data by handling missing values and outliers.
        Args:
            df: Input DataFrame with stock data
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Forward fill missing values
        df_cleaned = df.ffill()
        # Backward fill any remaining missing values
        df_cleaned = df_cleaned.bfill()
        # Remove any remaining rows with NaN values
        df_cleaned = df_cleaned.dropna()
        
        logger.info(f"Data cleaned: {len(df)} -> {len(df_cleaned)} rows")
        return df_cleaned
    
    def find_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Identify outliers using a box plot method, i.e., points that are more than three standard deviations from the mean.
        Reference: https://www.tandfonline.com/doi/full/10.1080/23322039.2022.2066762#abstract
        Args:
            df: Input DataFrame with stock data
            column: Column name to check for outliers
        Returns:
            pd.DataFrame: DataFrame containing outliers
        """
        if df.empty or column not in df.columns:
            logger.warning("Empty DataFrame or invalid column provided to find_outliers")
            return pd.DataFrame()
        
        mean = df[column].mean()
        std = df[column].std()
        threshold = 3 * std
        
        outliers = df[(df[column] < (mean - threshold)) | (df[column] > (mean + threshold))]
        
        logger.info(f"Found {len(outliers)} outliers in column '{column}'")
        return outliers
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate percentage returns from price data.
        Args:
            prices: Array of price data
        Returns:
            numpy.ndarray: Array of percentage returns
        """
        if len(prices) < 2:
            return np.array([])
        
        returns = np.diff(prices) / prices[:-1] * 100
        return returns
    
    def calculate_moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate moving average of data.
        Args:
            data: Input data array
            window: Window size for moving average
        Returns:
            numpy.ndarray: Moving average array
        """
        if len(data) < window:
            logger.warning(f"Data length {len(data)} is less than window size {window}")
            return np.array([])
        
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def get_data_summary(self, data: np.ndarray) -> Dict[str, float]:
        """
        Get statistical summary of data.
        Args:
            data: Input data array
        Returns:
            dict: Dictionary containing statistical measures
        """
        if len(data) == 0:
            return {}
        
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'count': len(data)
        }
    
    def normalize_data(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize data using specified method.
        Args:
            data: Input data array
            method: Normalization method ('minmax' or 'zscore')
        Returns:
            numpy.ndarray: Normalized data array
        """
        if len(data) == 0:
            return data
        
        if method == 'minmax':
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max == data_min:
                return np.zeros_like(data)
            return (data - data_min) / (data_max - data_min)
        
        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return np.zeros_like(data)
            return (data - mean) / std
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")