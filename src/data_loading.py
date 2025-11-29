"""
Data Loading Module

This module handles loading sentiment analysis datasets from CSV files.
"""

import pandas as pd
from typing import Optional


def load_data(filepath: str, drop_unnamed: bool = True) -> pd.DataFrame:
    """
    Load sentiment dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the sentiment data
    drop_unnamed : bool, default=True
        Whether to drop unnamed index columns that may have been saved
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with sentiment data
        
    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist
    pd.errors.EmptyDataError
        If the file is empty
        
    Example:
    --------
    >>> df = load_data('../Data/sentimentdataset.csv')
    >>> print(df.shape)
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        
        if drop_unnamed:
            # Drop unnamed columns (typically index columns from previous saves)
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
                print(f"Dropped {len(unnamed_cols)} unnamed columns: {unnamed_cols}")
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Empty CSV file: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def get_data_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to get information about
    """
    print("\n=== Dataset Information ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst few rows:\n{df.head()}")