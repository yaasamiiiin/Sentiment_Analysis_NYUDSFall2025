"""
Sentiment Analysis Data Processing Package

This package provides utilities for loading, preprocessing, and analyzing
social media sentiment data.

Usage:
    from src import load_data, clean_data
    
    df = load_data('../Data/sentimentdataset.csv')
    df_clean = clean_data(df)
"""

from .data_loading import load_data, get_data_info
from .data_preprocessing import (
    clean_data,
    remove_duplicates,
    clean_categorical_columns,
    validate_data,
    map_sentiments,
    add_time_features
)

__version__ = '1.0.0'
__author__ = 'Yasamin'

__all__ = [
    'load_data',
    'get_data_info',
    'clean_data',
    'remove_duplicates',
    'clean_categorical_columns',
    'validate_data',
    'map_sentiments',
    'add_time_features'
]