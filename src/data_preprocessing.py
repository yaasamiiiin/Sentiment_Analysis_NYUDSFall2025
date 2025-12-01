"""
Data Preprocessing Module

This module contains functions for cleaning and preprocessing sentiment data.
"""

import pandas as pd
from typing import List, Dict, Optional


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with duplicates removed
    """
    initial_shape = df.shape[0]
    df_clean = df.drop_duplicates()
    duplicates_removed = initial_shape - df_clean.shape[0]
    
    print(f"Found and removed {duplicates_removed} duplicate rows.")
    print(f"New shape: {df_clean.shape}")
    
    return df_clean


def clean_categorical_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean categorical columns by stripping whitespace.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list of str, optional
        List of column names to clean. If None, defaults to 
        ['Platform', 'Sentiment', 'Country']
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with cleaned categorical columns
    """
    if columns is None:
        columns = ['Platform', 'Sentiment', 'Country']
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.strip()
            print(f"Cleaned '{col}' column - unique values: {df_clean[col].nunique()}")
        else:
            print(f"Warning: Column '{col}' not found in dataframe")
    
    return df_clean


def validate_data(df: pd.DataFrame) -> Dict:
    """
    Perform data quality assessment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary containing validation results
    """
    validation_results = {}
    
    # Check for missing values
    missing_values = df.isnull().sum()
    validation_results['missing_values'] = missing_values
    print("\n=== Missing Values ===")
    print(missing_values)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    validation_results['duplicates'] = duplicates
    print(f"\n=== Duplicates ===")
    print(f"Found {duplicates} duplicate rows.")
    
    # Validate timestamp column
    if 'Timestamp' in df.columns:
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            validation_results['timestamp_valid'] = True
            print("\n=== Timestamp Validation ===")
            print("Timestamp column successfully converted to datetime.")
        except Exception as e:
            validation_results['timestamp_valid'] = False
            print(f"\nError converting Timestamp: {str(e)}")
    
    # Check text column
    if 'Text' in df.columns:
        empty_posts = df[df['Text'].str.len() == 0].shape[0]
        validation_results['empty_posts'] = empty_posts
        print(f"\n=== Text Validation ===")
        print(f"Found {empty_posts} posts with no text.")
    
    # Basic statistics
    # print("\n=== Descriptive Statistics ===")
    # print(df.describe())
    
    return validation_results


def map_sentiments(df: pd.DataFrame, sentiment_col: str = 'Sentiment', map_type: str = 'full') -> pd.DataFrame:
    """
    Map detailed sentiments to broader categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    sentiment_col : str, default='Sentiment'
        Name of the sentiment column to map
    map_type : str, default='full'
        'full'  -> use sentiment_map (Joy / Sadness / Anger / Fear / Guilt / Neutral/Other)
        '3cat'  -> use sentiment_map_3 (Positive / Negative / Neutral)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with new 'Sentiment_Group' column
    """
    sentiment_map = {
        # == Joy ==
        'positive': 'Joy', 'happiness': 'Joy', 'joy': 'Joy', 'love': 'Joy', 'amusement': 'Joy', 
        'enjoyment': 'Joy', 'admiration': 'Joy', 'affection': 'Joy', 'awe': 'Joy', 'adoration': 'Joy', 
        'excitement': 'Joy', 'kind': 'Joy', 'pride': 'Joy', 'elation': 'Joy', 'euphoria': 'Joy', 
        'contentment': 'Joy', 'serenity': 'Joy', 'gratitude': 'Joy', 'hope': 'Joy', 'empowerment': 'Joy', 
        'compassion': 'Joy', 'tenderness': 'Joy', 'arousal': 'Joy', 'enthusiasm': 'Joy', 'fulfillment': 'Joy', 
        'reverence': 'Joy', 'hopeful': 'Joy', 'proud': 'Joy', 'grateful': 'Joy', 'empathetic': 'Joy', 
        'compassionate': 'Joy', 'playful': 'Joy', 'free-spirited': 'Joy', 'inspired': 'Joy', 'confident': 'Joy', 
        'thrill': 'Joy', 'overjoyed': 'Joy', 'inspiration': 'Joy', 'motivation': 'Joy', 'satisfaction': 'Joy', 
        'blessed': 'Joy', 'appreciation': 'Joy', 'confidence': 'Joy', 'accomplishment': 'Joy', 'wonderment': 'Joy', 
        'optimism': 'Joy', 'enchantment': 'Joy', 'playfuljoy': 'Joy', 'dreamchaser': 'Joy', 'elegance': 'Joy', 
        'whimsy': 'Joy', 'harmony': 'Joy', 'creativity': 'Joy', 'radiance': 'Joy', 'wonder': 'Joy', 
        'rejuvenation': 'Joy', 'coziness': 'Joy', 'adventure': 'Joy', 'melodic': 'Joy', 'festivejoy': 'Joy', 
        'freedom': 'Joy', 'dazzle': 'Joy', 'adrenaline': 'Joy', 'artisticburst': 'Joy', 'culinaryodyssey': 'Joy', 
        'resilience': 'Joy', 'spark': 'Joy', 'marvel': 'Joy', 'positivity': 'Joy', 'kindness': 'Joy', 
        'friendship': 'Joy', 'success': 'Joy', 'exploration': 'Joy', 'amazement': 'Joy', 'romance': 'Joy', 
        'captivation': 'Joy', 'tranquility': 'Joy', 'grandeur': 'Joy', 'energy': 'Joy', 'celebration': 'Joy', 
        'charm': 'Joy', 'ecstasy': 'Joy', 'colorful': 'Joy', 'hypnotic': 'Joy', 'connection': 'Joy', 
        'iconic': 'Joy', 'engagement': 'Joy', 'touched': 'Joy', 'triumph': 'Joy', 'heartwarming': 'Joy', 
        'breakthrough': 'Joy', 'joy in baking': 'Joy', 'imagination': 'Joy', 'vibrancy': 'Joy', 'mesmerizing': 'Joy', 
        'culinary adventure': 'Joy', 'winter magic': 'Joy', 'thrilling journey': 'Joy', "nature's beauty": 'Joy', 
        'celestial wonder': 'Joy', 'creative inspiration': 'Joy', 'runway creativity': 'Joy', "ocean's freedom": 'Joy', 
        'relief': 'Joy', 'mischievous': 'Joy', 'happy': 'Joy', 'joyfulreunion': 'Joy', 'solace': 'Joy', 
        'envisioning history': 'Joy',

        # == Sadness ==
        'sadness': 'Sadness', 'disappointed': 'Sadness', 'despair': 'Sadness', 'grief': 'Sadness', 'loneliness': 'Sadness', 
        'melancholy': 'Sadness', 'yearning': 'Sadness', 'devastated': 'Sadness', 'heartbreak': 'Sadness', 'betrayal': 'Sadness', 
        'suffering': 'Sadness', 'emotionalstorm': 'Sadness', 'isolation': 'Sadness', 'disappointment': 'Sadness', 
        'lostlove': 'Sadness', 'exhaustion': 'Sadness', 'sorrow': 'Sadness', 'darkness': 'Sadness', 'desperation': 'Sadness', 
        'ruins': 'Sadness', 'desolation': 'Sadness', 'loss': 'Sadness', 'heartache': 'Sadness', 'solitude': 'Sadness', 
        'sympathy': 'Sadness', 'sad': 'Sadness', 'bittersweet': 'Sadness',

        # == Anger ==
        'negative': 'Anger', 'anger': 'Anger', 'disgust': 'Anger', 'bitter': 'Anger', 'resentment': 'Anger', 
        'frustration': 'Anger', 'jealousy': 'Anger', 'envy': 'Anger', 'bitterness': 'Anger', 'jealous': 'Anger', 
        'frustrated': 'Anger', 'envious': 'Anger', 'dismissive': 'Anger', 'hate': 'Anger', 'bad': 'Anger', 
        'mean-spirited': 'Anger',

        # == Fear ==
        'fear': 'Fear', 'boredom': 'Fear', 'anxiety': 'Fear', 'intimidation': 'Fear', 'helplessness': 'Fear', 
        'fearful': 'Fear', 'apprehensive': 'Fear', 'overwhelmed': 'Fear', 'suspense': 'Fear', 'pressure': 'Fear', 
        'obstacle': 'Fear', 'challenge': 'Fear',

        # == Guilt ==
        'shame': 'Guilt', 'regret': 'Guilt', 'embarrassed': 'Guilt', 'miscalculation': 'Guilt',

        # == Neutral/Other ==
        'neutral': 'Neutral/Other', 'surprise': 'Neutral/Other', 'acceptance': 'Neutral/Other', 
        'anticipation': 'Neutral/Other', 'calmness': 'Neutral/Other', 'confusion': 'Neutral/Other', 
        'curiosity': 'Neutral/Other', 'indifference': 'Neutral/Other', 'numbness': 'Neutral/Other', 
        'nostalgia': 'Neutral/Other', 'ambivalence': 'Neutral/Other', 'determination': 'Neutral/Other', 
        'contemplation': 'Neutral/Other', 'reflection': 'Neutral/Other', 'mindfulness': 'Neutral/Other', 
        'pensive': 'Neutral/Other', 'innerjourney': 'Neutral/Other', 'immersion': 'Neutral/Other', 'emotion': 'Neutral/Other', 
        'journey': 'Neutral/Other', 'renewed effort': 'Neutral/Other', 'whispers of the past': 'Neutral/Other', 
        'intrigue': 'Neutral/Other'
    }

    sentiment_map_3 = {
        # Positive
        'positive': 'Positive', 'happiness': 'Positive', 'joy': 'Positive', 'love': 'Positive', 'amusement': 'Positive', 
        'enjoyment': 'Positive', 'admiration': 'Positive', 'affection': 'Positive', 'awe': 'Positive', 'acceptance': 'Positive', 
        'adoration': 'Positive', 'excitement': 'Positive', 'kind': 'Positive', 'pride': 'Positive', 'elation': 'Positive', 
        'euphoria': 'Positive', 'contentment': 'Positive', 'serenity': 'Positive', 'gratitude': 'Positive', 'hope': 'Positive', 
        'empowerment': 'Positive', 'compassion': 'Positive', 'tenderness': 'Positive', 'arousal': 'Positive', 
        'enthusiasm': 'Positive', 'fulfillment': 'Positive', 'reverence': 'Positive', 'determination': 'Positive', 'zest': 'Positive', 
        'hopeful': 'Positive', 'proud': 'Positive', 'grateful': 'Positive', 'empathetic': 'Positive', 'compassionate': 'Positive', 
        'playful': 'Positive', 'free-spirited': 'Positive', 'inspired': 'Positive', 'confident': 'Positive', 'thrill': 'Positive', 
        'overjoyed': 'Positive', 'inspiration': 'Positive', 'motivation': 'Positive', 'satisfaction': 'Positive', 'blessed': 'Positive', 
        'appreciation': 'Positive', 'confidence': 'Positive', 'accomplishment': 'Positive', 'wonderment': 'Positive', 
        'optimism': 'Positive', 'enchantment': 'Positive', 'intrigue': 'Positive', 'playfuljoy': 'Positive', 'dreamchaser': 'Positive', 
        'elegance': 'Positive', 'whimsy': 'Positive', 'harmony': 'Positive', 'creativity': 'Positive', 'radiance': 'Positive', 
        'wonder': 'Positive', 'rejuvenation': 'Positive', 'coziness': 'Positive', 'adventure': 'Positive', 'melodic': 'Positive', 
        'festivejoy': 'Positive', 'freedom': 'Positive', 'dazzle': 'Positive', 'adrenaline': 'Positive', 'artisticburst': 'Positive', 
        'culinaryodyssey': 'Positive', 'resilience': 'Positive', 'spark': 'Positive', 'marvel': 'Positive', 'positivity': 'Positive', 
        'kindness': 'Positive', 'friendship': 'Positive', 'success': 'Positive', 'exploration': 'Positive', 'amazement': 'Positive', 
        'romance': 'Positive', 'captivation': 'Positive', 'tranquility': 'Positive', 'grandeur': 'Positive', 'energy': 'Positive', 
        'celebration': 'Positive', 'charm': 'Positive', 'ecstasy': 'Positive', 'colorful': 'Positive', 'hypnotic': 'Positive', 
        'connection': 'Positive', 'iconic': 'Positive', 'journey': 'Positive', 'engagement': 'Positive', 'touched': 'Positive', 
        'triumph': 'Positive', 'heartwarming': 'Positive', 'breakthrough': 'Positive', 'joy in baking': 'Positive', 
        'envisioning history': 'Positive', 'imagination': 'Positive', 'vibrancy': 'Positive', 'mesmerizing': 'Positive', 
        'culinary adventure': 'Positive', 'winter magic': 'Positive', 'thrilling journey': 'Positive', "nature's beauty": 'Positive', 
        'celestial wonder': 'Positive', 'creative inspiration': 'Positive', 'runway creativity': 'Positive', "ocean's freedom": 'Positive', 
        'relief': 'Positive', 'mischievous': 'Positive', 'happy': 'Positive', 'joyfulreunion': 'Positive', 'solace': 'Positive',

        # Negative
        'negative': 'Negative', 'anger': 'Negative', 'fear': 'Negative', 'sadness': 'Negative', 'disgust': 'Negative', 
        'disappointed': 'Negative', 'bitter': 'Negative', 'shame': 'Negative', 'despair': 'Negative', 'grief': 'Negative', 
        'loneliness': 'Negative', 'jealousy': 'Negative', 'resentment': 'Negative', 'frustration': 'Negative', 'boredom': 'Negative', 
        'anxiety': 'Negative', 'intimidation': 'Negative', 'helplessness': 'Negative', 'envy': 'Negative', 'regret': 'Negative', 
        'bitterness': 'Negative', 'yearning': 'Negative', 'fearful': 'Negative', 'apprehensive': 'Negative', 'overwhelmed': 'Negative', 
        'jealous': 'Negative', 'devastated': 'Negative', 'frustrated': 'Negative', 'envious': 'Negative', 'dismissive': 'Negative', 
        'heartbreak': 'Negative', 'betrayal': 'Negative', 'suffering': 'Negative', 'emotionalstorm': 'Negative', 'isolation': 'Negative', 
        'disappointment': 'Negative', 'lostlove': 'Negative', 'exhaustion': 'Negative', 'sorrow': 'Negative', 'darkness': 'Negative', 
        'desperation': 'Negative', 'ruins': 'Negative', 'desolation': 'Negative', 'loss': 'Negative', 'heartache': 'Negative', 
        'solitude': 'Negative', 'suspense': 'Negative', 'obstacle': 'Negative', 'sympathy': 'Negative', 'pressure': 'Negative', 
        'renewed effort': 'Negative', 'miscalculation': 'Negative', 'challenge': 'Negative', 'embarrassed': 'Negative', 'sad': 'Negative', 
        'hate': 'Negative', 'bad': 'Negative',

        # Neutral
        'neutral': 'Neutral', 'surprise': 'Neutral', 'anticipation': 'Neutral', 'calmness': 'Neutral', 'confusion': 'Neutral', 
        'curiosity': 'Neutral', 'indifference': 'Neutral', 'numbness': 'Neutral', 'melancholy': 'Neutral', 'nostalgia': 'Neutral', 
        'ambivalence': 'Neutral', 'bittersweet': 'Neutral', 'contemplation': 'Neutral', 'reflection': 'Neutral', 'mindfulness': 'Neutral', 
        'pensive': 'Neutral', 'innerjourney': 'Neutral', 'immersion': 'Neutral', 'emotion': 'Neutral', 
        'whispers of the past': 'Neutral'
    }
    
    df_mapped = df.copy()
    
    # Standardize to lowercase
    df_mapped['Sentiment_Clean'] = df_mapped[sentiment_col].str.lower()

    # Choose which map to use
    mapping = sentiment_map_3 if map_type == '3cat' else sentiment_map
    
    # Map to groups
    df_mapped['Sentiment_Group'] = df_mapped['Sentiment_Clean'].map(mapping)
    
    # Fill unmapped values
    df_mapped['Sentiment_Group'] = df_mapped['Sentiment_Group'].fillna('Neutral/Other')
    
    print("\n=== Sentiment Mapping Results ===")
    print(df_mapped['Sentiment_Group'].value_counts())
    
    return df_mapped


def add_time_features(df: pd.DataFrame, timestamp_col: str = 'Timestamp') -> pd.DataFrame:
    """
    Extract time-based features from timestamp column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    timestamp_col : str, default='Timestamp'
        Name of the timestamp column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional time feature columns
    """
    df_time = df.copy()
    
    # Ensure timestamp is datetime
    df_time[timestamp_col] = pd.to_datetime(df_time[timestamp_col])
    
    # Extract features
    df_time['year'] = df_time[timestamp_col].dt.year
    df_time['month'] = df_time[timestamp_col].dt.month
    df_time['day'] = df_time[timestamp_col].dt.day
    df_time['hour'] = df_time[timestamp_col].dt.hour
    df_time['day_of_week'] = df_time[timestamp_col].dt.dayofweek
    df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
    
    print("\n=== Time Features Added ===")
    print(f"Added columns: year, month, day, hour, day_of_week, is_weekend")
    
    return df_time


def clean_data(df: pd.DataFrame, 
               remove_dups: bool = True,
               clean_categorical: bool = True,
               validate: bool = True,
               map_sentiment: bool = True,
               add_time: bool = False) -> pd.DataFrame:
    """
    Complete data cleaning pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    remove_dups : bool, default=True
        Whether to remove duplicate rows
    clean_categorical : bool, default=True
        Whether to clean categorical columns
    validate : bool, default=True
        Whether to perform data validation
    map_sentiment : bool, default=True
        Whether to map sentiments to broader groups
    add_time : bool, default=False
        Whether to add time-based features
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and processed dataframe
    """
    print("Starting data cleaning pipeline...")
    df_clean = df.copy()
    
    if clean_categorical:
        df_clean = clean_categorical_columns(df_clean)

    if remove_dups:
        df_clean = remove_duplicates(df_clean)
    
    if validate:
        validate_data(df_clean)
    
    if map_sentiment and 'Sentiment' in df_clean.columns:
        df_clean = map_sentiments(df_clean)
    
    if add_time and 'Timestamp' in df_clean.columns:
        df_clean = add_time_features(df_clean)
    
    print(f"\n=== Cleaning Complete ===")
    print(f"Final shape: {df_clean.shape}")
    
    return df_clean