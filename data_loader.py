"""
Data Loading Functions
"""

import numpy as np
import pandas as pd
import warnings


def load_amazon_csv(filepath):
    """Load Amazon ratings-only dataset (no headers)"""
    print(f"Loading {filepath.split('\\')[-1]}...")

    df = pd.read_csv(filepath, header=None, names=['product_id', 'user_id', 'rating', 'timestamp'], low_memory=False)

    print(f"   {len(df):,} rows read")

    df = df[['user_id', 'product_id', 'rating']]
    df = df.dropna()

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])

    df = df[df['rating'] > 0]

    print(f"   {len(df):,} valid interactions loaded")
    print(f"   Rating range: [{df['rating'].min():.1f}, {df['rating'].max():.1f}]")

    return df


def load_book_crossing_csv(ratings_path, books_path=None):
    """Load Book-Crossing dataset"""
    print(f"Loading {ratings_path.split('\\\\')[-1]}...")

    # Try different delimiters
    try:
        df = pd.read_csv(ratings_path, sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    except:
        try:
            df = pd.read_csv(ratings_path, sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False)
        except:
            df = pd.read_csv(ratings_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)

    print(f"   {len(df):,} rows read")

    # Detect columns
    df.columns = df.columns.str.strip()
    cols = df.columns.tolist()
    print(f"   Columns: {cols}")

    # Automatic detection
    user_col = None
    item_col = None
    rating_col = None

    for col in cols:
        col_lower = col.lower()
        if 'user' in col_lower and user_col is None:
            user_col = col
        elif ('isbn' in col_lower or 'book' in col_lower or 'item' in col_lower) and item_col is None:
            item_col = col
        elif 'rating' in col_lower and rating_col is None:
            rating_col = col

    # Fallback to column index
    if user_col is None:
        user_col = cols[0]
    if item_col is None:
        item_col = cols[1] if len(cols) > 1 else cols[0]
    if rating_col is None:
        rating_col = cols[2] if len(cols) > 2 else cols[1]

    print(f"   Identified: user='{user_col}', item='{item_col}', rating='{rating_col}'")

    # Select columns
    df = df[[user_col, item_col, rating_col]].copy()
    df.columns = ['user_id', 'product_id', 'rating']

    # Drop missing values
    df = df.dropna()

    # Convert rating to numeric
    print("   Converting rating to numeric...")
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])

    # Remove zero ratings
    df = df[df['rating'] > 0]

    print(f"   {len(df):,} valid interactions loaded")
    print(f"   Rating range: [{df['rating'].min():.1f}, {df['rating'].max():.1f}]")

    return df


def filter_k_core(df, k=5):
    """Apply k-core filter"""
    print(f"Filtering {k}-core...")
    initial = len(df)

    iteration = 0
    while True:
        iteration += 1
        user_counts = df['user_id'].value_counts()
        item_counts = df['product_id'].value_counts()

        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index

        df_filtered = df[df['user_id'].isin(valid_users) & df['product_id'].isin(valid_items)]

        if len(df_filtered) == len(df) or iteration > 100:
            break
        df = df_filtered

    print(f"   {len(df):,} interactions remaining ({100 * (initial - len(df)) / initial:.1f}% removed)")
    print(f"   Users: {df['user_id'].nunique():,}, Items: {df['product_id'].nunique():,}")
    return df


def create_user_item_mappings(df, user_col='user_id', item_col='product_id'):
    """Create mapping from ID to index"""
    unique_users = df[user_col].unique()
    unique_items = df[item_col].unique()

    user_mapping = {user: idx for idx, user in enumerate(unique_users)}
    item_mapping = {item: idx for idx, item in enumerate(unique_items)}

    return user_mapping, item_mapping


def preprocess_data(df, user_mapping, item_mapping,
                    user_col='user_id', item_col='product_id', rating_col='rating'):
    """Convert DataFrame to a numpy array for training"""
    df_copy = df.copy()
    df_copy['user_idx'] = df_copy[user_col].map(user_mapping)
    df_copy['item_idx'] = df_copy[item_col].map(item_mapping)

    # Drop rows without mapping
    df_copy = df_copy.dropna(subset=['user_idx', 'item_idx'])

    # Convert to numpy array
    data = df_copy[['user_idx', 'item_idx', rating_col]].values

    # Ensure proper data types
    data[:, 0] = data[:, 0].astype(np.int64)
    data[:, 1] = data[:, 1].astype(np.int64)
    data[:, 2] = data[:, 2].astype(np.float32)

    return data


