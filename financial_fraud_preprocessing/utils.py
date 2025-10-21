import logging
import pandas as pd


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s - %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_numeric_and_categorical(df: pd.DataFrame, exclude=None):
    if exclude is None:
        exclude = []
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # remove excluded columns
    numeric = [c for c in numeric if c not in exclude]
    categorical = [c for c in categorical if c not in exclude]
    return numeric, categorical
