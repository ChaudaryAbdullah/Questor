import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from .config import MISSING_THRESHOLD_PERCENT, KNN_NEIGHBORS, RANDOM_STATE
from .utils import get_numeric_and_categorical


class FraudDataPreprocessor:
    """
    Class that encapsulates preprocessing logic.

    Methods
    -------
    preprocess_version_a(df):
        Drops high-missing columns (> threshold), median imputes numeric, label-encodes categoricals,
        applies RobustScaler to numeric features, returns processed dataframe (features + target).

    preprocess_version_b(df):
        Keeps all columns, applies KNN imputation for numeric values, label-encodes categorical,
        applies RobustScaler, returns processed dataframe.
    """

    def __init__(self, target_col='is_fraudulent', id_cols=None, missing_threshold=MISSING_THRESHOLD_PERCENT):
        if id_cols is None:
            id_cols = ['company_name', 'cik', 'year']
        self.target_col = target_col
        self.id_cols = id_cols
        self.missing_threshold = missing_threshold
        self.label_encoders = {}
        logging.info(f'Preprocessor initialized with threshold={self.missing_threshold}')

    def _drop_high_missing(self, df: pd.DataFrame):
        miss = pd.DataFrame({
            'missing_percent': (df.isna().sum() / len(df)) * 100
        })
        cols_to_drop = miss[miss['missing_percent'] > self.missing_threshold].index.tolist()
        logging.info(f'Dropping {len(cols_to_drop)} columns with >{self.missing_threshold}% missing')
        return df.drop(columns=cols_to_drop), cols_to_drop

    def _auto_label_encode(self, df: pd.DataFrame, fit=True):
        # Detect categorical columns (object or category) excluding ids and target
        exclude = [self.target_col] + [c for c in self.id_cols if c in df.columns]
        _, categorical = get_numeric_and_categorical(df, exclude=exclude)
        # Also include object dtype columns not in numeric set
        categorical = [c for c in df.columns if df[c].dtype == 'object' and c not in exclude]

        logging.info(f'Auto-detected categorical columns to encode: {categorical}')
        for col in categorical:
            if fit:
                le = LabelEncoder()
                df[col] = df[col].fillna('___MISSING___').astype(str)
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    # fallback to fit-transform if encoder missing
                    le = LabelEncoder()
                    df[col] = df[col].fillna('___MISSING___').astype(str)
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    df[col] = df[col].fillna('___MISSING___').astype(str)
                    # handle unseen labels by mapping to -1 then re-encoding
                    mapped = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                    df[col] = mapped
        return df

    def preprocess_version_a(self, df: pd.DataFrame):
        """
        Version A: drop columns with high missingness, median impute numeric, label encode categoricals,
        apply RobustScaler to numeric features. Returns processed dataframe with target column as last column.
        """
        df_work = df.copy()
        df_work, dropped = self._drop_high_missing(df_work)

        # Separate target
        if self.target_col not in df_work.columns:
            raise ValueError(f'Target column {self.target_col} not found in dataframe')
        y = df_work[self.target_col].reset_index(drop=True)

        # Identify numeric columns to impute/scale (exclude ids and target)
        exclude = [self.target_col] + [c for c in self.id_cols if c in df_work.columns]
        numeric, _ = get_numeric_and_categorical(df_work, exclude=exclude)

        # Median impute + Robust scale
        if numeric:
            imputer = SimpleImputer(strategy='median')
            scaler = RobustScaler()
            X_num = pd.DataFrame(imputer.fit_transform(df_work[numeric]), columns=numeric)
            X_num = pd.DataFrame(scaler.fit_transform(X_num), columns=numeric)
        else:
            X_num = pd.DataFrame()

        # Encode categorical columns (auto detect)
        X_cat = df_work.drop(columns=numeric + exclude, errors='ignore')
        X_cat = self._auto_label_encode(X_cat, fit=True)

        # Reconstruct dataframe (preserve id columns if present)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
        result = pd.concat([X, y.reset_index(drop=True)], axis=1)
        logging.info('Preprocessing version A complete')
        return result, dropped

    def preprocess_version_b(self, df: pd.DataFrame):
        """
        Version B: keep all columns, KNN impute numeric, label encode categorical columns, apply RobustScaler.
        """
        df_work = df.copy()
        if self.target_col not in df_work.columns:
            raise ValueError(f'Target column {self.target_col} not found in dataframe')
        y = df_work[self.target_col].reset_index(drop=True)

        exclude = [self.target_col] + [c for c in self.id_cols if c in df_work.columns]
        numeric, _ = get_numeric_and_categorical(df_work, exclude=exclude)

        # KNN Imputer for numeric
        if numeric:
            knn = KNNImputer(n_neighbors=KNN_NEIGHBORS)
            X_num = pd.DataFrame(knn.fit_transform(df_work[numeric]), columns=numeric)
            X_num = pd.DataFrame(RobustScaler().fit_transform(X_num), columns=numeric)
        else:
            X_num = pd.DataFrame()

        # Categorical: label encode (auto-detect)
        X_cat = df_work.drop(columns=numeric + exclude, errors='ignore')
        X_cat = self._auto_label_encode(X_cat, fit=True)

        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
        result = pd.concat([X, y.reset_index(drop=True)], axis=1)
        logging.info('Preprocessing version B complete')
        return result

    def transform_new(self, df: pd.DataFrame, version='b'):
        """
        Transform a new dataframe using the fitted encoders and imputers.
        For version A, this assumes you drop the same high-missing columns as used in fit.
        """
        # NOTE: This method is a lightweight placeholder — in real pipeline you should persist imputers/scalers
        if version == 'a':
            return self.preprocess_version_a(df)[0]
        else:
            return self.preprocess_version_b(df)