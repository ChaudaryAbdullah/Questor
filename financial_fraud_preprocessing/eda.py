import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid display errors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .config import PLOT_STYLE


class FraudEDA:
    """
    Exploratory Data Analysis helper class.
    Produces plots and summary statistics saved as image files.
    """

    def __init__(self, output_dir="output"):
        plt.style.use(PLOT_STYLE)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def missing_summary(self, df: pd.DataFrame, top_n: int = 20):
        miss = pd.DataFrame({
            'missing_count': df.isna().sum(),
            'missing_percent': (df.isna().sum() / len(df)) * 100
        }).sort_values('missing_percent', ascending=False)
        return miss.head(top_n)

    def plot_missing_bar(self, df: pd.DataFrame, top_n: int = 20):
        miss = self.missing_summary(df, top_n)
        plt.figure(figsize=(10, max(4, top_n * 0.25)))
        miss['missing_percent'].sort_values().plot(kind='barh')
        plt.title(f'Top {top_n} columns by % missing values')
        plt.xlabel('% missing')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "missing_values.png")
        plt.savefig(filepath)
        plt.close()

    def plot_class_balance(self, df: pd.DataFrame, target_col: str = 'is_fraudulent'):
        counts = df[target_col].value_counts()
        plt.figure(figsize=(5, 3))
        sns.barplot(x=counts.index.astype(str), y=counts.values)
        plt.title('Class distribution')
        plt.xlabel(target_col)
        plt.ylabel('Count')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "class_balance.png")
        plt.savefig(filepath)
        plt.close()

    def show_numeric_summary(self, df: pd.DataFrame, exclude=None):
        if exclude is None:
            exclude = []
        numeric = df.select_dtypes(include=['number']).drop(
            columns=[c for c in exclude if c in df.columns], errors='ignore'
        )
        desc = numeric.describe().T
        desc['missing_percent'] = (numeric.isna().sum() / len(numeric)) * 100
        return desc

    def plot_skew_comparison(self, df: pd.DataFrame, cols: list):
        for col in cols:
            series = df[col].dropna()
            if series.empty:
                continue
            shifted = abs(series.min()) + 1 if (series <= -1).any() else 0
            transformed = np.log1p(series + shifted)

            plt.figure(figsize=(10, 3))
            plt.subplot(1, 2, 1)
            plt.hist(series, bins=40)
            plt.title(f'{col} (original)')

            plt.subplot(1, 2, 2)
            plt.hist(transformed, bins=40)
            plt.title(f'{col} (log1p)')

            plt.tight_layout()
            filepath = os.path.join(self.output_dir, f"skew_{col}.png")
            plt.savefig(filepath)
            plt.close()

    def plot_correlation_heatmap(self, df: pd.DataFrame, numeric_subset=None, vmax=0.9):
        if numeric_subset is None:
            numeric = df.select_dtypes(include=['number']).columns.tolist()
            numeric_subset = numeric[:30]
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[numeric_subset].corr(), cmap='coolwarm', center=0, vmax=vmax, vmin=-vmax)
        plt.title('Correlation heatmap (subset)')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(filepath)
        plt.close()
