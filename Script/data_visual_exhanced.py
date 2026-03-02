
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import os

# Create output directory for plots_enhanced
if not os.path.exists("plots_enhanced"):
    os.makedirs("plots_enhanced")

# -----------------------------
# 1. Load Data
# -----------------------------
def load_data(file_path):
    """Load Excel file and return DataFrame."""
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}\n")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        exit()

# -----------------------------
# 2. Data Type Overview
# -----------------------------
def data_type_overview(df):
    """Display data types and non-numeric columns."""
    print("Column Data Types Overview:\n", df.dtypes.value_counts(), "\n")
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    print("Non-numeric columns:", non_numeric_cols, "\n")

# -----------------------------
# 3. Missing Value Analysis
# -----------------------------
def missing_value_analysis(df):
    """Analyze and display missing values in columns."""
    missing = df.isnull().mean().sort_values(ascending=False)
    print("Top 10 columns with missing values:\n", missing.head(10), "\n")

# -----------------------------
# 4. Descriptive Statistics
# -----------------------------
def descriptive_statistics(df, exclude_cols):
    """Calculate and display descriptive statistics for numeric columns."""
    numeric_df = df.select_dtypes(include=np.number).drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')
    print(f"Numeric columns considered for stats: {len(numeric_df.columns)}\n")
    
    desc_stats = numeric_df.describe().T
    desc_stats["skewness"] = numeric_df.skew()
    desc_stats["kurtosis"] = numeric_df.kurtosis()
    desc_stats["missing_%"] = df[numeric_df.columns].isnull().mean()
    desc_stats["variance"] = numeric_df.var()
    
    print("Descriptive Statistics (Numeric Only, Excluding Identifiers):")
    print(desc_stats.head(10), "\n")
    return numeric_df, desc_stats

# -----------------------------
# 5. Fraud Class Distribution
# -----------------------------
def plot_fraud_distribution(df):
    """Plot and save distribution of fraud vs non-fraud cases."""
    if 'is_fraudulent' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='is_fraudulent', data=df, palette="Set2")
        plt.title("Distribution of Fraud vs Non-Fraud Cases")
        plt.xlabel("Is Fraudulent (0 = No, 1 = Yes)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("plots_enhanced/fraud_distribution.png", dpi=300)
        plt.close()
        print("Fraud distribution plot saved as 'plots_enhanced/fraud_distribution.png'\n")
    else:
        print("⚠️ Column 'is_fraudulent' not found, skipping class distribution plot.\n")

# -----------------------------
# 6. Correlation with Fraud
# -----------------------------
def correlation_analysis(df, numeric_df):
    """Calculate and plot correlations with fraud, excluding target from matrix."""
    if 'is_fraudulent' in df.columns:
        # Exclude 'is_fraudulent' from correlation matrix
        corr = numeric_df.corr(numeric_only=True)
        # Get correlations with 'is_fraudulent' separately
        corr_with_fraud = df[numeric_df.columns.tolist() + ['is_fraudulent']].corr(numeric_only=True)['is_fraudulent'].drop('is_fraudulent', errors='ignore')
        top_corr = corr_with_fraud.sort_values(ascending=False).head(15)
        print("Top 15 Features Correlated with Fraud:\n", top_corr, "\n")
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_corr.values, y=top_corr.index, hue=top_corr.index, palette="viridis", legend=False)
        plt.title("Top 15 Features Correlated with Fraud")
        plt.xlabel("Correlation Coefficient")
        plt.tight_layout()
        plt.savefig("plots_enhanced/correlation_with_fraud.png", dpi=300)
        plt.close()
        print("Correlation plot saved as 'plots_enhanced/correlation_with_fraud.png'\n")
        return top_corr
    return None

# -----------------------------
# 7. Distributions of Key Variables
# -----------------------------
def plot_distributions(numeric_df):
    """Plot and save histograms for the first 5 numeric features."""
    cols_to_plot = numeric_df.columns[:5]  # First 5 numeric features
    for col in cols_to_plot:
        plt.figure(figsize=(6, 4))
        sns.histplot(numeric_df[col], bins=40, kde=True, color="skyblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"plots_enhanced/hist_{col}.png", dpi=300)
        plt.close()
        print(f"Histogram for {col} saved as 'plots_enhanced/hist_{col}.png'")

# -----------------------------
# 8. Outlier Detection via Boxplots_enhanced
# -----------------------------
def plot_boxplots_enhanced(numeric_df):
    """Plot and save boxplots_enhanced for the first 5 numeric features."""
    cols_to_plot = numeric_df.columns[:5]  # First 5 numeric features
    for col in cols_to_plot:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=numeric_df[col], color="lightcoral")
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(f"plots_enhanced/boxplot_{col}.png", dpi=300)
        plt.close()
        print(f"Boxplot for {col} saved as 'plots_enhanced/boxplot_{col}.png'")

# -----------------------------
# 9. Pairplot for Top Correlated Features
# -----------------------------
def plot_pairplot(df, top_corr):
    """Plot and save pairplot for top 5 correlated features."""
    if 'is_fraudulent' in df.columns and top_corr is not None:
        top_features = top_corr.index[:5]  # Top 5 features (excluding target)
        sns.pairplot(df, vars=top_features, hue='is_fraudulent', diag_kind="kde", palette="husl")
        plt.suptitle("Pairplot of Top 5 Correlated Features with Fraud", y=1.02)
        plt.savefig("plots_enhanced/pairplot_top_features.png", dpi=300)
        plt.close()
        print("Pairplot saved as 'plots_enhanced/pairplot_top_features.png'\n")

# -----------------------------
# 10. Skewness & Kurtosis Summary
# -----------------------------
def skewness_kurtosis_summary(numeric_df):
    """Calculate and display skewness and kurtosis for numeric features."""
    skewness = numeric_df.apply(skew, nan_policy='omit')
    kurt = numeric_df.apply(kurtosis, nan_policy='omit')
    
    skew_summary = pd.DataFrame({
        'Skewness': skewness,
        'Kurtosis': kurt
    }).sort_values(by='Skewness', ascending=False)
    
    print("\nTop 10 Most Skewed Features:\n", skew_summary.head(10))
    print("\nTop 10 Least Skewed Features:\n", skew_summary.tail(10))
    return skew_summary

# -----------------------------
# 11. Generate Summary Report
# -----------------------------
def generate_summary_report(df, desc_stats, top_corr, skew_summary):
    """Generate and save a Markdown summary report."""
    try:
        missing_table = df.isnull().mean().sort_values(ascending=False).head(10).to_markdown()
        desc_table = desc_stats.head(10).to_markdown()
        top_corr_table = top_corr.to_markdown() if top_corr is not None else "Column `is_fraudulent` not found"
        skew_top_table = skew_summary.head(10).to_markdown()
        skew_bottom_table = skew_summary.tail(10).to_markdown()
    except ImportError:
        print("⚠️ Missing 'tabulate' package. Install it using 'pip install tabulate'.")
        # Fallback to string representation
        missing_table = str(df.isnull().mean().sort_values(ascending=False).head(10))
        desc_table = str(desc_stats.head(10))
        top_corr_table = str(top_corr) if top_corr is not None else "Column `is_fraudulent` not found"
        skew_top_table = str(skew_summary.head(10))
        skew_bottom_table = str(skew_summary.tail(10))

    report = f"""
# Financial Data Analysis Report

## 1. Data Overview
- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns
- **Non-numeric columns**: {', '.join(df.select_dtypes(exclude=np.number).columns.tolist())}

## 2. Missing Values
{missing_table}

## 3. Descriptive Statistics
{desc_table}

## 4. Fraud Distribution
{'Fraud distribution plot saved at `plots_enhanced/fraud_distribution.png`' if 'is_fraudulent' in df.columns else 'Column `is_fraudulent` not found'}

## 5. Correlation with Fraud
{top_corr_table}

## 6. Distributions and Outliers
- Histograms saved at `plots_enhanced/hist_<column>.png`
- Boxplots_enhanced saved at `plots_enhanced/boxplot_<column>.png`

## 7. Pairplot
{'Pairplot saved at `plots_enhanced/pairplot_top_features.png`' if top_corr is not None else 'Column `is_fraudulent` not found'}

## 8. Skewness and Kurtosis
**Top 10 Most Skewed Features:**
{skew_top_table}

**Top 10 Least Skewed Features:**
{skew_bottom_table}
"""
    with open("plots_enhanced/analysis_report.md", "w") as f:
        f.write(report)
    print("Summary report saved as 'plots_enhanced/analysis_report.md'\n")
    print("Analysis complete! Check the 'plots_enhanced' directory for visualizations and report.\n")

# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Execute the full data analysis pipeline."""
    file_path = "output/financial_data_enhanced.xlsx"
    exclude_cols = {'year', 'company_name', 'cik', 'is_fraudulent'}
    
    # Execute analysis steps
    df = load_data(file_path)
    data_type_overview(df)
    missing_value_analysis(df)
    numeric_df, desc_stats = descriptive_statistics(df, exclude_cols)
    plot_fraud_distribution(df)
    top_corr = correlation_analysis(df, numeric_df)
    plot_distributions(numeric_df)
    plot_boxplots_enhanced(numeric_df)
    plot_pairplot(df, top_corr)
    skew_summary = skewness_kurtosis_summary(numeric_df)
    generate_summary_report(df, desc_stats, top_corr, skew_summary)

if __name__ == "__main__":
    main()