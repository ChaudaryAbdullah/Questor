import pandas as pd
import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------
file_path = "output/financial_data_enhanced.xlsx"
df = pd.read_excel(file_path)

# ----------------------------------------------------
# 2. Benford's Law Functions
# ----------------------------------------------------
def benford_first_digit(series):
    """Extract first nonzero digit from each value."""
    series = series.dropna().astype(str)
    digits = series[series.str.contains(r'\d')].str.replace(r"[^0-9]", "", regex=True)
    digits = digits[digits != ""].str.lstrip("0").str[:1]
    digits = digits[digits != ""]
    return digits.astype(float) if not digits.empty else pd.Series([])

def benford_test(series):
    """Perform chi-squared test comparing first-digit frequencies to Benford's Law."""
    digits = benford_first_digit(series)
    if digits.empty:
        return np.nan

    observed_counts = digits.value_counts().sort_index()
    total = observed_counts.sum()
    if total == 0:
        return np.nan
    observed = observed_counts / total

    benford_probs = np.log10(1 + 1 / np.arange(1, 10))
    expected = pd.Series(benford_probs, index=np.arange(1, 10))

    observed = observed.reindex(expected.index, fill_value=0)

    chi_stat, p_value = chisquare(f_obs=observed * total, f_exp=expected * total)
    return chi_stat

# ----------------------------------------------------
# 3. Identify Numerical Columns (Excluding Specified Columns)
# ----------------------------------------------------
exclude_cols = ['company_name', 'year', 'is_fraudulent', 'cik']
numeric_cols = [col for col in df.columns 
                if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

# Ensure columns are numeric, convert invalid values to NaN
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ----------------------------------------------------
# 4. Apply Benford's Law to Numerical Columns
# ----------------------------------------------------
for col in numeric_cols:
    new_col = f"benford_chi_{col}"
    print(f"Applying Benford's Law to {col}")
    df[new_col] = benford_test(df[col])

# ----------------------------------------------------
# 5. Optional Visualization for Each Column
# ----------------------------------------------------
def plot_benford(series, title="Benford's Law Test"):
    digits = benford_first_digit(series)
    if digits.empty:
        print(f"No valid digits for {title}")
        return
    
    observed_counts = digits.value_counts(normalize=True).sort_index()
    benford_probs = np.log10(1 + 1 / np.arange(1, 10))

    plt.figure(figsize=(6, 4))
    plt.bar(observed_counts.index, observed_counts.values, color="skyblue", label="Observed")
    plt.plot(np.arange(1, 10), benford_probs, "r--", label="Benford Expected")
    plt.xlabel("Leading Digit")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/benford_plot_{title.replace(' ', '_').replace(':', '')}.png")
    plt.close()

for col in numeric_cols:
    plot_benford(df[col], f"Benford Test: {col}")

# ----------------------------------------------------
# 6. Save Enhanced Dataset
# ----------------------------------------------------
output_path = "output/financial_data_benford.xlsx"
df.to_excel(output_path, index=False)
print(f"\n✅ Enhanced dataset with Benford's Law features saved to '{output_path}'")