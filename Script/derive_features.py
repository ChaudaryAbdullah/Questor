import pandas as pd
import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Load your cleaned dataset
# ----------------------------------------------------
file_path = "output/financial_data.xlsx"
df = pd.read_excel(file_path)

# Ensure required columns exist before computation
def safe_div(numer, denom):
    """Safe divide to avoid division by zero."""
    return np.where((denom == 0) | (denom.isna()), np.nan, numer / denom)

# ----------------------------------------------------
# 2. PROFITABILITY RATIOS
# ----------------------------------------------------
df["gross_profit_margin"] = safe_div(
    df["income_statement_total_revenues"] - df["income_statement_total_operating_expenses"],
    df["income_statement_total_revenues"]
)

df["return_on_assets"] = safe_div(
    df["income_statement_net_income_loss"],
    df["balance_sheet_total_assets"]
)

df["return_on_equity"] = safe_div(
    df["income_statement_net_income_loss"],
    df["balance_sheet_total_shareholders_equity"]
)

df["operating_margin"] = safe_div(
    df["income_statement_profit_loss_operations"],
    df["income_statement_total_revenues"]
)

# ----------------------------------------------------
# 3. LIQUIDITY RATIOS
# ----------------------------------------------------
df["current_ratio"] = safe_div(
    df["balance_sheet_total_current_assets"],
    df["balance_sheet_accounts_payable_accrued"]
)

df["quick_ratio"] = safe_div(
    (df["balance_sheet_cash"] + df["balance_sheet_accounts_receivable"]),
    df["balance_sheet_accounts_payable_accrued"]
)

# ----------------------------------------------------
# 4. SOLVENCY RATIOS
# ----------------------------------------------------
df["debt_to_equity"] = safe_div(
    df["balance_sheet_total_liabilities"],
    df["balance_sheet_total_shareholders_equity"]
)

df["debt_ratio"] = safe_div(
    df["balance_sheet_total_liabilities"],
    df["balance_sheet_total_assets"]
)

df["times_interest_earned"] = safe_div(
    df["income_statement_profit_loss_before_taxes"],
    df["income_statement_interest_expense"]
)

# ----------------------------------------------------
# 5. EFFICIENCY RATIOS
# ----------------------------------------------------
df["asset_turnover"] = safe_div(
    df["income_statement_total_revenues"],
    df["balance_sheet_total_assets"]
)

df["days_sales_outstanding"] = 365 * safe_div(
    df["balance_sheet_accounts_receivable"],
    df["income_statement_total_revenues"]
)

df["inventory_turnover"] = safe_div(
    df["income_statement_total_revenues"],
    df["balance_sheet_inventory"]
)

# ----------------------------------------------------
# 6. BENFORD'S LAW ANOMALY DETECTION
# ----------------------------------------------------
def benford_first_digit(series):
    """Extract first nonzero digit from each value."""
    series = series.dropna().astype(str)
    # Remove non-numeric characters
    digits = series[series.str.contains(r'\d')].str.replace(r"[^0-9]", "", regex=True)
    # Strip leading zeros and take the first character
    digits = digits[digits != ""].str.lstrip("0").str[:1]
    # Filter out any remaining empty strings
    digits = digits[digits != ""]
    # Convert to float, return NaN if empty
    return digits.astype(float) if not digits.empty else pd.Series([])

def benford_test(series):
    """Perform chi-squared test comparing first-digit frequencies to Benford's Law."""
    digits = benford_first_digit(series)
    if digits.empty:
        return np.nan

    observed_counts = digits.value_counts().sort_index()
    total = observed_counts.sum()
    observed = observed_counts / total

    benford_probs = np.log10(1 + 1 / np.arange(1, 10))
    expected = pd.Series(benford_probs, index=np.arange(1, 10))

    # Align observed with expected
    observed = observed.reindex(expected.index, fill_value=0)

    chi_stat, p_value = chisquare(f_obs=observed * total, f_exp=expected * total)
    return chi_stat

# Apply Benford’s test to selected numeric financial columns
numeric_cols = [
    "balance_sheet_total_assets",
    "balance_sheet_total_liabilities",
    "income_statement_total_revenues",
    "income_statement_net_income_loss",
    "cash_flow_net_cash_operating"
]

for col in numeric_cols:
    new_col = f"benford_chi_{col}"
    df[new_col] = benford_test(df[col])

# ----------------------------------------------------
# 7. Summary Statistics of New Ratios
# ----------------------------------------------------
ratio_cols = [
    "gross_profit_margin", "return_on_assets", "return_on_equity", "operating_margin",
    "current_ratio", "quick_ratio", "debt_to_equity", "debt_ratio",
    "times_interest_earned", "asset_turnover", "days_sales_outstanding",
    "inventory_turnover"
]

print("\n✅ Ratio Feature Summary:")
print(df[ratio_cols].describe().T[["mean", "std", "min", "max"]])

# ----------------------------------------------------
# 8. Benford’s Law Visualization (optional)
# ----------------------------------------------------
def plot_benford(series, title="Benford's Law Test"):
    digits = benford_first_digit(series)
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
    plt.show()

# Example plot
plot_benford(df["income_statement_total_revenues"], "Benford Test: Total Revenues")

# ----------------------------------------------------
# 9. Save Enhanced Dataset
# ----------------------------------------------------
df.to_excel("output/financial_data_enhanced.xlsx", index=False)
print("\n✅ Enhanced dataset with ratios and Benford features saved to 'output/financial_data_enhanced.xlsx'")
