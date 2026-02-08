import pandas as pd
import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt

# Load dataset
file_path = "output/financial_data.xlsx"
df = pd.read_excel(file_path)

# Safe division function
def safe_div(numer, denom):
    """Safe divide to avoid division by zero."""
    return np.where((denom == 0) | (denom.isna()), np.nan, numer / denom)

# ----------------------------------------------------
# 1. ORIGINAL RATIOS (from your initial code)
# ----------------------------------------------------

# --- Profitability Ratios ---
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

# --- Liquidity Ratios ---
df["current_ratio"] = safe_div(
    df["balance_sheet_total_current_assets"],
    df["balance_sheet_accounts_payable_accrued"]
)

df["quick_ratio"] = safe_div(
    (df["balance_sheet_cash"] + df["balance_sheet_accounts_receivable"]),
    df["balance_sheet_accounts_payable_accrued"]
)

# --- Solvency Ratios ---
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

# --- Efficiency Ratios ---
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

# --- Benford's Law Anomaly Detection ---
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
    observed = observed_counts / total

    benford_probs = np.log10(1 + 1 / np.arange(1, 10))
    expected = pd.Series(benford_probs, index=np.arange(1, 10))

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

# for col in numeric_cols:
#     new_col = f"benford_chi_{col}"
#     df[new_col] = benford_test(df[col])

# ----------------------------------------------------
# 2. NEW FEATURES (from previous response, excluding operating_leverage_ratio)
# ----------------------------------------------------

# --- Profitability Ratios ---
df["net_profit_margin"] = safe_div(
    df["income_statement_net_income_loss"],
    df["income_statement_total_revenues"]
)

df["capital_employed"] = df["balance_sheet_total_assets"] - df["balance_sheet_accounts_payable_accrued"]
df["ebit"] = df["income_statement_profit_loss_operations"]
df["roce"] = safe_div(
    df["ebit"],
    df["capital_employed"]
)

df["ebitda"] = df["income_statement_profit_loss_operations"] + df["income_statement_depreciation_amortization"]
df["ebitda_margin"] = safe_div(
    df["ebitda"],
    df["income_statement_total_revenues"]
)

# --- Liquidity Ratios ---
df["cash_ratio"] = safe_div(
    df["balance_sheet_cash"],
    df["balance_sheet_accounts_payable_accrued"]
)

df["operating_cash_flow_ratio"] = safe_div(
    df["cash_flow_net_cash_operating"],
    df["balance_sheet_accounts_payable_accrued"]
)

# --- Solvency Ratios ---
df["total_capital"] = df["balance_sheet_total_liabilities"] + df["balance_sheet_total_shareholders_equity"]
df["debt_to_capital"] = safe_div(
    df["balance_sheet_total_liabilities"],
    df["total_capital"]
)

df["interest_coverage_ebitda"] = safe_div(
    df["ebitda"],
    df["income_statement_interest_expense"]
)

# --- Efficiency Ratios ---
df["receivables_turnover"] = safe_div(
    df["income_statement_total_revenues"],
    df["balance_sheet_accounts_receivable"]
)

df["fixed_asset_turnover"] = safe_div(
    df["income_statement_total_revenues"],
    df["balance_sheet_fixed_assets_net"]
)

df["working_capital"] = df["balance_sheet_total_current_assets"] - df["balance_sheet_accounts_payable_accrued"]
df["working_capital_turnover"] = safe_div(
    df["income_statement_total_revenues"],
    df["working_capital"]
)

# --- Growth and Stability Metrics ---
df = df.sort_values(["company_name", "year"])
df["prev_total_revenues"] = df.groupby("company_name")["income_statement_total_revenues"].shift(1)
df["revenue_growth_rate"] = safe_div(
    df["income_statement_total_revenues"] - df["prev_total_revenues"],
    df["prev_total_revenues"]
)

df["equity_to_assets"] = safe_div(
    df["balance_sheet_total_shareholders_equity"],
    df["balance_sheet_total_assets"]
)

# --- Cash Flow Metrics ---
df["capital_expenditures"] = -df["cash_flow_sale_purchase_fixed_assets"]
df["free_cash_flow"] = df["cash_flow_net_cash_operating"] - df["capital_expenditures"]

df["cash_flow_to_debt"] = safe_div(
    df["cash_flow_net_cash_operating"],
    df["balance_sheet_total_liabilities"]
)

# --- Fraud Detection Features ---
df["expense_to_revenue"] = safe_div(
    df["income_statement_total_operating_expenses"],
    df["income_statement_total_revenues"]
)

df["related_party_ratio"] = safe_div(
    df["balance_sheet_due_to_related_parties"],
    df["balance_sheet_total_liabilities"]
)

# --- Valuation Metrics ---
df["book_value_per_share"] = safe_div(
    df["balance_sheet_total_shareholders_equity"],
    df["income_statement_shares_outstanding"]
)

# ----------------------------------------------------
# 3. Summary Statistics of All Features
# ----------------------------------------------------
ratio_cols = [
    "gross_profit_margin", "return_on_assets", "return_on_equity", "operating_margin",
    "current_ratio", "quick_ratio", "debt_to_equity", "debt_ratio",
    "times_interest_earned", "asset_turnover", "days_sales_outstanding",
    "inventory_turnover",
    "net_profit_margin", "roce", "ebitda_margin",
    "cash_ratio", "operating_cash_flow_ratio",
    "debt_to_capital", "interest_coverage_ebitda",
    "receivables_turnover", "fixed_asset_turnover", "working_capital_turnover",
    "revenue_growth_rate", "equity_to_assets",
    "free_cash_flow", "cash_flow_to_debt",
    "expense_to_revenue", "related_party_ratio",
    "book_value_per_share"
]

benford_cols = [f"benford_chi_{col}" for col in numeric_cols]

print("\n✅ Ratio Feature Summary:")
print(df[ratio_cols].describe().T[["mean", "std", "min", "max"]])

# print("\n✅ Benford’s Law Chi-Square Statistics Summary:")
# print(df[benford_cols].describe().T[["mean", "std", "min", "max"]])

# ----------------------------------------------------
# 4. Benford’s Law Visualization (unchanged from original)
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
#plot_benford(df["income_statement_total_revenues"], "Benford Test: Total Revenues")

# ----------------------------------------------------
# 5. Save Enhanced Dataset
# ----------------------------------------------------
df.to_excel("output/financial_data_super_enhanced.xlsx", index=False)
print("\n✅ Enhanced dataset with all features saved to 'output/financial_data_super_enhanced.xlsx'")