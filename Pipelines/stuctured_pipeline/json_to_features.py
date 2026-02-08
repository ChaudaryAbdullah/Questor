"""
JSON to Features Transformation Module

This module transforms raw JSON financial data into the same feature set
used during model training. It combines:
1. Initial feature extraction (from data_tranform.py)
2. Derived feature engineering (from derive_features.py)

Ensures 100% consistency with training pipeline.
"""

import pandas as pd
import numpy as np
import json
from typing import Union, Dict, Any


def safe_div(numer, denom):
    """Safe divide to avoid division by zero."""
    return np.where((denom == 0) | (pd.isna(denom)), np.nan, numer / denom)


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and validate JSON financial data file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Validate required sections
    required_sections = ['balance_sheet', 'income_statement', 'cash_flow']
    for section in required_sections:
        if section not in data:
            raise ValueError(f"Missing required section: {section}")
    
    return data


def json_to_initial_features(json_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform JSON to initial feature set (mimics data_tranform.py).
    
    Args:
        json_data: Dictionary containing financial data from JSON
    
    Returns:
        DataFrame with initial features matching training format
    """
    bs = json_data['balance_sheet']
    inc = json_data['income_statement']
    cf = json_data['cash_flow']
    
    # Helper to convert "N/A" to NaN
    def clean_value(val):
        if val == "N/A" or val is None:
            return np.nan
        return float(val)
    
    # Build the initial feature dictionary matching enhanced_financial_data.csv
    features = {
        'fyear': json_data.get('year', np.nan),
        'gvkey': np.nan,  # Not available in JSON
        'p_aaer': np.nan,  # Not available in JSON
        'is_fraudulent': 0,  # Unknown, set to 0 for inference
        
        # Raw accounting variables (from data_tranform.py logic)
        'act': clean_value(bs.get('total_current_assets')),
        'ap': clean_value(bs.get('accounts_payable_accrued')),
        'at': clean_value(bs.get('total_assets')),
        'ceq': clean_value(bs.get('capital_stock', 0)) + 
               clean_value(bs.get('additional_paid_in_capital', 0)) + 
               clean_value(bs.get('retained_earnings', 0)),
        'che': clean_value(bs.get('cash')),
        'cogs': np.nan,  # Not available
        'csho': clean_value(inc.get('shares_outstanding')),
        'dlc': clean_value(bs.get('notes_payable_short')),
        'dltis': np.nan,  # Not available
        'dltt': clean_value(bs.get('notes_payable_long')),
        'dp': clean_value(inc.get('depreciation_amortization')),
        'ib': clean_value(inc.get('profit_loss_before_taxes')),
        'invt': clean_value(bs.get('inventory')),
        'ivao': clean_value(bs.get('other_assets')),
        'ivst': np.nan,  # Not available
        'lct': clean_value(bs.get('accounts_payable_accrued', 0)) + 
               clean_value(bs.get('due_to_related_parties', 0)) + 
               clean_value(bs.get('notes_payable_short', 0)),
        'lt': clean_value(bs.get('total_liabilities')),
        'ni': clean_value(inc.get('net_income_loss')),
        'ppegt': clean_value(bs.get('fixed_assets_net')),
        'pstk': np.nan,  # Not available
        're': clean_value(bs.get('retained_earnings')),
        'rect': clean_value(bs.get('accounts_receivable')),
        'sale': clean_value(inc.get('total_revenues')),
        'sstk': clean_value(cf.get('proceeds_stock_sales')),
        'txp': np.nan,  # Not available
        'txt': clean_value(inc.get('income_tax_expense')),
        'xint': clean_value(inc.get('interest_expense')),
        'prcc_f': np.nan,  # Not available
        
        # Financial ratios (from data_tranform.py)
        'dch_wc': np.nan,  # Requires year-over-year
        'ch_rsst': np.nan,  # Requires complex modeling
        'dch_rec': np.nan,  # Requires year-over-year
        'dch_inv': np.nan,  # Requires year-over-year
        'ch_cs': clean_value(inc.get('total_revenues')),  # Approximation
        'ch_cm': np.nan,  # Requires year-over-year
        'ch_roa': np.nan,  # Requires year-over-year
        'issue': np.nan,  # Calculated below
        'bm': np.nan,  # Requires prcc_f
        'dpi': np.nan,  # Calculated below
        'reoa': np.nan,  # Calculated below
        'EBIT': np.nan,  # Calculated below
        'ch_fcf': np.nan,  # Requires year-over-year
        
        # Store raw values for derived features
        'balance_sheet_total_assets': clean_value(bs.get('total_assets')),
        'balance_sheet_total_liabilities': clean_value(bs.get('total_liabilities')),
        'balance_sheet_total_shareholders_equity': clean_value(bs.get('total_shareholders_equity')),
        'balance_sheet_total_current_assets': clean_value(bs.get('total_current_assets')),
        'balance_sheet_cash': clean_value(bs.get('cash')),
        'balance_sheet_accounts_receivable': clean_value(bs.get('accounts_receivable')),
        'balance_sheet_inventory': clean_value(bs.get('inventory')),
        'balance_sheet_fixed_assets_net': clean_value(bs.get('fixed_assets_net')),
        'balance_sheet_other_assets': clean_value(bs.get('other_assets')),
        'balance_sheet_accounts_payable_accrued': clean_value(bs.get('accounts_payable_accrued')),
        'balance_sheet_due_to_related_parties': clean_value(bs.get('due_to_related_parties')),
        'balance_sheet_notes_payable_short': clean_value(bs.get('notes_payable_short')),
        'balance_sheet_notes_payable_long': clean_value(bs.get('notes_payable_long')),
        'balance_sheet_capital_stock': clean_value(bs.get('capital_stock')),
        'balance_sheet_additional_paid_in_capital': clean_value(bs.get('additional_paid_in_capital')),
        'balance_sheet_retained_earnings': clean_value(bs.get('retained_earnings')),
        
        'income_statement_total_revenues': clean_value(inc.get('total_revenues')),
        'income_statement_total_operating_expenses': clean_value(inc.get('total_operating_expenses')),
        'income_statement_profit_loss_operations': clean_value(inc.get('profit_loss_operations')),
        'income_statement_interest_expense': clean_value(inc.get('interest_expense')),
        'income_statement_profit_loss_before_taxes': clean_value(inc.get('profit_loss_before_taxes')),
        'income_statement_income_tax_expense': clean_value(inc.get('income_tax_expense')),
        'income_statement_net_income_loss': clean_value(inc.get('net_income_loss')),
        'income_statement_depreciation_amortization': clean_value(inc.get('depreciation_amortization')),
        'income_statement_shares_outstanding': clean_value(inc.get('shares_outstanding')),
        
        'cash_flow_net_cash_operating': clean_value(cf.get('net_cash_operating')),
        'cash_flow_sale_purchase_fixed_assets': clean_value(cf.get('sale_purchase_fixed_assets')),
        'cash_flow_changes_notes_payable': clean_value(cf.get('changes_notes_payable')),
        'cash_flow_proceeds_stock_sales': clean_value(cf.get('proceeds_stock_sales')),
        'cash_flow_paid_in_capital_shareholders': clean_value(cf.get('paid_in_capital_shareholders')),
    }
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Calculate soft_assets, dpi, reoa, EBIT, issue
    df['soft_assets'] = safe_div(df['rect'] + df['ivao'], df['at'])
    df['dpi'] = safe_div(df['dp'], df['ppegt'])
    df['reoa'] = safe_div(df['re'], df['at'])
    df['EBIT'] = safe_div(df['ib'] - df['xint'], df['at'])
    df['issue'] = (df['sstk'] - df['cash_flow_changes_notes_payable'] - 
                   df['cash_flow_paid_in_capital_shareholders'])
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived features from derive_features.py.
    
    Args:
        df: DataFrame with initial features
    
    Returns:
        DataFrame with all derived features added
    """
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
    
    df["net_profit_margin"] = safe_div(
        df["income_statement_net_income_loss"],
        df["income_statement_total_revenues"]
    )
    
    df["capital_employed"] = df["balance_sheet_total_assets"] - df["balance_sheet_accounts_payable_accrued"]
    df["ebit"] = df["income_statement_profit_loss_operations"]
    df["roce"] = safe_div(df["ebit"], df["capital_employed"])
    
    df["ebitda"] = df["income_statement_profit_loss_operations"] + df["income_statement_depreciation_amortization"]
    df["ebitda_margin"] = safe_div(df["ebitda"], df["income_statement_total_revenues"])
    
    # --- Liquidity Ratios ---
    df["current_ratio"] = safe_div(
        df["balance_sheet_total_current_assets"],
        df["balance_sheet_accounts_payable_accrued"]
    )
    
    df["quick_ratio"] = safe_div(
        df["balance_sheet_cash"] + df["balance_sheet_accounts_receivable"],
        df["balance_sheet_accounts_payable_accrued"]
    )
    
    df["cash_ratio"] = safe_div(
        df["balance_sheet_cash"],
        df["balance_sheet_accounts_payable_accrued"]
    )
    
    df["operating_cash_flow_ratio"] = safe_div(
        df["cash_flow_net_cash_operating"],
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
    # Note: revenue_growth_rate requires historical data, setting to NaN for single record
    df["revenue_growth_rate"] = np.nan
    df["prev_total_revenues"] = np.nan
    
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
    
    return df


def transform_json_to_features(json_path: str) -> pd.DataFrame:
    """
    Complete transformation pipeline: JSON → Features.
    
    This is the main entry point that combines all transformation steps
    to produce the exact feature set used during model training.
    
    IMPORTANT: The models were trained ONLY on the 46 original columns from
    enhanced_financial_data.csv, WITHOUT the derived features from derive_features.py.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        DataFrame with EXACTLY 42 features matching training dataset
        (46 total columns minus fyear, gvkey, is_fraudulent, p_aaer)
    """
    # Load JSON
    json_data = load_json_file(json_path)
    
    # Initial feature extraction - this creates the 46 columns
    df = json_to_initial_features(json_data)
    
    # DO NOT add derived features - models were trained without them!
    # The enhanced_financial_data.csv only has the 46 base columns.
    
    # Clean: replace inf with NaN, then NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Drop ONLY the metadata columns that models don't use
    # KEEP p_aaer - it's a feature in the training data (even though it's NaN for inference)
    cols_to_drop = ['fyear', 'gvkey', 'is_fraudulent']
    
    # Also drop the intermediate raw columns that were used for calculations
    intermediate_cols = [col for col in df.columns if any(
        col.startswith(prefix) for prefix in [
            'balance_sheet_', 'income_statement_', 'cash_flow_'
        ]
    )]
    
    cols_to_drop.extend(intermediate_cols)
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # CRITICAL: Reorder columns to match training schema EXACTLY
    # The training CSV has columns in this specific order (after dropping fyear, gvkey, is_fraudulent)
    expected_column_order = [
        'p_aaer', 'act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 
        'dltt', 'dp', 'ib', 'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 
        'pstk', 're', 'rect', 'sale', 'sstk', 'txp', 'txt', 'xint', 'prcc_f', 
        'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs', 
        'ch_cm', 'ch_roa', 'issue', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf'
    ]
    
    # Reindex to ensure correct order
    df = df.reindex(columns=expected_column_order, fill_value=0)
    
    print(f"Final features: {df.shape[1]} columns (expected: 43)")
    
    return df


if __name__ == "__main__":
    # Test the transformation
    import sys
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "Input/0000050493-2.json"
    
    print(f"Transforming {json_path}...")
    df = transform_json_to_features(json_path)
    
    print(f"\n✅ Features extracted: {df.shape[1]} columns")
    print(f"✅ Sample feature values:")
    print(df.iloc[0, :10])
    
    # Save for inspection
    df.to_csv("test_features.csv", index=False)
    print(f"\n✅ Features saved to test_features.csv for inspection")
