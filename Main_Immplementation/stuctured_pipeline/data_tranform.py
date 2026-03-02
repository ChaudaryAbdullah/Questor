import pandas as pd
import numpy as np

# Load the dataset
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Transform dataset to the specified order with calculated/empty columns
def transform_dataset(df):
    # Initialize the new DataFrame with the specified order
    new_columns = ['fyear', 'gvkey', 'p_aaer', 'misstate', 'act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 
                   'dlc', 'dltis', 'dltt', 'dp', 'ib', 'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 
                   'pstk', 're', 'rect', 'sale', 'sstk', 'txp', 'txt', 'xint', 'prcc_f', 'dch_wc', 'ch_rsst', 
                   'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa', 'issue', 'bm', 'dpi', 'reoa', 
                   'EBIT', 'ch_fcf']
    transformed_df = pd.DataFrame(columns=new_columns)

    # Copy existing columns that match (if any)
    transformed_df['fyear'] = df['year']  # Assuming 'year' maps to 'fyear'
    transformed_df['gvkey'] = np.nan  # Placeholder, no gvkey in raw data
    transformed_df['p_aaer'] = np.nan  # Placeholder, no p_aaer in raw data
    if 'is_fraudulent' in df.columns:  # Check if column exists
        # Handle string, boolean, or integer formats
        if df['is_fraudulent'].dtype == 'object':  # String case (e.g., 'TRUE', 'FALSE')
            transformed_df['misstate'] = df['is_fraudulent'].map(
                lambda x: 1 if str(x).strip().upper() == 'TRUE' else 0 if str(x).strip().upper() == 'FALSE' else np.nan
            ).fillna(0)  # Default to 0 if unmapped
        elif df['is_fraudulent'].dtype == 'bool':  # Boolean case (True/False)
            transformed_df['misstate'] = df['is_fraudulent'].astype(int)  # True->1, False->0
        else:  # Assume integer case (1/0) or numeric
            transformed_df['misstate'] = df['is_fraudulent'].fillna(0).astype(int)  # Default to 0 if NaN
    else:
        transformed_df['misstate'] = 0  # Default to 0 if column is missing
    

    # Calculate raw accounting variables
    transformed_df['act'] = df['balance_sheet_total_current_assets']
    transformed_df['ap'] = df['balance_sheet_accounts_payable_accrued']  # Assumption: includes trade payables
    transformed_df['at'] = df['balance_sheet_total_assets']
    transformed_df['ceq'] = (df['balance_sheet_capital_stock'] + df['balance_sheet_additional_paid_in_capital'] + 
                             df['balance_sheet_retained_earnings'])
    transformed_df['che'] = df['balance_sheet_cash']  # Assumption: includes short-term investments
    transformed_df['cogs'] = np.nan  # Not calculable
    transformed_df['csho'] = df['income_statement_shares_outstanding']
    transformed_df['dlc'] = df['balance_sheet_notes_payable_short']  # Assumption: main current debt
    transformed_df['dltis'] = np.nan  # Not calculable
    transformed_df['dltt'] = df['balance_sheet_notes_payable_long']  # Assumption: main long-term debt
    transformed_df['dp'] = df['income_statement_depreciation_amortization']
    transformed_df['ib'] = df['income_statement_profit_loss_before_taxes']  # Assumption: excludes extraordinary items
    transformed_df['invt'] = df['balance_sheet_inventory']
    transformed_df['ivao'] = df['balance_sheet_other_assets']  # Assumption: dominated by investments
    transformed_df['ivst'] = np.nan  # Not calculable
    transformed_df['lct'] = (df['balance_sheet_accounts_payable_accrued'] + df['balance_sheet_due_to_related_parties'] + 
                             df['balance_sheet_notes_payable_short'])
    transformed_df['lt'] = df['balance_sheet_total_liabilities']
    transformed_df['ni'] = df['income_statement_net_income_loss']
    transformed_df['ppegt'] = df['balance_sheet_fixed_assets_net']
    transformed_df['pstk'] = np.nan  # Not calculable
    transformed_df['re'] = df['balance_sheet_retained_earnings']
    transformed_df['rect'] = df['balance_sheet_accounts_receivable']
    transformed_df['sale'] = df['income_statement_total_revenues']
    transformed_df['sstk'] = df['cash_flow_proceeds_stock_sales']  # Assumption: includes stock sales
    transformed_df['txp'] = np.nan  # Not calculable
    transformed_df['txt'] = df['income_statement_income_tax_expense']
    transformed_df['xint'] = df['income_statement_interest_expense']
    transformed_df['prcc_f'] = np.nan  # Not calculable without stock price

    # Calculate financial ratios (single-year approximations where possible)
    transformed_df['dch_wc'] = np.nan  # Requires year-over-year data
    transformed_df['ch_rsst'] = np.nan  # Requires complex modeling
    transformed_df['dch_rec'] = np.nan  # Requires year-over-year data
    transformed_df['dch_inv'] = np.nan  # Requires year-over-year data
    transformed_df['soft_assets'] = (transformed_df['rect'] + transformed_df['ivao']) / transformed_df['at']
    transformed_df['ch_cs'] = transformed_df['sale']  # Approximation for single year
    transformed_df['ch_cm'] = np.nan  # Requires year-over-year data
    transformed_df['ch_roa'] = np.nan  # Requires year-over-year data
    transformed_df['issue'] = (transformed_df['sstk'] - df['cash_flow_changes_notes_payable'] - 
                               df['cash_flow_paid_in_capital_shareholders'])  # Partial approximation
    transformed_df['bm'] = np.nan  # Requires prcc_f
    transformed_df['dpi'] = transformed_df['dp'] / transformed_df['ppegt']
    transformed_df['reoa'] = transformed_df['re'] / transformed_df['at']
    transformed_df['EBIT'] = (transformed_df['ib'] - transformed_df['xint']) / transformed_df['at']
    transformed_df['ch_fcf'] = np.nan  # Requires year-over-year data

    # Drop original columns not needed in the transformed dataset
    transformed_df = transformed_df[new_columns]  # Ensure exact order

    return transformed_df

# Main function to execute the transformation
def main():
    # Load the dataset
    df = load_data('output/financial_data.xlsx')  # Update with your file path

    # Transform the dataset
    transformed_df = transform_dataset(df)

    # Save the transformed dataset
    transformed_df.to_csv('enhanced_financial_data.csv', index=False)
    print("Dataset transformed and saved as 'enhanced_financial_data.csv'.")

if __name__ == "__main__":
    main()