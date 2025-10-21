# financial_fraud_preprocessing

Modular preprocessing package for financial fraud detection (FYP).

## Installation

Place the `financial_fraud_preprocessing` folder in your project directory and import it with:

```python
from financial_fraud_preprocessing.preprocessor import FraudDataPreprocessor
from financial_fraud_preprocessing.eda import FraudEDA
```

## Usage

Run from command line:

```bash
python -m financial_fraud_preprocessing.main --input /path/to/financial_data.csv
```

Or import and call programmatically:

```python
from financial_fraud_preprocessing.preprocessor import FraudDataPreprocessor
pre = FraudDataPreprocessor()
df_A, df_B = pre.preprocess_version_a(df)  # version A returns (processed_df, dropped_columns)
df_B = pre.preprocess_version_b(df)  # version B
```

Notes:

- The package **does not save CSV files** (by your preference), it returns processed dataframes in memory ready for modeling.
- Visualizations are produced by `eda.py` functions and displayed when run.
