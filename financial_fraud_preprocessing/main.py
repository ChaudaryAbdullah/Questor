import argparse
import logging
import pandas as pd
from .utils import setup_logging
from .eda import FraudEDA
from .preprocessor import FraudDataPreprocessor


def run(input_path: str):
    setup_logging()
    logging.info('Loading dataset...')
    df = pd.read_csv(input_path)

    eda = FraudEDA()
    pre = FraudDataPreprocessor()

    logging.info('Running EDA: missing summary and class balance')
    print(eda.missing_summary(df, top_n=20))
    eda.plot_missing_bar(df, top_n=20)
    eda.plot_class_balance(df)

    logging.info('Generating numeric summary...')
    print(eda.show_numeric_summary(df, exclude=['is_fraudulent', 'company_name', 'cik']))

    logging.info('Running preprocessing Version A (drop high-missing + median)')
    df_A, dropped = pre.preprocess_version_a(df)
    logging.info(f'Version A shape: {df_A.shape}')

    logging.info('Running preprocessing Version B (keep all + KNN)')
    df_B = pre.preprocess_version_b(df)
    logging.info(f'Version B shape: {df_B.shape}')

    # Return both for programmatic usage
    return df_A, df_B


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run EDA and preprocessing for financial fraud dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    args = parser.parse_args()
    run(args.input)