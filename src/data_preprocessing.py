"""
Data Preprocessing Module for Customer Churn Prediction
Handles raw data loading, data cleaning, feature categorization,
Scikit-Learn ColumnTransformer pipeline definition, and stratified data splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

TARGET_COLUMN = 'Churn'


def load_raw_data(csv_path: str = "data/customer_churn.csv") -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame):
    """
    Clean dataset:
    - Handle blank TotalCharges strings by converting to float and imputing 0.0 for tenure=0.
    - Drop customerID identifier column.
    - Ensure SeniorCitizen is treated as categorical string.
    - Map target variable Churn (Yes -> 1, No -> 0).

    Returns:
        df_clean (pd.DataFrame): Cleaned full DataFrame.
        X (pd.DataFrame): Features dataframe.
        y (pd.Series): Binary target series.
    """
    df_clean = df.copy()

    # TotalCharges contains blank strings ' ' for 11 customers with tenure == 0
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)

    # Drop identifier
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])

    # Ensure SeniorCitizen is categorical
    df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(str)

    # Separate X and y
    y = (df_clean[TARGET_COLUMN] == 'Yes').astype(int)
    X = df_clean.drop(columns=[TARGET_COLUMN])

    return df_clean, X, y


def get_preprocessor() -> ColumnTransformer:
    """
    Construct a leak-proof ColumnTransformer:
    - StandardScaler applied to numerical features.
    - OneHotEncoder(drop='first') applied to categorical features.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42):
    """
    Split feature matrix and target into train and test sets using stratification.
    Stratification ensures both sets preserve the 73.5% / 26.5% class balance.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


if __name__ == "__main__":
    df = load_raw_data()
    df_clean, X, y = clean_data(df)
    print(f"Data shape after cleaning: {df_clean.shape}")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")
