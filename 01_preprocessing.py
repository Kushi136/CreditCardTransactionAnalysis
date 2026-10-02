"""
Step 1: Data Preprocessing and Feature Engineering
Credit Card Transaction Analysis & Fraud Detection
"""

import os
import numpy as np
import pandas as pd

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) using the Haversine formula.
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def main():
    print("=" * 60)
    print("PART 4: Loading Datasets...")
    print("=" * 60)
    
    train_path = os.path.join("data", "fraudTrain.csv")
    test_path = os.path.join("data", "fraudTest.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Please make sure fraudTrain.csv and fraudTest.csv exist in the 'data' directory.\n"
            f"Looked for:\n - {train_path}\n - {test_path}"
        )

    print(f"Reading {train_path}...")
    train = pd.read_csv(train_path)
    print(f"Reading {test_path}...")
    test = pd.read_csv(test_path)

    print("\nTraining data shape:", train.shape)
    print("Testing data shape:", test.shape)
    print("\nColumns in dataset:\n", list(train.columns))

    print("\n" + "=" * 60)
    print("PART 5: Dataset Summary and Quality Checks")
    print("=" * 60)
    print("\nMissing values in Train:\n", train.isnull().sum()[train.isnull().sum() > 0])
    print(f"Duplicate records in Train: {train.duplicated().sum()}")
    print(f"Duplicate records in Test: {test.duplicated().sum()}")

    print("\n" + "=" * 60)
    print("PART 6: Date & Time Feature Extraction")
    print("=" * 60)
    for df_name, df in [("Train", train), ("Test", test)]:
        print(f"Extracting datetime features for {df_name}...")
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
        df["month"] = df["trans_date_trans_time"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    print("\n" + "=" * 60)
    print("PART 7: Customer Age Calculation")
    print("=" * 60)
    for df_name, df in [("Train", train), ("Test", test)]:
        print(f"Calculating customer age for {df_name}...")
        df["dob"] = pd.to_datetime(df["dob"])
        df["customer_age"] = (
            df["trans_date_trans_time"].dt.year - df["dob"].dt.year
        )

    print("\n" + "=" * 60)
    print("PART 8: Merchant Distance Calculation")
    print("=" * 60)
    for df_name, df in [("Train", train), ("Test", test)]:
        print(f"Calculating merchant distance for {df_name}...")
        df["merchant_distance"] = calculate_distance(
            df["lat"],
            df["long"],
            df["merch_lat"],
            df["merch_long"]
        )

    print("\n" + "=" * 60)
    print("PART 9: Amount Categorization")
    print("=" * 60)
    for df_name, df in [("Train", train), ("Test", test)]:
        print(f"Binning transaction amount for {df_name}...")
        df["amount_category"] = pd.cut(
            df["amt"],
            bins=[0, 50, 100, 500, 1000, float("inf")],
            labels=["Low", "Medium", "High", "Very High", "Extreme"]
        )

    print("\n" + "=" * 60)
    print("PART 10: Dropping Unnecessary & PII Columns")
    print("=" * 60)
    drop_columns = [
        "Unnamed: 0",
        "first",
        "last",
        "street",
        "trans_num",
        "dob",
        "cc_num",
        "unix_time",
        "lat",
        "long",
        "merch_lat",
        "merch_long"
    ]
    
    cols_to_drop_train = [c for c in drop_columns if c in train.columns]
    cols_to_drop_test = [c for c in drop_columns if c in test.columns]
    
    print(f"Dropping columns from Train: {cols_to_drop_train}")
    train_clean = train.drop(columns=cols_to_drop_train)
    print(f"Dropping columns from Test: {cols_to_drop_test}")
    test_clean = test.drop(columns=cols_to_drop_test)

    print("\nCleaned Train shape:", train_clean.shape)
    print("Cleaned Test shape:", test_clean.shape)
    print("Cleaned Train columns:\n", list(train_clean.columns))

    print("\n" + "=" * 60)
    print("PART 11: Saving Processed Datasets")
    print("=" * 60)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    
    train_clean_path = os.path.join("data", "processed", "fraudTrain_clean.csv")
    test_clean_path = os.path.join("data", "processed", "fraudTest_clean.csv")

    print(f"Saving {train_clean_path}...")
    train_clean.to_csv(train_clean_path, index=False)
    
    print(f"Saving {test_clean_path}...")
    test_clean.to_csv(test_clean_path, index=False)

    print("\n" + "=" * 60)
    print("SUCCESS: Preprocessing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
