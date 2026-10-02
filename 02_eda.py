"""
Step 2: Exploratory Data Analysis (EDA) & Visualizations
Credit Card Transaction Analysis & Fraud Detection
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set visual aesthetic
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

OUTPUT_DIR = os.path.join("outputs", "graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("PART 12: Loading Cleaned Data for EDA...")
    print("=" * 60)
    
    clean_train_path = os.path.join("data", "processed", "fraudTrain_clean.csv")
    if not os.path.exists(clean_train_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {clean_train_path}. Run 01_preprocessing.py first.")
        
    print(f"Reading {clean_train_path}...")
    train = pd.read_csv(clean_train_path)
    print(f"Loaded {len(train):,} records with {len(train.columns)} columns.")

    # -------------------------------------------------------------
    # EDA 1 & 2: Fraud vs Genuine Count and Percentage
    # -------------------------------------------------------------
    print("\n--- EDA 1 & 2: Fraud vs Genuine Distribution ---")
    fraud_counts = train["is_fraud"].value_counts()
    fraud_pct = train["is_fraud"].value_counts(normalize=True) * 100
    print("Counts:\n", fraud_counts)
    print("Percentages:\n", fraud_pct)

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=train, x="is_fraud", palette="coolwarm")
    plt.title("Distribution of Genuine vs Fraudulent Transactions", fontsize=14, fontweight="bold")
    plt.xlabel("Transaction Type (0 = Genuine, 1 = Fraud)")
    plt.ylabel("Number of Transactions")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_fraud_vs_genuine.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # EDA 3: Transaction Amount Distribution & Boxplot
    # -------------------------------------------------------------
    print("\n--- EDA 3: Transaction Amount Distribution ---")
    plt.figure(figsize=(10, 5))
    sns.histplot(data=train[train["amt"] < 1000], x="amt", bins=50, kde=True, color="#2b5c8f")
    plt.title("Transaction Amount Distribution (< $1,000)", fontsize=14, fontweight="bold")
    plt.xlabel("Transaction Amount ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_amount_distribution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train, x="is_fraud", y="amt", palette="Set2", showfliers=False)
    plt.title("Transaction Amount by Fraud Status (Outliers Excluded from View)", fontsize=14, fontweight="bold")
    plt.xlabel("Fraud Status (0 = Genuine, 1 = Fraud)")
    plt.ylabel("Transaction Amount ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_amount_boxplot.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # EDA 4: Transactions & Fraud by Category
    # -------------------------------------------------------------
    print("\n--- EDA 4: Transactions by Category ---")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train, x="category", hue="is_fraud", palette="Set1")
    plt.title("Transaction Frequency by Category and Fraud Status", fontsize=14, fontweight="bold")
    plt.xlabel("Merchant Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Is Fraud", labels=["Genuine", "Fraud"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_category_fraud.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # EDA 5: Fraud by Transaction Hour
    # -------------------------------------------------------------
    print("\n--- EDA 5: Fraud by Hour ---")
    plt.figure(figsize=(12, 5))
    fraud_only = train[train["is_fraud"] == 1]
    sns.countplot(data=fraud_only, x="transaction_hour", palette="viridis")
    plt.title("Fraudulent Transactions by Hour of Day", fontsize=14, fontweight="bold")
    plt.xlabel("Hour of Day (0 - 23)")
    plt.ylabel("Number of Fraud Transactions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_fraud_by_hour.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # EDA 6: Fraud by Day of Week
    # -------------------------------------------------------------
    print("\n--- EDA 6: Fraud by Day of Week ---")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plt.figure(figsize=(10, 5))
    sns.countplot(data=fraud_only, x="day_of_week", palette="magma")
    plt.title("Fraudulent Transactions by Day of the Week", fontsize=14, fontweight="bold")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Fraud Transactions")
    plt.xticks(ticks=range(7), labels=day_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_fraud_by_day.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # EDA 7: Fraud by Gender
    # -------------------------------------------------------------
    print("\n--- EDA 7: Fraud by Gender ---")
    plt.figure(figsize=(7, 5))
    sns.countplot(data=train, x="gender", hue="is_fraud", palette="pastel")
    plt.title("Transaction Breakdown by Gender and Fraud Status", fontsize=14, fontweight="bold")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.legend(title="Is Fraud", labels=["Genuine", "Fraud"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_fraud_by_gender.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # EDA 8: Fraud by State (Top 10)
    # -------------------------------------------------------------
    print("\n--- EDA 8: Top States with Highest Fraud ---")
    top_states = fraud_only["state"].value_counts().head(10)
    print("Top 10 States with highest fraud occurrences:\n", top_states)
    
    plt.figure(figsize=(10, 5))
    top_states.plot(kind="bar", color="#d9534f", edgecolor="black")
    plt.title("Top 10 States by Number of Fraudulent Transactions", fontsize=14, fontweight="bold")
    plt.xlabel("State")
    plt.ylabel("Number of Fraud Transactions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_fraud_by_state.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # EDA 9: Customer Age Distribution
    # -------------------------------------------------------------
    print("\n--- EDA 9: Customer Age Distribution ---")
    plt.figure(figsize=(10, 5))
    sns.histplot(data=train, x="customer_age", hue="is_fraud", bins=30, kde=True, palette=["#3498db", "#e74c3c"])
    plt.title("Customer Age Distribution: Genuine vs Fraud", fontsize=14, fontweight="bold")
    plt.xlabel("Customer Age (Years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_fraud_by_age.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print(f"SUCCESS: All 9 EDA graphs saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
