"""
Step 4: Association Rule Mining (Apriori & FP-Growth)
Credit Card Transaction Analysis & Fraud Detection
Algorithms: mlxtend.frequent_patterns.apriori & fpgrowth
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

OUTPUT_MODELS_DIR = os.path.join("outputs", "models")
OUTPUT_GRAPHS_DIR = os.path.join("outputs", "graphs")
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_GRAPHS_DIR, exist_ok=True)


def discretize_hour(hour):
    if 22 <= hour or hour < 5:
        return "hour=Late_Night"
    elif 5 <= hour < 12:
        return "hour=Morning"
    elif 12 <= hour < 17:
        return "hour=Afternoon"
    else:
        return "hour=Evening"


def prepare_transactions():
    clean_train_path = os.path.join("data", "processed", "fraudTrain_clean.csv")
    print(f"Loading cleaned dataset: {clean_train_path}...")
    df = pd.read_csv(clean_train_path)

    # Focus association mining on patterns associated with fraud:
    # All fraudulent transactions + an equal sample of genuine transactions for contrast
    fraud_df = df[df["is_fraud"] == 1]
    genuine_df = df[df["is_fraud"] == 0].sample(n=min(len(fraud_df) * 3, 30000), random_state=42)
    sample_df = pd.concat([fraud_df, genuine_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Prepared sample of {len(sample_df):,} transactions ({len(fraud_df):,} fraud cases) for itemset mining.")

    transactions = []
    for _, row in sample_df.iterrows():
        items = [
            f"cat={row['category']}",
            f"amt={row['amount_category']}",
            discretize_hour(row["transaction_hour"]),
            f"gender={row['gender']}",
            "weekend=Yes" if row["is_weekend"] == 1 else "weekend=No",
            "fraud=Yes" if row["is_fraud"] == 1 else "fraud=No"
        ]
        transactions.append(items)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    return df_encoded


def main():
    print("=" * 60)
    print("STAGE 4: ASSOCIATION RULE MINING")
    print("=" * 60)

    df_encoded = prepare_transactions()

    min_support = 0.05
    print(f"\nFinding Frequent Itemsets with min_support = {min_support}...")

    # 1. Apriori Benchmark
    print("\nRunning Apriori Algorithm...")
    t0 = time.time()
    apriori_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    apriori_time = time.time() - t0
    print(f"Apriori finished in {apriori_time:.3f} seconds. Found {len(apriori_itemsets)} itemsets.")

    # 2. FP-Growth Benchmark
    print("\nRunning FP-Growth Algorithm...")
    t0 = time.time()
    fpgrowth_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    fpgrowth_time = time.time() - t0
    print(f"FP-Growth finished in {fpgrowth_time:.3f} seconds. Found {len(fpgrowth_itemsets)} itemsets.")

    # 3. Association Rules Generation
    print("\nGenerating Association Rules (metric='lift', min_threshold=1.2)...")
    rules = association_rules(fpgrowth_itemsets, metric="lift", min_threshold=1.2)
    print(f"Total rules discovered: {len(rules)}")

    # Convert frozensets to readable strings
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(list(s)))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(list(s)))

    # Filter rules specifically predicting fraud
    fraud_rules = rules[rules["consequents_str"].str.contains("fraud=Yes")].sort_values(by="lift", ascending=False)
    print(f"\nRules predicting fraud=Yes: {len(fraud_rules)}")
    print(fraud_rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(10))

    # Save rules
    fraud_rules.to_csv(os.path.join(OUTPUT_MODELS_DIR, "fraud_association_rules.csv"), index=False)
    rules.to_csv(os.path.join(OUTPUT_MODELS_DIR, "all_association_rules.csv"), index=False)
    print(f"\nRules saved to {OUTPUT_MODELS_DIR}/")

    # Visualization 1: Apriori vs FP-Growth Execution Time Comparison
    plt.figure(figsize=(7, 5))
    bars = plt.bar(["Apriori", "FP-Growth"], [apriori_time, fpgrowth_time], color=["#e74c3c", "#2ecc71"], width=0.5)
    plt.title("Execution Time Comparison: Apriori vs FP-Growth", fontsize=13, fontweight="bold")
    plt.ylabel("Execution Time (seconds)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}s", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPHS_DIR, "12_apriori_vs_fpgrowth_runtime.png"), dpi=300)
    plt.close()

    # Visualization 2: Support vs Confidence vs Lift
    if len(rules) > 0:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="plasma", alpha=0.8, s=50)
        plt.colorbar(scatter, label="Lift")
        plt.title("Association Rules: Support vs Confidence (Colored by Lift)", fontsize=13, fontweight="bold")
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_GRAPHS_DIR, "13_association_rules_scatter.png"), dpi=300)
        plt.close()

    print("\n" + "=" * 60)
    print("SUCCESS: Association rule mining completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
