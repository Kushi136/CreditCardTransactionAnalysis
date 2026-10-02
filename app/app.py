"""
Credit Card Transaction Analysis & Fraud Detection System
Flask Web Application
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "random_forest.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "model_comparison.json")
RULES_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "fraud_association_rules.csv")

# Load model and artifacts
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Random Forest model successfully loaded.")

model_metrics = []
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        model_metrics = json.load(f)

CATEGORIES = [
    "grocery_pos", "gas_transport", "shopping_net", "shopping_pos",
    "food_dining", "personal_care", "entertainment", "health_fitness",
    "misc_pos", "misc_net", "travel", "home", "kids_pets", "grocery_net"
]


@app.route("/")
def dashboard():
    # Overall summary metrics based on actual train + test datasets
    kpis = {
        "total_transactions": "1,852,394",
        "train_count": "1,296,675",
        "test_count": "555,719",
        "fraud_count": "9,651",
        "fraud_rate": "0.52%",
        "best_model": "Random Forest",
        "best_roc_auc": "0.9947"
    }
    return render_template("dashboard.html", kpis=kpis, metrics=model_metrics)


@app.route("/eda")
def eda():
    graphs = [
        {
            "id": "01",
            "file": "01_fraud_vs_genuine.png",
            "title": "EDA 1 & 2: Genuine vs Fraudulent Distribution",
            "insight": "Extreme class imbalance is evident: out of 1.29M transactions, only ~7,506 (0.58%) are fraudulent. Accuracy alone is misleading; Recall and ROC-AUC are paramount."
        },
        {
            "id": "02",
            "file": "02_amount_distribution.png",
            "title": "EDA 3: Transaction Amount Distribution (< $1,000)",
            "insight": "The vast majority of genuine everyday retail transactions fall under $100, showing a strong right-skewed distribution."
        },
        {
            "id": "03",
            "file": "03_amount_boxplot.png",
            "title": "EDA 3: Transaction Amount: Fraud vs Genuine Boxplot",
            "insight": "Fraudulent transactions have significantly higher median amounts and broader upper quartiles compared to genuine purchases."
        },
        {
            "id": "04",
            "file": "04_category_fraud.png",
            "title": "EDA 4: Transactions & Fraud by Merchant Category",
            "insight": "Fraud disproportionately concentrates in grocery_pos, shopping_net, and misc_net, while dining and transportation have relatively lower fraud rates."
        },
        {
            "id": "05",
            "file": "05_fraud_by_hour.png",
            "title": "EDA 5: Fraudulent Transactions by Hour of Day",
            "insight": "A noticeable spike in fraudulent transactions occurs during late night/early morning hours (22:00 to 03:00) when cardholders are asleep."
        },
        {
            "id": "06",
            "file": "06_fraud_by_day.png",
            "title": "EDA 6: Fraud by Day of the Week",
            "insight": "Weekends (Saturday & Sunday) and Mondays register elevated fraud activities compared to mid-week days."
        },
        {
            "id": "07",
            "file": "07_fraud_by_gender.png",
            "title": "EDA 7: Fraud Distribution by Gender",
            "insight": "Both female and male cardholders experience comparable proportions of fraud, indicating fraud targeting is merchant/time-driven rather than gender-driven."
        },
        {
            "id": "08",
            "file": "08_fraud_by_state.png",
            "title": "EDA 8: Top 10 States with Highest Fraud Counts",
            "insight": "High population states (NY, TX, PA, CA, OH, FL) account for the largest volumes of fraudulent card transactions."
        },
        {
            "id": "09",
            "file": "09_fraud_by_age.png",
            "title": "EDA 9: Customer Age Distribution",
            "insight": "Cardholders aged 25 to 55 represent the highest absolute volume of fraud, tracking standard workforce credit card usage."
        },
        {
            "id": "10",
            "file": "10_roc_curves_comparison.png",
            "title": "Model Comparison: ROC Curves (AUC = 0.995 vs 0.901)",
            "insight": "Random Forest demonstrates near-perfect discrimination (ROC-AUC 0.995) outperforming linear Logistic Regression."
        },
        {
            "id": "11",
            "file": "11_confusion_matrices.png",
            "title": "Confusion Matrices on Held-out Test Set",
            "insight": "Random Forest achieved 83.4% recall on unseen test data while maintaining extremely low false alarms (190 false positives out of 49,807 genuine transactions)."
        }
    ]
    return render_template("eda.html", graphs=graphs)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_result = None

    if request.method == "POST":
        try:
            amt = float(request.form.get("amt", 0))
            category = request.form.get("category", "grocery_pos")
            gender = request.form.get("gender", "M")
            city_pop = float(request.form.get("city_pop", 10000))
            transaction_hour = int(request.form.get("transaction_hour", 12))
            day_of_week = int(request.form.get("day_of_week", 0))
            is_weekend = 1 if day_of_week >= 5 else 0
            customer_age = float(request.form.get("customer_age", 35))
            merchant_distance = float(request.form.get("merchant_distance", 50))

            # Assemble DataFrame matching training features
            input_df = pd.DataFrame([{
                "amt": amt,
                "category": category,
                "gender": gender,
                "city_pop": city_pop,
                "transaction_hour": transaction_hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "customer_age": customer_age,
                "merchant_distance": merchant_distance
            }])

            if model is not None:
                prob = float(model.predict_proba(input_df)[0][1])
                is_fraud = bool(prob >= 0.50)
            else:
                prob = 0.0
                is_fraud = False

            # Risk factors
            risk_factors = []
            if amt > 500:
                risk_factors.append(f"High transaction amount (${amt:,.2f})")
            if 22 <= transaction_hour or transaction_hour < 4:
                risk_factors.append(f"High-risk late night transaction hour ({transaction_hour:02d}:00)")
            if category in ["grocery_pos", "shopping_net", "misc_net"]:
                risk_factors.append(f"Frequent fraud merchant category ({category})")
            if merchant_distance > 100:
                risk_factors.append(f"Unusually large merchant distance ({merchant_distance:.1f} km)")

            prediction_result = {
                "is_fraud": is_fraud,
                "probability": round(prob * 100, 2),
                "verdict": "POTENTIAL FRAUD (High Risk)" if is_fraud else "GENUINE TRANSACTION (Low Risk)",
                "risk_factors": risk_factors,
                "inputs": {
                    "amt": amt,
                    "category": category,
                    "gender": gender,
                    "transaction_hour": transaction_hour,
                    "day_of_week": day_of_week,
                    "customer_age": customer_age,
                    "merchant_distance": merchant_distance,
                    "city_pop": city_pop
                }
            }
        except Exception as e:
            prediction_result = {"error": str(e)}

    return render_template("predict.html", categories=CATEGORIES, result=prediction_result)


@app.route("/patterns")
def patterns():
    rules_data = []
    if os.path.exists(RULES_PATH):
        df_rules = pd.read_csv(RULES_PATH)
        rules_data = df_rules.head(20).to_dict(orient="records")

    benchmark = {
        "apriori_time": "0.049s",
        "fpgrowth_time": "0.191s",
        "total_rules": 366,
        "fraud_rules": 50,
        "max_lift": "8.76"
    }
    return render_template("patterns.html", rules=rules_data, benchmark=benchmark)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        prob = float(model.predict_proba(input_df)[0][1])
        return jsonify({
            "is_fraud": bool(prob >= 0.50),
            "fraud_probability": round(prob, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
