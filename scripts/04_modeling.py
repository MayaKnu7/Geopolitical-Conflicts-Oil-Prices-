# 04_modeling.py
# Geopolitical Conflict and Oil Stock Performance
# Group 26 - Sana Shah, Maya Knutsvig, Illina Islam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import os
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import pickle

load_dotenv()

# ── Configuration ──────────────────────────────────────────
os.makedirs("reports/modeling", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Database Connection ─────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(
        host        = os.getenv("DB_HOST"),
        dbname      = os.getenv("DB_NAME"),
        user        = os.getenv("DB_USER"),
        password    = os.getenv("DB_PASSWORD"),
        sslmode     = "verify-full",
        sslrootcert = "./global-bundle.pem"
    )

# ── Load Data ───────────────────────────────────────────────
def load_data():
    print("Loading data from database...")
    conn     = get_db_connection()
    stock_df = pd.read_sql("SELECT * FROM stock_data", conn)
    oil_df   = pd.read_sql("SELECT * FROM oil_prices", conn)
    conn.close()

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    oil_df["date"]   = pd.to_datetime(oil_df["date"])

    return stock_df, oil_df

# ── Feature Engineering ─────────────────────────────────────
def engineer_features(stock_df, oil_df):
    print("\n── Engineering features ──")

    stock_df = stock_df.sort_values(["ticker", "conflict_period", "date"])

    # Daily return (percentage change)
    stock_df["daily_return"] = stock_df.groupby(
        ["ticker", "conflict_period"])["close"].pct_change() * 100

    # Rolling volatility of returns
    stock_df["volatility_10d"] = stock_df.groupby(
        ["ticker", "conflict_period"])["daily_return"].transform(
        lambda x: x.rolling(10, min_periods=1).std())

    stock_df["volatility_30d"] = stock_df.groupby(
        ["ticker", "conflict_period"])["daily_return"].transform(
        lambda x: x.rolling(30, min_periods=1).std())

    # Moving averages of RETURNS not prices
    stock_df["moving_avg_return_20"] = stock_df.groupby(
        ["ticker", "conflict_period"])["daily_return"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())

    stock_df["moving_avg_return_50"] = stock_df.groupby(
        ["ticker", "conflict_period"])["daily_return"].transform(
        lambda x: x.rolling(50, min_periods=1).mean())

    # RSI (Relative Strength Index)
    def compute_rsi(series, window=14):
        delta    = series.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi      = 100 - (100 / (1 + rs))
        return rsi

    stock_df["rsi"] = stock_df.groupby(
        ["ticker", "conflict_period"])["close"].transform(compute_rsi)

    # Merge oil prices — use pct_change only, not raw price
    oil_df_slim = oil_df[["date", "pct_change"]].rename(
        columns={"pct_change": "oil_pct_change"})
    stock_df = stock_df.merge(oil_df_slim, on="date", how="left")

    # Days since conflict start
    conflict_starts = {
        "First Gulf War":  pd.Timestamp("1990-08-02"),
        "Second Gulf War": pd.Timestamp("2003-03-20"),
        "US-Iran":         pd.Timestamp("2020-01-03")
    }
    stock_df["days_since_event"] = stock_df.apply(
        lambda row: max(0, (row["date"] - conflict_starts.get(
            row["conflict_period"], row["date"])).days),
        axis=1
    )

    # Target variable: 1 if price went up, 0 if down
    stock_df["target"] = (stock_df["daily_return"] > 0).astype(int)

    # Feature columns — return based only, no raw prices
    feature_cols = [
        "volatility_10d",
        "volatility_30d",
        "moving_avg_return_20",
        "moving_avg_return_50",
        "rsi",
        "oil_pct_change",
        "days_since_event"
    ]

    stock_df = stock_df.dropna(subset=feature_cols + ["target"])

    print(f"  Features engineered: {len(stock_df)} rows")
    return stock_df, feature_cols

# ── Save Model Results to DB ────────────────────────────────
def save_model_results(model_name, ticker, conflict_period,
                       accuracy, precision, recall, f1, notes=""):
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_results
            (run_date, model_name, ticker, conflict_period,
             accuracy, precision_score, recall, f1_score, notes)
            VALUES (CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (model_name, ticker, conflict_period,
              accuracy, precision, recall, f1, notes))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"  Warning: Could not save to DB: {e}")

# ── Train and Evaluate ──────────────────────────────────────
def train_evaluate(X_train, X_test, y_train, y_test,
                   model, model_name, label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  {model_name} Results ({label}):")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, accuracy, precision, recall, f1, y_pred

# ── Plot Confusion Matrix ───────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name, label):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix\n{model_name} - {label}")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Down", "Up"])
    plt.yticks([0, 1], ["Down", "Up"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center", color="black")
    plt.tight_layout()
    filename = f"reports/modeling/confusion_{model_name.replace(' ', '_')}_{label.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

# ── Cross Validation ────────────────────────────────────────
def cross_validate_by_period(stock_df, feature_cols):
    print("\n── Cross Validation by Conflict Period ──")

    periods = stock_df["conflict_period"].unique()
    results = []

    for test_period in periods:
        train_df = stock_df[stock_df["conflict_period"] != test_period]
        test_df  = stock_df[stock_df["conflict_period"] == test_period]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols])
        X_test  = scaler.transform(test_df[feature_cols])
        y_train = train_df["target"]
        y_test  = test_df["target"]

        for model_name, model in [
            ("Logistic Regression", LogisticRegression(
                max_iter=1000, random_state=42)),
            ("Random Forest", RandomForestClassifier(
                n_estimators=100, max_depth=5,
                min_samples_leaf=50, max_features="sqrt",
                random_state=42, n_jobs=-1))
        ]:
            model.fit(X_train, y_train)
            y_pred    = model.predict(X_test)
            accuracy  = accuracy_score(y_test, y_pred)
            f1        = f1_score(y_test, y_pred, zero_division=0)

            print(f"  Test: {test_period} | {model_name} | "
                  f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

            results.append({
                "test_period": test_period,
                "model":       model_name,
                "accuracy":    accuracy,
                "f1":          f1
            })

    results_df = pd.DataFrame(results)

    # Plot cross validation results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, metric in enumerate(["accuracy", "f1"]):
        pivot = results_df.pivot(
            index="test_period", columns="model", values=metric)
        pivot.plot(kind="bar", ax=axes[i], rot=15)
        axes[i].set_title(f"Cross Validation - {metric.capitalize()}")
        axes[i].set_xlabel("Test Period")
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_ylim(0, 1)
        axes[i].legend(title="Model")
    plt.tight_layout()
    plt.savefig("reports/modeling/cross_validation_results.png")
    plt.close()
    print("  Saved: cross_validation_results.png")

    return results_df

# ── Main Modeling Pipeline ──────────────────────────────────
def run_modeling(stock_df, feature_cols):
    print("\n── Running models ──")

    scaler  = StandardScaler()
    results = []

    # ── Experiment 1: Train on First Gulf War → Test on Second Gulf War
    print("\n" + "="*50)
    print("EXPERIMENT 1: Train on First Gulf War → Test on Second Gulf War")
    print("="*50)

    train_df = stock_df[stock_df["conflict_period"] == "First Gulf War"]
    test_df  = stock_df[stock_df["conflict_period"] == "Second Gulf War"]

    if len(train_df) > 0 and len(test_df) > 0:
        X_train = scaler.fit_transform(train_df[feature_cols])
        X_test  = scaler.transform(test_df[feature_cols])
        y_train = train_df["target"]
        y_test  = test_df["target"]

        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model, acc, prec, rec, f1, y_pred = train_evaluate(
            X_train, X_test, y_train, y_test,
            lr_model, "Logistic Regression", "Gulf War 1→2")
        plot_confusion_matrix(y_test, y_pred,
                              "Logistic Regression", "Gulf_War_1to2")
        save_model_results("Logistic Regression", "ALL",
                           "First→Second Gulf War",
                           acc, prec, rec, f1, "Train:GW1 Test:GW2")
        results.append({"model": "Logistic Regression",
                        "experiment": "GW1→GW2",
                        "accuracy": acc, "f1": f1})

        # Random Forest with constraints
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
        rf_model, acc, prec, rec, f1, y_pred = train_evaluate(
            X_train, X_test, y_train, y_test,
            rf_model, "Random Forest", "Gulf War 1→2")
        plot_confusion_matrix(y_test, y_pred,
                              "Random Forest", "Gulf_War_1to2")
        save_model_results("Random Forest", "ALL",
                           "First→Second Gulf War",
                           acc, prec, rec, f1, "Train:GW1 Test:GW2")
        results.append({"model": "Random Forest",
                        "experiment": "GW1→GW2",
                        "accuracy": acc, "f1": f1})

        # Save models
        with open("models/logistic_regression_gw.pkl", "wb") as f:
            pickle.dump(lr_model, f)
        with open("models/random_forest_gw.pkl", "wb") as f:
            pickle.dump(rf_model, f)
        with open("models/scaler_gw.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("\n  Models saved to models/")

    # ── Experiment 2: Train on Gulf Wars → Test on US-Iran
    print("\n" + "="*50)
    print("EXPERIMENT 2: Train on Gulf Wars → Test on US-Iran")
    print("="*50)

    train_df2 = stock_df[stock_df["conflict_period"].isin(
        ["First Gulf War", "Second Gulf War"])]
    test_df2  = stock_df[stock_df["conflict_period"] == "US-Iran"]

    rf_model2 = None

    if len(train_df2) > 0 and len(test_df2) > 0:
        X_train2 = scaler.fit_transform(train_df2[feature_cols])
        X_test2  = scaler.transform(test_df2[feature_cols])
        y_train2 = train_df2["target"]
        y_test2  = test_df2["target"]

        # Logistic Regression
        lr_model2 = LogisticRegression(max_iter=1000, random_state=42)
        lr_model2, acc, prec, rec, f1, y_pred = train_evaluate(
            X_train2, X_test2, y_train2, y_test2,
            lr_model2, "Logistic Regression", "Gulf Wars→US-Iran")
        plot_confusion_matrix(y_test2, y_pred,
                              "Logistic Regression", "GulfWars_to_USIran")
        save_model_results("Logistic Regression", "ALL",
                           "Gulf Wars→US-Iran",
                           acc, prec, rec, f1,
                           "Train:GW1+GW2 Test:US-Iran")
        results.append({"model": "Logistic Regression",
                        "experiment": "GW→US-Iran",
                        "accuracy": acc, "f1": f1})

        # Random Forest with constraints
        rf_model2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
        rf_model2, acc, prec, rec, f1, y_pred = train_evaluate(
            X_train2, X_test2, y_train2, y_test2,
            rf_model2, "Random Forest", "Gulf Wars→US-Iran")
        plot_confusion_matrix(y_test2, y_pred,
                              "Random Forest", "GulfWars_to_USIran")
        save_model_results("Random Forest", "ALL",
                           "Gulf Wars→US-Iran",
                           acc, prec, rec, f1,
                           "Train:GW1+GW2 Test:US-Iran")
        results.append({"model": "Random Forest",
                        "experiment": "GW→US-Iran",
                        "accuracy": acc, "f1": f1})

        # Save models
        with open("models/logistic_regression_iran.pkl", "wb") as f:
            pickle.dump(lr_model2, f)
        with open("models/random_forest_iran.pkl", "wb") as f:
            pickle.dump(rf_model2, f)
        with open("models/scaler_iran.pkl", "wb") as f:
            pickle.dump(scaler, f)

    # ── Feature Importance Plot ──────────────────────────────
    print("\n── Feature Importance (Random Forest) ──")
    importances = pd.Series(
        rf_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importances.plot(kind="bar", color="steelblue")
    plt.title("Feature Importance - Random Forest (Gulf War 1→2)")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("reports/modeling/feature_importance.png")
    plt.close()
    print("  Saved: feature_importance.png")

    # ── Model Comparison Chart ───────────────────────────────
    results_df = pd.DataFrame(results)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 6))

    for i, metric in enumerate(["accuracy", "f1"]):
        pivot = results_df.pivot(
            index="experiment", columns="model", values=metric)
        pivot.plot(kind="bar", ax=axes[i], rot=0)
        axes[i].set_title(f"Model Comparison - {metric.capitalize()}")
        axes[i].set_xlabel("Experiment")
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend(title="Model")
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("reports/modeling/model_comparison.png")
    plt.close()
    print("  Saved: model_comparison.png")

    return rf_model, rf_model2, scaler

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Starting modeling...")
    print("=" * 50)

    stock_df, oil_df       = load_data()
    stock_df, feature_cols = engineer_features(stock_df, oil_df)

    # Cross validation first
    cv_results = cross_validate_by_period(stock_df, feature_cols)

    # Main experiments
    rf_model, rf_model2, scaler = run_modeling(stock_df, feature_cols)

    print("\n" + "=" * 50)
    print("Modeling complete!")
    print("Charts saved to reports/modeling/")
    print("Models saved to models/")
    print("=" * 50)