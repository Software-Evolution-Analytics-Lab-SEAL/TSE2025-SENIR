import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# File Paths
FEATURES_FILE = "data/processed/features_dataset.csv"
OUTPUT_FILE = "data/results/rq2_feature_importance.json"
SHAP_PLOT_FILE = "data/results/rq2_shap_feature_importance.png"
CORRELATION_FILE = "data/results/rq2_correlation_analysis.json"

# Load Feature Dataset
def load_data(input_path):
    """Loads the extracted feature dataset."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: {input_path} not found!")

    return pd.read_csv(input_path)

# Train Logistic Regression Model for Feature Importance
def train_logistic_regression(X, y):
    """Trains a logistic regression model for feature importance analysis."""
    print("[INFO] Training Logistic Regression Model...")

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X, y)

    feature_importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    return model, feature_importance

# Train Mixed-Effects Model for Feature Significance
def train_mixed_effects_model(df):
    """Trains a Mixed-Effects Logistic Regression Model."""
    print("[INFO] Training Mixed-Effects Model...")

    formula = "resolved ~ " + " + ".join([col for col in df.columns if col not in ["resolved", "conversation_id"]])
    model = smf.mixedlm(formula, df, groups=df["conversation_id"], family=sm.families.Binomial())
    result = model.fit()

    return result

# SHAP Feature Importance Analysis
def shap_feature_analysis(model, X):
    """Performs SHAP analysis for feature importance."""
    print("[INFO] Running SHAP analysis...")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Generate SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(SHAP_PLOT_FILE, bbox_inches="tight")
    print(f"[INFO] SHAP feature importance plot saved: {SHAP_PLOT_FILE}")

    return shap_values

# Feature Group Analysis
def analyze_feature_groups(df):
    """Analyzes the importance of different feature groups."""
    feature_groups = {
        "text-based": ["text_code_ratio", "readability_cli", "url_count"],
        "user-based": ["avg_sentiment", "active_questioner", "questioner_response_ratio"],
        "conversation-based": ["num_messages", "num_participants", "duration"]
    }

    results = {}

    for group, features in feature_groups.items():
        available_features = [f for f in features if f in df.columns]
        if available_features:
            X = df[available_features]
            y = df["resolved"]

            model, importance = train_logistic_regression(X, y)
            results[group] = importance.to_dict()

    return results

# Correlation Analysis
def correlation_analysis(df):
    """Computes Spearman & Pearson correlation between features and resolution status."""
    correlations = {}

    for col in df.columns:
        if col not in ["resolved", "conversation_id"]:
            spearman_corr, _ = spearmanr(df[col], df["resolved"])
            pearson_corr, _ = pearsonr(df[col], df["resolved"])
            correlations[col] = {"Spearman": spearman_corr, "Pearson": pearson_corr}

    return correlations

# Save Results
def save_results(results, output_path):
    """Saves the analysis results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Results saved to {output_path}")

# Main Execution
def main():
    print("[INFO] Loading feature dataset...")
    df = load_data(FEATURES_FILE)

    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ["resolved", "conversation_id"]]
    X = df[feature_cols]
    y = df["resolved"]

    # Train Logistic Regression Model
    model, feature_importance = train_logistic_regression(X, y)

    # Train Mixed-Effects Model
    mixed_model = train_mixed_effects_model(df)

    # SHAP Analysis
    shap_values = shap_feature_analysis(model, X)

    # Feature Group Analysis
    feature_group_results = analyze_feature_groups(df)

    # Correlation Analysis
    correlation_results = correlation_analysis(df)

    # Save Results
    results = {
        "Logistic Regression Importance": feature_importance.to_dict(),
        "Mixed-Effects Model Summary": str(mixed_model.summary()),
        "Feature Group Importance": feature_group_results,
        "Correlation Analysis": correlation_results
    }
    save_results(results, OUTPUT_FILE)

    print("[SUCCESS] RQ2 analysis completed!")

if __name__ == "__main__":
    main()
