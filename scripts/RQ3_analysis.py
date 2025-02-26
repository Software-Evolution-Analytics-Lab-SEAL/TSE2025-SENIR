import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr

# File Paths
INPUT_FILE = "data/processed/labeled_dataset.json"
OUTPUT_FILE = "data/results/rq3_entity_intent_resolution_analysis.json"
HEATMAP_FILE = "data/results/rq3_entity_intent_cooccurrence_heatmap.png"
BARPLOT_FILE = "data/results/rq3_entity_intent_resolution_barplot.png"
BEST_WORST_FILE = "data/results/rq3_best_worst_entity_intent.json"

# Load Labeled Data
def load_data(input_path):
    """Loads the labeled dataset from JSON format."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: {input_path} not found!")

    data = []
    with open(input_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    return pd.DataFrame(data)

# Analyze Entity-Intent Co-occurrence
def compute_entity_intent_matrix(df):
    """Computes a co-occurrence matrix of entity types and intents."""
    print("[INFO] Computing entity-intent co-occurrence matrix...")

    entity_list = ["PROGRAMMING_LANGUAGE", "LIBRARY", "API", "ERROR_NAME", "FILE_TYPE", "FRAMEWORK"]
    intent_list = ["LEARNING", "DISCREPANCY", "ERRORS", "REVIEW", "CONCEPTUAL", "API USAGE", "API CHANGE"]

    matrix = pd.DataFrame(0, index=entity_list, columns=intent_list)

    for _, row in df.iterrows():
        for entity in row["entities"]:
            if entity in entity_list and row["intent"] in intent_list:
                matrix.loc[entity, row["intent"]] += 1

    return matrix

# Entity-Intent Impact on Resolution
def compute_entity_intent_resolution_correlation(df):
    """Computes the impact of entity-intent combinations on resolution."""
    print("[INFO] Analyzing entity-intent impact on resolution...")

    entity_intent_resolved = {}

    for _, row in df.iterrows():
        for entity in row["entities"]:
            key = f"{entity}_{row['intent']}"
            if key not in entity_intent_resolved:
                entity_intent_resolved[key] = {"SOLVED": 0, "UNSOLVED": 0}

            entity_intent_resolved[key][row["resolution_status"]] += 1

    results = {}
    for key, counts in entity_intent_resolved.items():
        total = counts["SOLVED"] + counts["UNSOLVED"]
        results[key] = {
            "total_occurrences": total,
            "solved_ratio": counts["SOLVED"] / total if total > 0 else 0
        }

    return results

# Identify Best & Worst Entity-Intent Pairs
def get_best_worst_entity_intents(resolution_data):
    """Finds the best and worst entity-intent combinations."""
    sorted_data = sorted(resolution_data.items(), key=lambda x: x[1]["solved_ratio"], reverse=True)
    
    best_pairs = sorted_data[:5]  # Top 5
    worst_pairs = sorted_data[-5:]  # Bottom 5

    return {
        "Best Entity-Intent Pairs": best_pairs,
        "Worst Entity-Intent Pairs": worst_pairs
    }

# Error Analysis for Misclassified Cases
def analyze_misclassified_cases(df):
    """Identifies frequently unresolved entity-intent combinations."""
    unresolved_cases = df[df["resolution_status"] == "UNSOLVED"]
    
    entity_intent_counts = unresolved_cases.groupby(["entities", "intent"]).size().reset_index(name="count")
    entity_intent_counts = entity_intent_counts.sort_values("count", ascending=False).head(5)

    return entity_intent_counts.to_dict(orient="records")

# Save Analysis Results
def save_results(results, output_path):
    """Saves the entity-intent-resolution analysis results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Analysis results saved to {output_path}")

# Main Execution
def main():
    print("[INFO] Loading labeled dataset...")
    df = load_data(INPUT_FILE)

    print("[INFO] Computing entity-intent co-occurrence...")
    entity_intent_matrix = compute_entity_intent_matrix(df)

    print("[INFO] Computing entity-intent impact on resolution...")
    entity_intent_resolution = compute_entity_intent_resolution_correlation(df)

    print("[INFO] Identifying best and worst entity-intent pairs...")
    best_worst_pairs = get_best_worst_entity_intents(entity_intent_resolution)

    print("[INFO] Performing error analysis for unresolved cases...")
    misclassified_cases = analyze_misclassified_cases(df)

    # Save results
    results = {
        "Entity-Intent Co-occurrence Matrix": entity_intent_matrix.to_dict(),
        "Entity-Intent Resolution Impact": entity_intent_resolution,
        "Best/Worst Entity-Intent Pairs": best_worst_pairs,
        "Most Frequent Unresolved Cases": misclassified_cases
    }
    save_results(results, OUTPUT_FILE)

    print("[SUCCESS] RQ3 analysis completed!")

if __name__ == "__main__":
    main()
