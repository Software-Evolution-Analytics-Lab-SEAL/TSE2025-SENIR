import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score

# File Paths
PREDICTED_FILE = "data/processed/labeled_dataset.json"
GOLDEN_FILES = [
    "data/processed/clojurians_golden_annotations.csv",
    "data/processed/golang_golden_annotations.csv",
    "data/processed/pythongeneral_golden_annotations.csv",
    "data/processed/racketgeneral_golden_annotations.csv"
]
OUTPUT_FILE = "data/results/rq1_evaluation_results.json"

# Load Labeled Datasets
def load_jsonl_file(file_path):
    """Loads a JSONL file into a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found!")

    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return pd.DataFrame(data)

def load_and_merge_golden_files(file_paths):
    """Loads multiple golden annotation CSV files and merges them."""
    df_list = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["dataset"] = os.path.basename(file_path).replace("_golden_annotations.csv", "")
            df_list.append(df)
        else:
            print(f"[WARNING] Golden annotation file not found: {file_path}")

    return pd.concat(df_list, ignore_index=True) if df_list else None

# Compute Performance Metrics
def compute_metrics(y_true, y_pred):
    """Computes Accuracy, Precision, Recall, and F1-score."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=1),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=1),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=1),
        "Cohen’s Kappa": cohen_kappa_score(y_true, y_pred)
    }

# Named Entity Recognition (NER)
def evaluate_ner(pred_df, gold_df):
    """Evaluates NER using entity matches."""
    gold_entities = gold_df.set_index("conversation_id")["entities"].apply(eval).to_dict()
    pred_entities = pred_df.set_index("conversation_id")["entities"].to_dict()

    y_true, y_pred = [], []
    for conv_id in gold_entities.keys():
        true_set = set(gold_entities[conv_id])
        pred_set = set(pred_entities.get(conv_id, []))

        y_true.append(len(true_set))
        y_pred.append(len(pred_set.intersection(true_set)))

    return compute_metrics(y_true, y_pred)

# Intent Detection
def evaluate_intent(pred_df, gold_df):
    """Evaluates Intent Classification."""
    gold_intents = gold_df.set_index("conversation_id")["intent"].to_dict()
    pred_intents = pred_df.set_index("conversation_id")["intent"].to_dict()

    y_true = [gold_intents[conv_id] for conv_id in gold_intents.keys()]
    y_pred = [pred_intents.get(conv_id, "UNKNOWN") for conv_id in gold_intents.keys()]

    return compute_metrics(y_true, y_pred)

# Resolution Classification
def evaluate_resolution(pred_df, gold_df):
    """Evaluates Resolution Classification."""
    gold_res = gold_df.set_index("conversation_id")["resolution_status"].map(lambda x: 1 if x == "SOLVED" else 0).to_dict()
    pred_res = pred_df.set_index("conversation_id")["resolution_status"].map(lambda x: 1 if x == "SOLVED" else 0).to_dict()

    y_true = list(gold_res.values())
    y_pred = [pred_res.get(conv_id, 0) for conv_id in gold_res.keys()]

    return compute_metrics(y_true, y_pred)

# Entity & Intent Frequency Analysis
def compute_entity_intent_frequencies(gold_df):
    """Computes most frequently occurring entities and intents."""
    entity_counts = gold_df["entities"].apply(eval).explode().value_counts().to_dict()
    intent_counts = gold_df["intent"].value_counts().to_dict()

    return {"Entity Frequency": entity_counts, "Intent Frequency": intent_counts}

# Save Results
def save_results(results, output_path):
    """Saves the evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Results saved to {output_path}")

# Main Execution
def main():
    print("[INFO] Loading predicted dataset...")
    pred_df = load_jsonl_file(PREDICTED_FILE)

    print("[INFO] Loading and merging golden datasets...")
    gold_df = load_and_merge_golden_files(GOLDEN_FILES)

    results = {
        "NER": evaluate_ner(pred_df, gold_df),
        "Intent Detection": evaluate_intent(pred_df, gold_df),
        "Resolution Classification": evaluate_resolution(pred_df, gold_df),
        "Entity & Intent Frequencies": compute_entity_intent_frequencies(gold_df),
    }

    save_results(results, OUTPUT_FILE)
    print("[✅ SUCCESS] RQ1 analysis completed!")

if __name__ == "__main__":
    main()
