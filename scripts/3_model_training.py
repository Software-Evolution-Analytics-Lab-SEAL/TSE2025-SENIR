import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import statsmodels.formula.api as smf
import statsmodels.api as sm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configurations
FEATURES_FILE = "data/processed/features_dataset.csv"
RESULTS_OUTPUT = "data/results/rq2_model_results.json"
SEED = 42

# Mistral Model Setup
mistral_checkpoint = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load the tokenizer and model
print(f"[INFO] Loading Mistral model from: {mistral_checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(mistral_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# Enable quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(mistral_checkpoint, quantization_config=bnb_config, device_map="auto")
model.eval()

# Load Data
def load_features(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    return pd.read_csv(path)

# LLM-Based Classification
def llm_predict_resolution(question_text):
    """Uses Mistral to classify if a question is 'solved' or 'unsolved'."""
    prompt = f"""You are an AI assistant who helps analyze software engineering chatroom discussions. Your task is to determine whether the following developer question will likely receive a resolved answer based on its clarity, completeness, and technical details.
    Instructions:
        - Read the developer question carefully.
        - Predict whether the question will be resolved (Yes) or not resolved (No).
        - Provide a confidence score between 0% and 100%, indicating how sure you are about your prediction.
        - Do not provide an explanation—only return the classification and confidence score.
    ####
    Input Format:
    Question: [INSERT DEVELOPER QUESTION HERE]

    Output Format (Strict JSON format):
    [
    "resolution_status": "Yes" or "No",
    "confidence_score": "XX%"
    ]
    ###
    Example Execution:
    Input:
    Question: [I am using TensorFlow version 2.5 and encountering an error during installation. How can I fix this?]

    Expected LLM Output:
    [
    "resolution_status": "Yes",
    "confidence_score": "82%"
    ]
    #####:\n\n{question_text}\n\nAnswer:"
    """
        
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=10)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return 1 if "Yes" in response else 0

def evaluate_llm_baseline(df):
    """Applies Mistral-based resolution classification to the dataset."""
    y_true = df["resolved"].values
    y_pred = [llm_predict_resolution(row["question"]) for _, row in df.iterrows()]
    
    auc = roc_auc_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return auc, recall

# Mixed-Effects Logistic Regression
def train_mixed_effects_model(df, feature_list):
    """Trains a mixed-effect logistic regression model."""
    formula = f"resolved ~ {' + '.join(feature_list)} + (1|conversation_id)"
    model = smf.mixedlm(formula, df, groups=df["conversation_id"], family=sm.families.Binomial()).fit()
    return model

# Cross-Validation & Bootstrapping
def cross_validate_mixed_effect(df, feature_list, n_splits=5):
    """Cross-validation for mixed-effect models."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    auc_scores = []

    for train_idx, test_idx in skf.split(df[feature_list], df["resolved"]):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        model = train_mixed_effects_model(train_df, feature_list)
        y_prob = model.predict(test_df)
        auc = roc_auc_score(test_df["resolved"], y_prob)
        auc_scores.append(auc)

    return np.mean(auc_scores), np.std(auc_scores)

# Handling Class Imbalance (Oversampling & Undersampling)
def apply_sampling_strategy(df, strategy="undersample"):
    """Applies Random Undersampling or SMOTE Oversampling."""
    feature_cols = [c for c in df.columns if c not in ["resolved", "conversation_id"]]
    X, y = df[feature_cols], df["resolved"]

    if strategy == "undersample":
        rus = RandomUnderSampler(random_state=SEED)
        X_res, y_res = rus.fit_resample(X, y)
    elif strategy == "smote":
        sm = SMOTE(random_state=SEED)
        X_res, y_res = sm.fit_resample(X, y)
    elif strategy == "oversample":
        X_res, y_res = resample(X, y, replace=True, random_state=SEED)
    else:
        return df

    res_df = pd.DataFrame(X_res, columns=feature_cols)
    res_df["resolved"] = y_res
    res_df["conversation_id"] = -1  
    return res_df

# Main Execution
def main():
    os.makedirs(os.path.dirname(RESULTS_OUTPUT), exist_ok=True)

    print("[INFO] Loading feature dataset...")
    df = load_features(FEATURES_FILE)

    print("[INFO] Evaluating LLM Baseline on full dataset...")
    auc_llm, recall_llm = evaluate_llm_baseline(df)

    # Feature Selection
    candidate_features = [c for c in df.columns if c not in ["resolved", "conversation_id", "question"]]

    results = {"LLM Baseline AUC": auc_llm, "LLM Baseline Recall": recall_llm, "Sampling_Results": {}}

    for strategy in ["undersample", "smote", "oversample"]:
        df_sampled = apply_sampling_strategy(df, strategy)
        auc_mean, auc_std = cross_validate_mixed_effect(df_sampled, candidate_features, n_splits=5)
        results["Sampling_Results"][strategy] = {"MeanAUC": auc_mean, "StdAUC": auc_std}

    with open(RESULTS_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print("[SUCCESS] Model training completed!")

if __name__ == "__main__":
    main()
