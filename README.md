# TSE2025-SENIR Replication Package
Welcome to the **SENIR** (SoftwarE-specific Named entity recognition, Intent detection, and Resolution classification) replication package. This repository accompanies the paper:

> **"Towards Refining Developer Questions using LLM-Based Named Entity Recognition for Developer Chatroom Conversations"**  
> *Submitted to IEEE Transactions on Software Engineering (TSE), 2025*

**Authors**:  
- Pouya Fathollahzadeh, Student Member, IEEE  
- Mariam El Mezouar , Fellow, IEEE  
- Hao Li , Fellow, IEEE  
- Ying Zou , Fellow, IEEE  
- Ahmed E. Hassan, Fellow, IEEE  

## Overview
This replication package contains:
- **Scripts** to process raw chat logs, label them with SENIR, extract features, train models, and run research-question analyses (RQ1–RQ3).
- **Dataset** placeholders for raw and processed data (shared under `data/`).
- **Instructions** for reproducing all results and figures from the paper.

## Repository Structure

```
SENIR-Replication-Package/
├── scripts/
├── data/
│   ├── raw/               
│   └── processed/
├── README.md
└—— requrements.txt
```

## Prerequisites

- **Python 3.8+**  
- **Pip or Conda** for installing dependencies  
- **Git LFS** if hosting large Mistral model checkpoints  

## Installation

1. **Clone** or **download** this repository:
   ```bash
   git clone https://github.com/<your-username>/SENIR-Replication-Package.git
   ```
2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   or  
   ```bash
   conda create -n senir python=3.8
   conda activate senir
   pip install -r requirements.txt
   ```

## Usage

Below is the typical pipeline to replicate results:

### 1. Data Collection & Preprocessing

```
python scripts/0_data_collection_and_preprocessing.py
```
- Gathers raw chat logs (e.g., DISCO dataset)  
- Outputs preprocessed JSON data into `data/processed/`

### 2. SENIR Labeling

```
python scripts/1_labeling_senir.py
```
- Uses **Mixtral-8x7B** to label each conversation with software entities, intent categories, and resolution status  
- Produces `labeled_dataset.json` in `data/processed/`

### 3. Feature Extraction

```
python scripts/2_feature_extraction.py
```
- Extracts textual, user-based, and entity-based features from the labeled dataset  
- Saves `features_dataset.csv` into `data/processed/`

### 4. Model Training

```
python scripts/3_model_training.py
```
- Trains **Mixed-Effect Logistic Regression** with multiple sampling strategies (undersampling, SMOTE, oversampling)  
- Evaluates performance via Cross-Validation & Bootstrapping  
- Compares **LLM baseline** (prompt-based) vs. classical approach  
- Saves final results under `results/`

### 5. RQ Analyses

**RQ1**: Evaluate SENIR labeling quality
```
python scripts/RQ1_analysis.py
```
- Compares SENIR predictions with manually annotated “golden” data
- Outputs metrics (Accuracy, Precision, Recall, F1, Cohen's Kappa)

**RQ2**: Identify influential features for resolution
```
python scripts/RQ2_analysis.py
```
- Logistic & Mixed-Effect models, plus SHAP / correlation analysis to find top predictive features

**RQ3**: Examine entity-intent interactions
```
python scripts/RQ3_analysis.py
```
- Computes entity-intent co-occurrence & resolution success
- Performs Chi-Square analysis & visualizations (heatmap, bar plots)

## Results

- **`results/rq1_evaluation_results.json`**: SENIR vs. manual labels  
- **`results/rq2_feature_importance.json`**: Feature importance from logistic & mixed-effect models  
- **`results/rq3_entity_intent_resolution_analysis.json`**: How entity-intent pairs affect resolution  

## Citing SENIR

If you use SENIR or build on our replication package, please cite the following:

```
@article{Pouya2025SENIR,
  author = {Fathollahzadeh, Pouya and El Mezouar, Mariam and Li, Hao and Zou, Ying and Hassan, Ahmed E.},
  title = {Towards Refining Developer Questions using LLM-Based Named Entity Recognition for Developer Chatroom Conversations},
  journal = {IEEE Transactions on Software Engineering (Under Submission)},
  year = {2025}
}
```


