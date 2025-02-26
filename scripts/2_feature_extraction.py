import os
import json
import numpy as np
import pandas as pd
import nltk
import textstat
import statistics
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
from urllib.parse import urlparse
from dateutil import parser
from sklearn.preprocessing import MinMaxScaler

# File Paths
INPUT_FILE = "data/processed/labeled_dataset.json"
OUTPUT_FILE = "data/processed/features_dataset.csv"

# Precompiled regex patterns for performance
CODE_BLOCK_REGEX = re.compile(r'```(.*?)```', re.DOTALL)
URL_PATTERN = re.compile(r'https?://\S+')

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('vader_lexicon')

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

# User Statistics
def compute_user_statistics(df):
    """
    Computes user-based statistics:
    - Total messages per user
    - Questions asked by each user
    - Questions receiving responses
    - Active chatroom participants
    """
    user_message_count = Counter()
    user_questions_with_responses = Counter()
    user_total_questions = Counter()

    for _, conversation in df.iterrows():
        users = [msg["user"] for msg in conversation["messages"]]
        user_message_count.update(users)

        first_user = users[0]
        user_total_questions[first_user] += 1
        if len(users) > 1:  # If at least one response exists
            user_questions_with_responses[first_user] += 1

    total_messages = sum(user_message_count.values())
    top_30_percent_threshold = sorted(user_message_count.values(), reverse=True)[int(0.3 * len(user_message_count)) - 1]

    return user_message_count, user_questions_with_responses, user_total_questions, top_30_percent_threshold

# Compute Feature Functions
def compute_code_text_ratio(texts):
    """Computes the ratio of code snippets to total text length."""
    total_text_length = sum(len(text) for text in texts)
    code_lengths = [len(code) for text in texts for code in CODE_BLOCK_REGEX.findall(text)]
    total_code_length = sum(code_lengths)
    return total_code_length / total_text_length if total_text_length > 0 else 0

def count_urls(texts):
    """Counts the number of URLs in the conversation."""
    return sum(len(URL_PATTERN.findall(text)) for text in texts)

def compute_sentiment_analysis(texts):
    """Computes average sentiment score of the conversation."""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(text)["compound"] for text in texts]
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Feature Extraction
def extract_features(df, user_message_count, user_questions_with_responses, user_total_questions, top_30_percent_threshold):
    """
    Extracts all required features, including:
    - Text-based features
    - User statistics
    - Sentiment analysis
    - Temporal features
    """
    feature_list = []

    for _, row in df.iterrows():
        try:
            timestamps = [parser.parse(msg["ts"]) for msg in row["messages"]]
            texts = [msg["text"] for msg in row["messages"]]
            users = [msg["user"] for msg in row["messages"]]
        except Exception as e:
            print(f"Error parsing data: {e}")
            continue

        first_user = users[0]
        question_end_index = len(texts)
        for i in range(1, len(users)):
            if users[i] != first_user:
                question_end_index = i
                break

        question_length = sum(len(texts[i]) for i in range(question_end_index))
        conversation_length = sum(len(text) for text in texts)

        question_response_time = None
        if question_end_index < len(timestamps):  # Ensure response exists
            last_question_time = timestamps[question_end_index - 1]
            first_response_time = timestamps[question_end_index]
            question_response_time = (first_response_time - last_question_time).total_seconds()

        messages_count = user_message_count[first_user]
        is_active_questioner = 1 if messages_count > top_30_percent_threshold else 0
        received_response_ratio = (user_questions_with_responses[first_user] / user_total_questions[first_user]) if user_total_questions[first_user] > 0 else 0

        start_time = min(timestamps)
        end_time = max(timestamps)

        # Additional features
        average_sentiment = compute_sentiment_analysis(texts)
        text_code_ratio = compute_code_text_ratio(texts)
        url_count = count_urls(texts)
        readability = textstat.coleman_liau_index(" ".join(texts))

        # Temporal & engagement features
        feature_list.append({
            "conversation_id": row["conversation_id"],
            "start_time": start_time.timestamp(),
            "end_time": end_time.timestamp(),
            "duration": (end_time - start_time).total_seconds(),
            "num_messages": len(texts),
            "num_participants": len(set(users)),
            "avg_message_length": sum(len(text) for text in texts) / len(texts) if texts else 0,
            "readability_cli": readability,
            "text_code_ratio": text_code_ratio,
            "url_count": url_count,
            "weekday": start_time.weekday(),
            "hour_of_day": start_time.hour,
            "avg_sentiment": average_sentiment,
            "question_length": question_length,
            "conversation_length": conversation_length,
            "question_response_time": question_response_time,
            "active_questioner": is_active_questioner,
        })

    return pd.DataFrame(feature_list)

# Normalize and Save Features
def normalize_features(df):
    """Normalizes numerical features using Min-Max Scaling."""
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def save_features(df, output_path):
    """Saves the extracted feature dataset as a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"[INFO] Features saved to {output_path}")

# Main Execution
def main():
    print("[INFO] Loading labeled dataset...")
    df = load_data(INPUT_FILE)

    print("[INFO] Computing user statistics...")
    user_stats = compute_user_statistics(df)

    print("[INFO] Extracting features...")
    df_features = extract_features(df, *user_stats)

    print("[INFO] Normalizing and saving features...")
    df_features = normalize_features(df_features)
    save_features(df_features, OUTPUT_FILE)

    print("[SUCCESS] Feature extraction completed!")

if __name__ == "__main__":
    main()
