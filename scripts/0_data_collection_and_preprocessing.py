import os
import json
import pandas as pd
import re

def load_raw_data(input_dir: str) -> pd.DataFrame:
    """Loads raw JSON files from the specified directory and combines them into a DataFrame."""
    all_data = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    all_data.extend(data)
                except json.JSONDecodeError:
                    print(f"[WARNING] Skipping corrupted file: {filename}")
    return pd.DataFrame(all_data)

def merge_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Groups messages by conversation ID and merges them into a single conversation block."""
    df = df.sort_values(by=["conversation_id", "timestamp"])
    merged_df = df.groupby("conversation_id")["message"].apply(lambda x: " ".join(x)).reset_index()
    return merged_df

def filter_one_word_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Removes messages that contain only a single word."""
    df["word_count"] = df["message"].apply(lambda x: len(re.findall(r"\w+", x)))
    df = df[df["word_count"] > 1]
    df.drop(columns=["word_count"], inplace=True)
    return df

def main():
    input_dir = os.path.join("data", "raw")
    output_file = os.path.join("data", "processed", "preprocessed_dataset.json")
    
    print("[INFO] Loading raw data...")
    df_raw = load_raw_data(input_dir)
    print(f"[INFO] Loaded {len(df_raw)} raw messages.")
    
    print("[INFO] Merging messages by conversation...")
    df_merged = merge_messages(df_raw)
    print(f"[INFO] Merged into {len(df_merged)} conversations.")
    
    print("[INFO] Filtering one-word messages...")
    df_cleaned = filter_one_word_messages(df_merged)
    print(f"[INFO] Filtered dataset contains {len(df_cleaned)} conversations.")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_cleaned.to_json(output_file, orient="records", lines=True)
    print(f"[INFO] Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    main()
