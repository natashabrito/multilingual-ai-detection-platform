import pandas as pd
import json
import os

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "data"

data = []

# -----------------------------
# LOAD LOCAL FILES
# -----------------------------
files = os.listdir(DATASET_PATH)

for file in files:
    path = os.path.join(DATASET_PATH, file)

    # -----------------------------
    # TXT FILES
    # -----------------------------
    if file.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            text = line.strip()
            if not text:
                continue

            # Label logic
            if "Dataset1" in file:
                label = 0   # Human
            else:
                label = 1   # AI

            data.append({"text": text, "label": label})

    # -----------------------------
    # JSONL FILES
    # -----------------------------
    elif file.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                text = obj.get("text", "").strip()
                label = obj.get("label", 1)

                if text:
                    data.append({
                        "text": text,
                        "label": int(label)
                    })

    # -----------------------------
    # CSV FILES
    # -----------------------------
    elif file.endswith(".csv"):
        df_temp = pd.read_csv(path)

        text_col = None
        label_col = None

        for col in df_temp.columns:
            if col.lower() in ["text", "content", "sentence"]:
                text_col = col
            if col.lower() in ["label", "target"]:
                label_col = col

        if text_col:
            for _, row in df_temp.iterrows():
                text = str(row[text_col]).strip()
                if not text:
                    continue

                if label_col:
                    label = int(row[label_col])
                else:
                    label = 1  # fallback

                data.append({
                    "text": text,
                    "label": label
                })

# -----------------------------
# LOAD HC3 DATASET (IMPROVED)
# -----------------------------
try:
    from datasets import load_dataset

    print("Loading HC3 dataset...")

    hc3 = load_dataset("Hello-SimpleAI/HC3", split="train[:3000]")

    hc3_count = 0

    for item in hc3:
        # Human
        if "human_answers" in item:
            for text in item["human_answers"]:
                text = text.strip()
                if text:
                    data.append({"text": text, "label": 0})
                    hc3_count += 1

        # AI
        if "chatgpt_answers" in item:
            for text in item["chatgpt_answers"]:
                text = text.strip()
                if text:
                    data.append({"text": text, "label": 1})
                    hc3_count += 1

    print(f"HC3 samples added: {hc3_count}")

except Exception as e:
    print("Error loading HC3:", e)

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame(data)

# -----------------------------
# CLEANING
# -----------------------------
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Ensure correct types
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

# -----------------------------
# BALANCE HINDI VS OTHER TEXT
# -----------------------------
# Detect Hindi characters
hindi_df = df[df["text"].str.contains(r'[ऀ-ॿ]', regex=True)]
other_df = df[~df["text"].str.contains(r'[ऀ-ॿ]', regex=True)]

print("Hindi samples before:", len(hindi_df))
print("Other samples before:", len(other_df))

# Limit Hindi to avoid dominance
if len(hindi_df) > 3000:
    hindi_df = hindi_df.sample(n=3000, random_state=42)

df = pd.concat([hindi_df, other_df])

# -----------------------------
# BALANCE LABELS (VERY IMPORTANT)
# -----------------------------
df_0 = df[df["label"] == 0]
df_1 = df[df["label"] == 1]

min_size = min(len(df_0), len(df_1))

df_0 = df_0.sample(n=min_size, random_state=42)
df_1 = df_1.sample(n=min_size, random_state=42)

df = pd.concat([df_0, df_1])

# -----------------------------
# FINAL SHUFFLE
# -----------------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# STATS
# -----------------------------
print("\nFinal dataset size:", len(df))
print("Label distribution:\n", df["label"].value_counts())

# -----------------------------
# SAVE
# -----------------------------
df.to_csv("data/final_dataset.csv", index=False)

print("final_dataset.csv saved successfully ✅")