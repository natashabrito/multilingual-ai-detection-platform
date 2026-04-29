import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/final_dataset.csv")

# Safety fixes
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

X = df["text"]
y = df["label"]

# -----------------------------
# TRAIN-TEST SPLIT (STRATIFIED)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT
)

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=300,
        class_weight="balanced"
    ),

    "SVM": LinearSVC(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=20
    )
}

results = {}

# -----------------------------
# TRAIN + EVALUATE
# -----------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    results[name] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4)
    }

# -----------------------------
# ADD DISTILBERT (YOUR RESULT)
# -----------------------------
results["DistilBERT"] = {
    "accuracy": 0.95,
    "precision": 0.95,
    "recall": 0.95,
    "f1_score": 0.95
}

# -----------------------------
# SAVE JSON
# -----------------------------
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved model_results.json ✅")