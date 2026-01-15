# ==============================
# Urgency Classification Model
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score
import joblib

# ==============================
# 1. Load data
# ==============================
df = pd.read_csv(
    "C:/Users/USER/Projects/nlp_text_classification/text_classification_data.csv"
)

df = df[["clean_feedback_text", "urgency"]].dropna()
df["clean_feedback_text"] = df["clean_feedback_text"].astype(str).str.strip()
df["urgency"] = df["urgency"].astype(str).str.strip()

# Binary target: High vs Not-High
df["urgency_binary"] = df["urgency"].str.lower().apply(
    lambda x: 1 if x == "high" else 0
)

print("Total samples:", df.shape[0])
print("High urgency samples:", df["urgency_binary"].sum())

# ==============================
# 2. Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_feedback_text"],
    df["urgency_binary"],
    test_size=0.2,
    random_state=42,
    stratify=df["urgency_binary"]
)

# ==============================
# 3. TF-IDF
# ==============================
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ==============================
# 4. Train model
# ==============================
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Urgency model trained ✅")

# ==============================
# 5. Evaluation
# ==============================
y_pred = model.predict(X_test_tfidf)

print("\nUrgency Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not-High", "High"],
        zero_division=0
    )
)

high_recall = recall_score(y_test, y_pred, pos_label=1)
print("High-urgency recall:", round(high_recall, 4))

# ==============================
# 6. Save artifacts
# ==============================
joblib.dump(tfidf, "urgency_tfidf.joblib")
joblib.dump(model, "urgency_model.joblib")

print("Urgency model artifacts saved ✅")
