# ==============================
# 1. Imports
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
import joblib


# ==============================
# 2. Load data
# ==============================
# Change path if needed
df = pd.read_csv("C:/Users/USER/Projects/nlp_text_classification/text_classification_data.csv")

print("Initial shape:", df.shape)
print(df.columns)


# ==============================
# 3. Select modeling columns
# ==============================
df = df[["clean_feedback_text", "issue_category"]]


# ==============================
# 4. Drop invalid rows
# ==============================
df["clean_feedback_text"] = df["clean_feedback_text"].astype(str).str.strip()
df["issue_category"] = df["issue_category"].astype(str).str.strip()

df = df[
    (df["clean_feedback_text"] != "") &
    (df["issue_category"] != "")
]

df = df.reset_index(drop=True)

print("After cleaning shape:", df.shape)


# ==============================
# 5. Encode target labels
# ==============================
label_encoder = LabelEncoder()
df["issue_category_encoded"] = label_encoder.fit_transform(df["issue_category"])

# Save label mapping (VERY important later)
label_mapping = dict(zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
))

print("Label mapping:")
print(label_mapping)


# ==============================
# 6. Define X and y
# ==============================
X = df["clean_feedback_text"]
y = df["issue_category_encoded"]


# ==============================
# 7. Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])


# ==============================
# 8. TF-IDF Vectorization
# ==============================
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),     # unigrams + bigrams
    min_df=2,               # ignore very rare words
    max_df=0.9,             # ignore overly common words
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)


# ==============================
# 9. Train baseline model
# ==============================
log_reg = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

log_reg.fit(X_train_tfidf, y_train)

print("Model training completed ✅")

# ==============================
# 10. Evaluate model
# ==============================

y_pred = log_reg.predict(X_test_tfidf)

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# Macro F1
macro_f1 = f1_score(y_test, y_pred, average="macro")
print("Macro F1 score:", round(macro_f1, 4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ==============================
# 11. Sanity checks
# ==============================
assert X_train_tfidf.shape[0] == y_train.shape[0]
assert X_test_tfidf.shape[0] == y_test.shape[0]

print("Sanity checks passed ✅")


# ==============================
# 12. Save artifacts
# ==============================
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")

print("Artifacts saved successfully")
