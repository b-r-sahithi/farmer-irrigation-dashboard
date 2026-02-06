import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("nasa_irrigation_dataset.csv")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# FEATURES (IMPORTANT: NO idi_7day)
# --------------------------------------------------
X = df[["rainfall", "eto"]]
y = df["action_needed"]

print("\nClass distribution:")
print(y.value_counts())

# --------------------------------------------------
# Train-test split (STRATIFIED)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# --------------------------------------------------
# Train Random Forest (imbalance-aware)
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
print("\nModel training completed")

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
y_pred = model.predict(X_test)

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, digits=3))

# --------------------------------------------------
# Save model
# --------------------------------------------------
joblib.dump(model, "model.pkl")
print("\n✅ model.pkl saved (NO LEAKAGE)")
