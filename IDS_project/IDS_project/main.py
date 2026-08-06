import pandas as pd
import numpy as np
import joblib   # ✅ ADD THIS

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("KDDTrain+.txt", header=None)

print("Dataset Shape:", df.shape)

# =========================
# SPLIT
# =========================
X = df.iloc[:, :-2]
y = df.iloc[:, -2]

# =========================
# FIX LABEL
# =========================
y = y.apply(lambda x: 0 if str(x).strip() == 'normal' else 1)

# =========================
# ENCODE CATEGORICAL
# =========================
cat_cols = X.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# =========================
# TRAIN TEST SPLIT
# =========================
# Split BEFORE scaling. Fitting the scaler on the full dataset lets the test
# set's mean and variance leak into the training transform, which inflates the
# reported accuracy.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# SCALE (fit on train only)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# =========================
# 🔥 SAVE MODEL + SCALER HERE
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ model.pkl and scaler.pkl saved!")

# =========================
# NORMAL TEST
# =========================
y_pred = model.predict(X_test)

normal_acc = accuracy_score(y_test, y_pred)

print("\n=== NORMAL DATA ===")
print("Accuracy:", normal_acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# =========================
# ADVERSARIAL TEST
# =========================
noise = np.random.normal(0, 0.5, X_test.shape)
X_test_adv = X_test + noise

y_pred_adv = model.predict(X_test_adv)

adv_acc = accuracy_score(y_test, y_pred_adv)

print("\n=== ADVERSARIAL DATA ===")
print("Accuracy:", adv_acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_adv))
print("Classification Report:\n", classification_report(y_test, y_pred_adv))

# =========================
# FINAL COMPARISON
# =========================
print("\n=== FINAL COMPARISON ===")
print(f"Normal Accuracy: {normal_acc:.4f}")
print(f"Adversarial Accuracy: {adv_acc:.4f}")