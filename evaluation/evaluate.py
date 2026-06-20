import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


from modules.classifier import classify_clause
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)




# ---------------------------
# LOAD DATASET
# ---------------------------
with open("evaluation/hardcoded_dataset.json", "r") as f:
    dataset = json.load(f)

y_true = []
y_pred = []

# ---------------------------
# RUN CLASSIFIER
# ---------------------------
for item in dataset:
    clause = item["clause"]
    true_label = item["severity"]

    result = classify_clause(clause)
    pred_label = result["severity"]

    y_true.append(true_label)
    y_pred.append(pred_label)


# ---------------------------
# METRICS
# ---------------------------
print("\n========== EVALUATION ==========\n")

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="weighted"))
print("Recall   :", recall_score(y_true, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_true, y_pred, average="weighted"))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)


# ---------------------------
# GRAPH 1: CONFUSION MATRIX HEATMAP
# ---------------------------
plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Add color bar
plt.colorbar()

# Add numbers inside boxes
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="black"
        )

# Optional: label axes with class names
labels = sorted(list(set(y_true)))

plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

plt.show()


# ---------------------------
# GRAPH 2: CLASS DISTRIBUTION
# ---------------------------
plt.figure()
pd.Series(y_pred).value_counts().plot(kind="bar")
plt.title("Predicted Severity Distribution")
plt.show()