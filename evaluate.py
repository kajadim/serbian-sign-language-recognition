import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

X_test  = np.load("models/X_test.npy")
y_test  = np.load("models/y_test.npy")
classes = np.load("models/classes.npy", allow_pickle=True)
model   = load_model("models/best_model.keras")

y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = np.mean(y_pred == y_test)
print(f"\nOverall accuracy of the model on the test set: {acc*100:.2f}%\n")

print("=" * 60)
print("PER-CLASS PERFORMANCE:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=classes))

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(classes))
)

print("=" * 60)
print("WORST PERFORMING CLASSES (F1 < 0.85):")
print("=" * 60)
for i, cls in enumerate(classes):
    if f1[i] < 0.85:
        print(f"  {cls:4s} -> Precision: {precision[i]:.2f}  Recall: {recall[i]:.2f}  F1: {f1[i]:.2f}  (n={support[i]})")

bad_classes = [i for i, s in enumerate(f1) if s < 0.85]
if bad_classes:
    cm_full = confusion_matrix(y_test, y_pred)
    print("\nTop 15 misclassifications (what the model predicts instead of the correct class):")
    errors = []
    for true_idx in bad_classes:
        for pred_idx in range(len(classes)):
            if true_idx != pred_idx and cm_full[true_idx][pred_idx] > 0:
                errors.append((cm_full[true_idx][pred_idx], classes[true_idx], classes[pred_idx]))
    errors.sort(reverse=True)
    for count, true_cls, pred_cls in errors[:15]:
        print(f"  True={true_cls:4s}  ->  Predicted={pred_cls:4s}  ({count} times)")
else:
    print("\nNo classes below the 0.85 threshold!")

plt.figure(figsize=(14, 5))
colors = ['red' if s < 0.85 else 'steelblue' for s in f1]
plt.bar(classes, f1, color=colors)
plt.axhline(y=0.85, color='orange', linestyle='--', label='Prag 0.85')
plt.title('F1 Score per Class - Model (body normalization + curl)')
plt.xlabel('Letter')
plt.ylabel('F1 score')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('class_evaluation.png', dpi=150)
plt.show()

print("\nPlot saved as 'class_evaluation.png'")