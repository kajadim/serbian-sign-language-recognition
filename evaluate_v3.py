import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

X_test  = np.load("models/X_test_v3.npy")
y_test  = np.load("models/y_test_v3.npy")
classes = np.load("models/classes_v3.npy", allow_pickle=True)
model   = load_model("models/best_model_v3.keras")

y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = np.mean(y_pred == y_test)
print(f"\nUkupna tacnost V3 modela na test setu: {acc*100:.2f}%\n")

print("=" * 60)
print("TACNOST PO KLASAMA (V3):")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=classes))

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(classes))
)

print("=" * 60)
print("NAJLOSIJE KLASE (F1 < 0.85):")
print("=" * 60)
for i, cls in enumerate(classes):
    if f1[i] < 0.85:
        print(f"  {cls:4s} -> Precision: {precision[i]:.2f}  Recall: {recall[i]:.2f}  F1: {f1[i]:.2f}  (n={support[i]})")

bad_classes = [i for i, s in enumerate(f1) if s < 0.85]
if bad_classes:
    cm_full = confusion_matrix(y_test, y_pred)
    print("\nTop 15 gresaka (sta model misli umesto tacnog):")
    errors = []
    for true_idx in bad_classes:
        for pred_idx in range(len(classes)):
            if true_idx != pred_idx and cm_full[true_idx][pred_idx] > 0:
                errors.append((cm_full[true_idx][pred_idx], classes[true_idx], classes[pred_idx]))
    errors.sort(reverse=True)
    for count, true_cls, pred_cls in errors[:15]:
        print(f"  Tacno={true_cls:4s}  ->  Model kaze={pred_cls:4s}  ({count}x)")
else:
    print("\nNema klasa ispod praga 0.85!")

plt.figure(figsize=(14, 5))
colors = ['red' if s < 0.85 else 'steelblue' for s in f1]
plt.bar(classes, f1, color=colors)
plt.axhline(y=0.85, color='orange', linestyle='--', label='Prag 0.85')
plt.title('F1 score po klasi - V3 model (normalizacija na telo + curl)')
plt.xlabel('Slovo')
plt.ylabel('F1 score')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('evaluacija_po_klasama_v3.png', dpi=150)
plt.show()

print("\nGrafik sacuvan kao 'evaluacija_po_klasama_v3.png'")