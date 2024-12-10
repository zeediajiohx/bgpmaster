import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Your existing code...

train_losses = []  # List to store training losses
test_accuracies = []  # List to store test accuracies
test_predictions = []  # List to store test predictions
test_targets = []  # List to store test targets

for epoch in range(epochs):
    # Your existing training loop...

    train_losses.append(loss.item())  # Store training loss

    # Your existing evaluation code...

    test_accuracies.append(accuracy)  # Store test accuracy
    test_predictions.append(pred_y)
    test_targets.append(test_y.numpy())

# After training loop completes...

# Plot loss curve
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# Combine test_predictions and test_targets to compute ROC, AUC, precision, recall
test_predictions = np.concatenate(test_predictions)
test_targets = np.concatenate(test_targets)

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(test_targets, test_predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Compute precision-recall curve and AUC
precision, recall, _ = precision_recall_curve(test_targets, test_predictions)
average_precision = average_precision_score(test_targets, test_predictions)

# Plot precision-recall curve
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve (AUC = %0.2f)' % average_precision)
plt.show()

# Calculate and print accuracy, recall, AUC
accuracy = np.sum(test_predictions == test_targets) / len(test_targets)
recall = np.sum((test_predictions == 1) & (test_targets == 1)) / np.sum(test_targets == 1)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("AUC:", roc_auc)
