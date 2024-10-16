# -*- coding: utf-8 -*-
"""nbcmp.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17df-jWUhNRvYdLiimxdDjGJbUEd1cssQ
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

# Custom Naive Bayes class
class CustomNaiveBayes:
    def __init__(self, epsilon=1e-6, noise_factor=0.01):
        self.classes = None
        self.feature_params = None
        self.epsilon = epsilon
        self.noise_factor = noise_factor

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.feature_params = {c: {} for c in self.classes}
        for c in self.classes:
            X_c = X[y == c]
            self.feature_params[c]['mean'] = np.mean(X_c, axis=0)
            self.feature_params[c]['var'] = np.var(X_c, axis=0)

    def predict(self, X):
        log_probs = np.zeros((X.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            mean = self.feature_params[c]['mean'] + np.random.normal(0, self.noise_factor, size=X.shape[1])
            var = self.feature_params[c]['var'] + self.noise_factor
            feature_log_prob = -0.5 * np.sum(np.log(2 * np.pi * var)) \
                               - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_probs[:, idx] = feature_log_prob
        return self.classes[np.argmax(log_probs, axis=1)]

# Load the dataset
file_path = "ion_binary_classification.csv"
data = pd.read_csv(file_path)
data = data.drop(columns=["Unnamed: 0"])  # Drop unnecessary column
data["Class"] = data["Class"].map({"good": 1, "bad": 0})  # Map classes to 1 and 0

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train Custom Naive Bayes
nb_custom = CustomNaiveBayes()
nb_custom.fit(X_train_pca, y_train)
y_pred_custom = nb_custom.predict(X_test_pca)

# Train Sklearn Naive Bayes
nb_sklearn = GaussianNB()
nb_sklearn.fit(X_train_pca, y_train)
y_pred_sklearn = nb_sklearn.predict(X_test_pca)

# Calculate metrics for both models
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

custom_metrics = [
    accuracy_score(y_test, y_pred_custom),
    precision_score(y_test, y_pred_custom, zero_division=0),
    recall_score(y_test, y_pred_custom, zero_division=0),
    f1_score(y_test, y_pred_custom, zero_division=0)
]

sklearn_metrics = [
    accuracy_score(y_test, y_pred_sklearn),
    precision_score(y_test, y_pred_sklearn, zero_division=0),
    recall_score(y_test, y_pred_sklearn, zero_division=0),
    f1_score(y_test, y_pred_sklearn, zero_division=0)
]

# Print metrics
print(f"Custom Naive Bayes Metrics:\n"
      f"Accuracy: {custom_metrics[0]:.2f}, Precision: {custom_metrics[1]:.2f}, "
      f"Recall: {custom_metrics[2]:.2f}, F1 Score: {custom_metrics[3]:.2f}")

print(f"Sklearn Naive Bayes Metrics:\n"
      f"Accuracy: {sklearn_metrics[0]:.2f}, Precision: {sklearn_metrics[1]:.2f}, "
      f"Recall: {sklearn_metrics[2]:.2f}, F1 Score: {sklearn_metrics[3]:.2f}")

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_custom), annot=True, cmap='Blues', fmt='d', ax=axes[0])
axes[0].set_title('Custom Naive Bayes Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_sklearn), annot=True, cmap='Blues', fmt='d', ax=axes[1])
axes[1].set_title('Sklearn Naive Bayes Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.show()

# Handle ROC curve in case of single-class prediction
try:
    y_prob_sklearn = nb_sklearn.predict_proba(X_test_pca)[:, 1]
except IndexError:
    y_prob_sklearn = np.zeros_like(y_test)  # Default to zero probabilities if needed

fpr_custom, tpr_custom, _ = roc_curve(y_test, y_pred_custom)
fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, y_prob_sklearn)

roc_auc_custom = auc(fpr_custom, tpr_custom) if len(fpr_custom) > 1 else 0.5
roc_auc_sklearn = auc(fpr_sklearn, tpr_sklearn) if len(fpr_sklearn) > 1 else 0.5

# Plot ROC curve comparison
plt.figure(figsize=(10, 6))
sns.lineplot(x=fpr_custom, y=tpr_custom, label=f'Custom Naive Bayes (AUC = {roc_auc_custom:.2f})', color='blue')
sns.lineplot(x=fpr_sklearn, y=tpr_sklearn, label=f'Sklearn Naive Bayes (AUC = {roc_auc_sklearn:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Plot bar chart comparison of metrics
metrics_df = pd.DataFrame({
    'Metrics': metrics,
    'Custom Naive Bayes': custom_metrics,
    'Sklearn Naive Bayes': sklearn_metrics
})

plt.figure(figsize=(10, 6))
metrics_df.plot(kind='bar', x='Metrics', width=0.8, ax=plt.gca())
plt.title('Performance Comparison of Custom and Sklearn Naive Bayes', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.ylim(0.0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()