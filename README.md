<div style="position: absolute; top: 1px; right: -1px;">
    <img src="https://github.com/user-attachments/assets/934c393e-0438-46b0-a004-4acc53996720" alt="iitjlogo" width="100"/>
</div>

# **ML-2024-Assignment - 2 (NB)**

**Title:** **Comparison of Custom Naive Bayes and Sklearn Naive Bayes Models for Binary Classification**

**Authors:**  
- **Brijesh Kumar Karna** (g24ait025@iitj.ac.in)  
- **Raj Kumar** (g24ait022@iitj.ac.in)  
- **Shaktijit Rautaray** (g24ait053@iitj.ac.in)  

**Supervisor:** Dr. Asif Ekbal  
**Institution:** Indian Institute of Technology Jodhpur  
**Date:** 16-Oct-2024  

---

## **Abstract**
This project explores and compares the performance of a custom-built Naive Bayes classifier with the widely used Sklearn Naive Bayes model for binary classification tasks. Both models are evaluated using a real-world dataset, and their performance is assessed based on key metrics such as accuracy, precision, recall, F1-score, and AUC (Area Under the Curve). Results indicate that the custom model provides better recall and F1-score, while the Sklearn implementation offers a higher AUC score. This project provides insights into the trade-offs between developing custom models and using standard implementations.

---

## **Table of Contents**
1. [Introduction](#1-introduction)  
2. [Objectives](#2-objectives)  
3. [Methodology](#3-methodology)  
   - [Dataset](#31-dataset)  
   - [Model Development and Implementation](#32-model-development-and-implementation)  
4. [Results and Discussion](#4-results-and-discussion)  
   - [Confusion Matrix Analysis](#41-confusion-matrix-analysis)  
   - [Performance Metrics Comparison](#42-performance-metrics-comparison)  
   - [ROC Curve Comparison](#43-roc-curve-comparison)  
5. [Conclusion and Future Work](#5-conclusion-and-future-work)  
6. [Source Code](#6-source-code)  
7. [References](#7-references)  

---

## 1. **Introduction**
In the field of machine learning, **Naive Bayes** is a probabilistic classifier based on **Bayes’ theorem** with an assumption of conditional independence between features. Its simplicity and computational efficiency make it a popular choice for many real-world applications, including spam detection, sentiment analysis, and disease prediction. 

However, while pre-built solutions like **Sklearn’s Gaussian Naive Bayes** offer convenience, building a **custom Naive Bayes** from scratch can provide a deeper understanding of the algorithm’s working, along with opportunities for optimization. The purpose of this project is to compare a custom-built Naive Bayes classifier with the standard Sklearn implementation to explore differences in performance and behavior across metrics.

This comparison aims to determine if a custom-built model can outperform the standard implementation in specific use cases, particularly in scenarios where minimizing **false negatives** is crucial.

---

## 2. **Objectives**
The main objectives of this project are:
1. To implement a **Custom Naive Bayes Classifier** from scratch, incorporating Gaussian noise handling.
2. To evaluate the **performance** of both the **Custom Naive Bayes** and the **Sklearn Naive Bayes** using a real-world dataset.
3. To compare both models using key metrics: **accuracy, precision, recall, F1-score**, and **AUC (Area Under Curve)**.
4. To identify scenarios where the **custom model** might provide better performance than the standard Sklearn model.

---

## 3. **Methodology**

### 3.1 **Dataset**
The **Ion Binary Classification Dataset** from the UCI Machine Learning Repository was used for this project. This dataset contains features derived from radar signals that classify whether a signal is **good** (Class 1) or **bad** (Class 0).  

**Data Preprocessing**:
- The original class labels were mapped to binary values:  
  - **good → 1**  
  - **bad → 0**  
- The features were scaled using **StandardScaler** to standardize the values.
- **Principal Component Analysis (PCA)** was applied to reduce dimensionality while retaining 95% of the dataset’s variance.

### 3.2 **Model Development and Implementation**
- **Custom Naive Bayes**:  
  The custom Naive Bayes model computes the **mean** and **variance** of features for each class during training. During prediction, **Gaussian noise** is added to the mean for better generalization, and log-probabilities are used to improve numerical stability.

- **Sklearn Naive Bayes**:  
  The **GaussianNB** implementation from the `sklearn` library was used for comparison, as it is optimized for continuous data.

Both models were trained on **80% of the data** and tested on the remaining **20%** to evaluate their performance.

---

## 4. **Results and Discussion**

### 4.1 **Confusion Matrix Analysis**
![Confusion Matrix](https://github.com/user-attachments/assets/2369f654-f8ae-47b4-a2c4-cb77a7ed152a)  
**Figure 1:** Confusion Matrices for Custom and Sklearn Naive Bayes  

| **Metric / Confusion Matrix Element** | **Custom Naive Bayes** | **Sklearn Naive Bayes** |
|---------------------------------------|------------------------|-------------------------|
| **True Positives (Class 1)**          | 42                     | 41                      |
| **True Negatives (Class 0)**          | 25                     | 25                      |
| **False Positives**                   | 3                      | 3                       |
| **False Negatives**                   | 1                      | 2                       |

---

### 4.2 **Performance Metrics Comparison**
![Performance Metrics](https://github.com/user-attachments/assets/b33dd8aa-2fdc-456a-8355-2c7d0b61b64e)  
**Figure 2:** Performance Metrics Comparison  

| **Metric**         | **Custom Naive Bayes** | **Sklearn Naive Bayes** |
|--------------------|------------------------|-------------------------|
| **Accuracy**       | 0.94                   | 0.93                    |
| **Precision**      | 0.93                   | 0.93                    |
| **Recall**         | 0.98                   | 0.95                    |
| **F1 Score**       | 0.95                   | 0.94                    |

---

### 4.3 **ROC Curve Comparison**
![ROC Curve](https://github.com/user-attachments/assets/55cd74dc-4fa8-4d10-a8fd-0ce196405288)  
**Figure 3:** ROC Curve Comparison  

| **Metric**         | **Custom Naive Bayes** | **Sklearn Naive Bayes** |
|--------------------|------------------------|-------------------------|
| **AUC (Area Under Curve)** | 0.93            | 0.97                    |

---

## **Why Noise is Used and How it Improves Results**

In the custom Naive Bayes implementation, **Gaussian noise** is introduced during prediction by adding random perturbations to the feature means. This improves performance in the following ways:

1. **Prevents Overfitting:** Noise ensures that the model does not overly rely on exact feature values.
2. **Improves Generalization:** The classifier becomes more robust to variations in new data.
3. **Balances Bias-Variance Trade-off:** Noise allows the model to avoid being too simplistic or too complex.
4. **Enhances Numerical Stability:** Adjusting the variance prevents division errors when working with low-variance features.

---

## 5. **Conclusion and Future Work**

### 5.1 **Conclusion**
This project demonstrates that both **Custom Naive Bayes** and **Sklearn Naive Bayes** are effective models for binary classification, with key observations as follows:
- **Custom Naive Bayes** performs slightly better in **recall** and **F1-score**.
- **Sklearn Naive Bayes** achieves a higher **AUC**, making it better for balanced threshold scenarios.

---

### 5.2 **Future Work**
1. **Hyperparameter Tuning**: Experiment with different noise factors.  
2. **Handling Feature Dependencies**: Explore **Bayesian Networks**.  
3. **Real-World Testing**: Apply models to complex datasets.

---

## 6. **Source Code**
- **[Colab Link](https://colab.research.google.com/drive/17df-jWUhNRvYdLiimxdDjGJbUEd1cssQ?usp=sharing)**  
- **[Python File Link](https://smalldev.tools/share-bin/gzIzlkAq)**

---

## 7. **References**
- [Sklearn Documentation](https://scikit-learn.org/1.5/modules/naive_bayes.html)  
- [Ionosphere Dataset](https://archive.ics.uci.edu/ml/datasets/Ionosphere)
