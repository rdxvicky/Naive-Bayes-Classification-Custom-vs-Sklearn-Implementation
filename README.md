# Naive Bayes Classification: Custom vs Sklearn Implementation

## 1. Introduction
In this project, we implement and compare two Naive Bayes models for binary classification:
1. **Custom Naive Bayes Implementation:** A hand-crafted implementation with intentional noise and minor tweaks.
2. **Sklearn Naive Bayes Implementation:** The Gaussian Naive Bayes model from the `sklearn` library.

The dataset used for this comparison is a binary classification dataset (`ion_binary_classification.csv`). 
We apply dimensionality reduction using PCA to retain 95% of the variance, followed by training and evaluation.

---

## 2. Implementation Details

### 2.1 Dataset Preparation
1. **Data Preprocessing:** The dataset has 36 columns. We dropped the unnecessary `Unnamed: 0` column.
2. **Class Mapping:** Mapped the class labels from `"good"` to `1` and `"bad"` to `0` for binary classification.
3. **Feature Standardization:** Used `StandardScaler` to normalize the feature values.
4. **Dimensionality Reduction:** Applied PCA to retain 95% of the dataset's variance.
5. **Data Splitting:** Split the data into training and testing sets with an 80-20 ratio.

### 2.2 Custom Naive Bayes Implementation
The custom Naive Bayes model was built from scratch with the following key aspects:
- Added small random noise to the mean to introduce variance.
- No smoothing parameter (epsilon) was applied to the variance intentionally.
- The class prior was ignored in the final probability calculation to introduce randomness.

### 2.3 Sklearn Naive Bayes Implementation
- Used the `GaussianNB` class from `sklearn`.
- The model was trained and evaluated on the same data for a fair comparison.

---

## 3. Results and Execution Time

### 3.1 Performance Metrics
| Metric       | Custom Naive Bayes | Sklearn Naive Bayes |
|--------------|--------------------|---------------------|
| Accuracy     | **0.94**           | 0.93                |
| Precision    | **0.93**           | 0.93                |
| Recall       | **0.98**           | 0.95                |
| F1 Score     | **0.95**           | 0.94                |

### 3.2 Execution Time
- **Custom Naive Bayes:** 0.0008 seconds
- **Sklearn Naive Bayes:** 0.0022 seconds

---

## 4. Analysis and Observations

1. **Performance:** 
   - The custom Naive Bayes model performs slightly better in terms of recall and F1 score.
   - Both models achieved comparable accuracy, but the custom model had a minor edge.

2. **Execution Time:**
   - The custom implementation was faster, primarily due to its minimalistic approach without smoothing or other optimizations.

3. **Trade-offs:** 
   - The custom model’s lack of smoothing can lead to variance in predictions for different datasets.
   - Sklearn’s implementation is more robust and consistent across datasets.

4. **Recommendations:**
   - For production or real-world usage, the `sklearn` implementation is recommended due to its consistency.
   - Custom models can be useful for educational purposes or specific applications requiring tweaks.

---

## 5. Visualizations

### 5.1 Confusion Matrices
- **Custom Naive Bayes Confusion Matrix** VS **Sklearn Naive Bayes Confusion Matrix**
  
  ![cfmatrix](https://github.com/user-attachments/assets/ff756ecb-a018-471a-864b-77bf8b45c2f7)

### 5.2 ROC Curve Comparison
- 
   ![RocCurve](https://github.com/user-attachments/assets/dfddc58c-f4e8-4127-b2bd-058c1013945a)


### 5.3 Metrics Comparison
- 
   ![PerformanceCNBSKNB](https://github.com/user-attachments/assets/01115937-82b6-412b-93c3-6c4b6b54b496)

---

## 6. Conclusion
This project demonstrates the implementation of Naive Bayes models from scratch and using the Sklearn library. 
While the custom model shows slight improvements in some metrics, the Sklearn model remains the better choice for practical use due to its robustness and reliability.
