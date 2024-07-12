
# Heart Disease Classification

## Overview
This project evaluates the performance of various machine learning models for predicting heart disease using the heart_uci dataset. The dataset contains features like age, gender, chest pain type, resting blood pressure, cholesterol levels, and more. The models trained and evaluated include Gaussian Naive Bayes, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Decision Tree, and Random Forest.

## Dataset
The dataset consists of the following columns:
- *age*: Age of the patient
- *sex*: Gender of the patient (1 = male; 0 = female)
- *cp*: Chest pain type (0-3)
- *trestbps*: Resting blood pressure
- *chol*: Serum cholesterol in mg/dl
- *fbs*: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- *restecg*: Resting electrocardiographic results (0-2)
- *thalach*: Maximum heart rate achieved
- *exang*: Exercise induced angina (1 = yes; 0 = no)
- *oldpeak*: ST depression induced by exercise relative to rest
- *slope*: The slope of the peak exercise ST segment (0-2)
- *ca*: Number of major vessels (0-3) colored by fluoroscopy
- *thal*: Thalassemia (1-3)
- *target*: Diagnosis of heart disease (1 = disease; 0 = no disease)

## Model Performance Comparison

### Performance Metrics

| Model                        | Accuracy   |
|------------------------------|------------|
| **Gaussian Naive Bayes**     |82.80%      |
| **K-Nearest Neighbors (KNN)**| 81.57%     |
| **Support Vector Classifier (SVC)** |  85.50%     |
| **Decision Tree**            |  80.26%    |
| **Random Forest**            |  82.89%    |

### Performance Metric Definitions

- **Precision (Class 1)**: Precision is the ratio of correctly predicted positive observations to the total predicted positives. High precision indicates a low false positive rate.
- **Recall (Class 1)**: Recall is the ratio of correctly predicted positive observations to all observations in the actual class. High recall indicates a low false negative rate.
- **F1 Score (Class 1)**: The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both concerns.
- **ROC-AUC Score**: Receiver Operating Characteristic - Area Under the Curve (ROC-AUC) summarizes the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.
- **Accuracy**: Overall accuracy of the model, which may be affected by class imbalance.

### Analysis of Graphs

#### Distribution of Age
![Screenshot 2024-07-12 220909](https://github.com/user-attachments/assets/ade22aa1-b59a-4a24-97c3-d68ddf6cf90a)


This histogram shows the age distribution of patients in the dataset. Most patients are between 40 and 70 years old, with a peak around 55 years.

#### Relation between Max Heart Rate and Target
![Screenshot 2024-07-12 221020](https://github.com/user-attachments/assets/421df32c-b976-46a7-8d47-5321ba2aeb4f)


This scatter plot shows the relationship between the maximum heart rate achieved (`thalach`) and the presence of heart disease (`target`). There is a positive correlation, indicating that patients with higher maximum heart rates tend to have heart disease.

### Gender Distribution
68.32% of the patients are male, and the remaining are female.

![download](https://github.com/user-attachments/assets/78e44707-9470-4214-b55b-708b824a1229)


### Confusion Matrix Analysis

#### SVC Confusion Matrix
|            | Predicted Disease | Predicted No Disease |
|------------|-------------------|----------------------|
| **Actual Disease**     | 24                | 9                    |
| **Actual No Disease**  | 2                 | 41                   |

- **True Positives (TP)**: 24
- **False Positives (FP)**: 9
- **True Negatives (TN)**: 41
- **False Negatives (FN)**: 2

### Classification Report
                precision    recall  f1-score   support

       0            0.92      0.73      0.81        33
       1            0.82      0.95      0.88        43
    accuracy                            0.86        76
    macro avg       0.87      0.84      0.85        76
    weighted avg    0.86      0.86      0.85        76

### AUC Score
The ROC-AUC score for the SVC model is 0.840.

### Insights and Recommendations

- **Support Vector Classifier (SVC)**:
  - **Precision**: High (0.82), indicating fewer false positives.
  - **Recall**: Very high (0.95), detecting most cases of heart disease.
  - **F1 Score**: Best balance (0.88) between precision and recall.
  - **ROC-AUC Score**: High (0.840), strong discriminative ability.
  - **Accuracy**: Highest (85.50%), making it the best-performing model in this comparison.

## Conclusion
The evaluation of machine learning models for heart disease classification shows that the Support Vector Classifier (SVC) performed the best. The SVC achieved the highest accuracy (85.5%) and a robust AUC Score (0.8404). 

### Interpretation of SVC Performance
The Support Vector Classifier (SVC) likely performed the best due to its ability to handle high-dimensional spaces and its effectiveness in finding the optimal hyperplane that maximizes the margin between different classes. The kernel trick allows SVC to efficiently perform a non-linear classification, which can capture more complex relationships within the dataset. This results in better generalization and higher accuracy in predicting heart disease compared to other models tested.


### Installation and Setup

#### Prerequisites
- Python 3.x
- Jupyter Notebook
- Necessary libraries: pandas, numpy, matplotlib, scikit-learn

#### Setup
1. Clone the repository:
   ```bash
   git clone "https://github.com/your_username/HeartDiseaseClassification.git"
   cd HeartDiseaseClassification

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook
