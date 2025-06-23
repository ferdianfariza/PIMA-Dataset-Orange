# Diabetes Prediction using Machine Learning

This project aims to predict whether a patient has diabetes using various machine learning models trained on the **PIMA Indians Diabetes Dataset**.

📄 **Reference Paper**:  
> *Prediction of Diabetes Using Machine Learning Algorithms*  
> [IJSREM, April 2024](https://ijsrem.com/download/prediction-of-diabetes-using-machine-learning-algorithms/)

---

## Dataset

- Source: [Kaggle – PIMA Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 rows × 9 columns (8 features + 1 target)
- All patients are women aged 21+ from the Pima Indian population
- Target column: `Outcome` (0 = non-diabetic, 1 = diabetic)

### Features
| Feature | Description |
|--------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic risk function |
| Age | Age in years |

---

## Preprocessing

1. Replace 0s in certain columns (Glucose, Insulin, etc.) with `NaN`
2. Impute missing values using **median**
3. Cap outliers using **Z-score** and **IQR methods**
4. Use **SMOTE** to fix class imbalance

---

## Machine Learning Models

| Model               | AUC   | Accuracy (CA) | F1    | Precision | Recall | MCC   |
| ------------------- | ----- | ------------- | ----- | --------- | ------ | ----- |
| Random Forest       | 0.889 | 0.811         | 0.811 | 0.811     | 0.811  | 0.627 |
| Gradient Boosting   | 0.879 | 0.808         | 0.808 | 0.809     | 0.808  | 0.619 |
| Neural Network      | 0.876 | 0.804         | 0.804 | 0.805     | 0.804  | 0.612 |
| Logistic Regression | 0.845 | 0.755         | 0.755 | 0.755     | 0.755  | 0.510 |
| SVM                 | 0.745 | 0.682         | 0.681 | 0.684     | 0.682  | 0.378 |

> *Random Forest achieved the highest overall performance.*

---

## Installation Instructions

### Windows, macOS, and Linux

#### Option A: Using Conda (Recommended)
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), then run:

```bash
conda create -n diabetes-ml python=3.10
conda activate diabetes-ml
conda install -c conda-forge pandas scikit-learn imbalanced-learn matplotlib seaborn
