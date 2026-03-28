# Logistic Regression on Breast Cancer Dataset

This project performs binary classification using logistic regression on the Breast Cancer Wisconsin dataset. The main objective is to predict whether a tumor is **malignant (M)** or **benign (B)** based on various medical measurements.

## 🔍 Dataset

The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. The target column is:
- `diagnosis`: Malignant (M = 1) or Benign (B = 0)

Source: `data.csv` (included in the `data/` directory)

## 📊 Workflow

1. **Load and preprocess dataset**
2. **Train-test split (80/20)**
3. **Feature standardization**
4. **Logistic Regression model fitting**
5. **Evaluation using:**
   - Confusion Matrix
   - Precision, Recall
   - ROC-AUC and ROC Curve
6. **Threshold tuning**
7. **Sigmoid function visualization**

## 🚀 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/logistic-regression-breast-cancer.git
   cd logistic-regression-breast-cancer
