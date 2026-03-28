# Intern_task_7


### 📄 `README.md`

````markdown
# 🧠 Breast Cancer Diagnosis with Support Vector Machines (SVM)

This project demonstrates how to use Support Vector Machines (SVM) for binary classification on the **Breast Cancer Wisconsin** dataset. It includes data preprocessing, training with both linear and RBF kernels, decision boundary visualization using PCA, hyperparameter tuning, and performance evaluation using cross-validation.

---

## 📁 Dataset

The dataset used is `breast-cancer.csv`, which includes features derived from digitized images of fine needle aspirate (FNA) of breast masses.

- **Target:** `diagnosis` (0 = Benign, 1 = Malignant)
- **Features:** 30 numerical columns (e.g., `radius_mean`, `texture_mean`, etc.)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

## 🛠️ Features & Workflow

1. **Data Preprocessing**
   - Removed unnecessary columns (e.g., `id`)
   - Encoded categorical labels
   - Scaled features using `StandardScaler`

2. **Model Training**
   - Trained SVM with **linear** and **RBF** kernels
   - Visualized decision boundaries using **PCA (2D)**

3. **Hyperparameter Tuning**
   - Used `GridSearchCV` to find optimal values for `C` and `gamma`

4. **Evaluation**
   - Confusion Matrix
   - Classification Report
   - Cross-validation scores

---

## 🖥️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-svm.git
   cd breast-cancer-svm
````

2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset file `breast-cancer.csv` in the project folder.

4. Run the script:

   ```bash
   python svm_breast_cancer.py
   ```

---

## 📊 Sample Output

* **Best Parameters:** e.g., `C=10`, `gamma=0.01`
* **Confusion Matrix** and **Classification Report**
* **Mean Cross-Validation Accuracy**: \~97–99%
* **Decision boundary plots** using PCA

---

## 📦 Requirements

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`

Alternatively, use:

```bash
pip install -r requirements.txt
```

---

## 📸 Visuals

Decision boundary visualizations (PCA-reduced):

* SVM with **Linear Kernel**
* SVM with **RBF Kernel**

---

````

---

### ✅ Optional: `requirements.txt`
Create a `requirements.txt` file:

```txt
numpy
pandas
matplotlib
scikit-learn
````
