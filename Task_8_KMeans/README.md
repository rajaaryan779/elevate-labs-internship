# Elevate_Labs_Task_8
# 🛍️ Mall Customer Segmentation using K-Means Clustering

This project demonstrates unsupervised learning using **K-Means Clustering** to segment mall customers based on their demographic and spending behavior. The objective is to identify different customer groups to help businesses tailor their marketing strategies.

---

## 📁 Dataset

- **Name**: Mall Customers Dataset
- **Attributes**:
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

> You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial) or use the file `Mall Customers.csv` provided in this repository.

---

## 📊 Tools & Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🧠 Key Concepts

- Unsupervised Learning
- K-Means Clustering
- Elbow Method
- PCA (Principal Component Analysis)
- Silhouette Score

---

## 📌 Steps Performed

1. **Load and Preprocess Data**
   - Encode categorical values
   - Standardize features

2. **PCA for Visualization**
   - Reduce dimensions to 2D for plotting

3. **Determine Optimal Clusters**
   - Use Elbow Method to find the best value for K

4. **Apply K-Means**
   - Cluster customers based on selected features

5. **Visualize Results**
   - Plot customer segments using PCA components

6. **Evaluate Model**
   - Compute Silhouette Score to assess cluster quality

---

## 📈 Output Example

- 📍 PCA plot of raw data  
- 📍 Elbow curve showing optimal `K = 5`
- 📍 Clustered customer segments visualized in 2D
- 📍 Silhouette Score printed in console (typically between 0.4 and 0.6)

---

## 💡 What You’ll Learn

- How K-Means groups data without labels
- How to determine the ideal number of clusters
- How to visualize high-dimensional data with PCA
- How to evaluate clustering results using silhouette analysis

---

## 📂 Run the Code

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the Python script
python kmeans_clustering.py
