# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("heart.csv")

# Split features and target
X = df.drop(columns="target")
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dtree, filled=True, feature_names=X.columns, class_names=["No Disease", "Disease"])
plt.title("Full Decision Tree")
plt.show()

# Limited depth tree to reduce overfitting
dtree_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree_limited.fit(X_train, y_train)
train_acc = accuracy_score(y_train, dtree_limited.predict(X_train))
test_acc = accuracy_score(y_test, dtree_limited.predict(X_test))
print(f"Decision Tree (depth=3) - Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest - Train Accuracy: {rf_train_acc:.2f}, Test Accuracy: {rf_test_acc:.2f}")

# Feature Importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importances.plot(kind="barh", figsize=(10, 6), title="Random Forest Feature Importances")
plt.xlabel("Importance")
plt.show()

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Random Forest 5-Fold CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
