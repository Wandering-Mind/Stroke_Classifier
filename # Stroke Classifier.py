# Stroke Classifier 

# Machine Learning with Python

# Step 1: Load the stroke dataset

import pandas as pd
url = "https://raw.githubusercontent.com/msabanluc/SalonFirstMLPython/refs/heads/main/healthcare-dataset-stroke-data.csv"
df = pd.read_csv(url)

# Step 2: Explore the dataset

print("First 5 rows:")
display(df.head())

df.info()

# 2a. Visualize class distribution with counts and percentages

import matplotlib.pyplot as plt
import seaborn as sns

class_counts = df['stroke'].value_counts() # Create a count of each value in "stroke" column (our target)

print("\nClass distribution:")
display(class_counts) # Display that count

class_percent = df['stroke'].value_counts(normalize=True) * 100 # Calculate the percentage of each class

# Create figure plotting class counts
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution')
plt.ylabel('Count')
plt.xlabel('Class')

# Add percentages on top of bars
for i, (count, percent) in enumerate(zip(class_counts, class_percent)):
    ax.text(i, count + 5, f'{percent:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.ylim(0, max(class_counts) * 1.15)
plt.show()

# 2b. Visualize distribution of continuous features

continuous_features = ['age', 'avg_glucose_level', 'bmi']

for col in continuous_features: #For each feature (age, avg_glucose_level, bmi),
    plt.figure(figsize=(6, 4)) #Make a figure
    sns.histplot(data=df, x=col, kde=True) #In the figure, plot a histogram of the given feature on that figure
    plt.title(f"Distribution of {col}") #Assign title to figure
    plt.grid(True) #Plot gridlines
    plt.show() #Show the figure as output

# 3. Clean and prepare data for modeling

# drop ID column if it exists
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Remove patients under 18, since stroke is very rare in children and BMI is less meaningful
df = df[df['age'] >= 18]

# Handle missing BMI
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# One-hot encode categoricals
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and target
X = df_encoded.drop(columns='stroke')
y = df_encoded['stroke']

# 4. Feature Check

# 4a. Absolute feature correlation with stroke

# Compute absolute correlation between each feature and stroke label
cor_scores = X.corrwith(y).abs().sort_values(ascending=False)

# Show top 10 features
top_features = cor_scores.head(10)
print("Top features correlated with stroke:")
display(top_features)

# Visualize as bar plot
top_features.plot(kind='barh')
plt.title("Top Features Correlated with Stroke (absolute value)")
plt.xlabel("Absolute Pearson Correlation")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True)
plt.show()

# 4b. Plot stroke vs non-stroke cases by age and glucose level.

plt.figure(figsize=(8, 6))

# Plot non-stroke cases (gray background)
sns.scatterplot(data=df[df['stroke'] == 0], x='age', y='avg_glucose_level',
                color='lightgray', alpha=0.4, label='No Stroke')

# Plot stroke cases on top
sns.scatterplot(data=df[df['stroke'] == 1], x='age', y='avg_glucose_level',
                color='blue', alpha=0.8, label='Stroke')

plt.title('Stroke vs Age and Glucose Level')
plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.grid(True)
plt.legend()
plt.show()

# Step 5: Split into train/test
from sklearn.model_selection import train_test_split

# Split with stratification to preserve stroke/no-stroke ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # means 70% is training # 30% is testing
    stratify=y,
    random_state=2025
)

# Step 6: Train Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None, # switch in the integer '5' to see the different results
    min_samples_split=2,
    min_samples_leaf=1,

    random_state=2025,
    class_weight='balanced_subsample'
)
rf_model.fit(X_train, y_train)

# Step 7: Predict and Evaluate on Train Set

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score
)

y_train_proba = rf_model.predict_proba(X_train)[:, 1]

thresh = 0.5

y_train_pred = (y_train_proba >= thresh).astype(int)

print("Train Set Evaluation")
train_acc = accuracy_score(y_train, y_train_pred)
train_prec = precision_score(y_train, y_train_pred)
train_rec = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

print(f"Accuracy: {train_acc:.3f}")
print(f"Precision: {train_prec:.3f}")
print(f"Recall: {train_rec:.3f}")
print(f"F1 Score: {train_f1:.3f}")
print(f"ROC AUC: {train_auc:.3f}")

# Step 8: Predict on Test Set
y_proba = rf_model.predict_proba(X_test)[:, 1]

thresh = 0.5

y_pred = (y_proba >= thresh).astype(int)

# Step 9: Evaluate model performance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC and PR Curves
fpr, tpr, _ = roc_curve(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(12, 5))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# PR Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.tight_layout()
plt.show()

# 9a. Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = auc(*roc_curve(y_test, y_proba)[:2])

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")

# 9b. Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values()
plt.figure(figsize=(8, 6))
importances.plot(kind='barh')
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.grid(True)
plt.tight_layout()
plt.show()

# 9c. SHAP

!pip install shap --quiet
import shap
shap.initjs()

explainer = shap.Explainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap_c1 = shap_values[:, :, 1]

# Summary plot (for class 1: malignant)
shap.summary_plot(shap_c1, X_test, plot_type="bar")

# Optional: full summary plot with beeswarm layout
shap.summary_plot(shap_c1, X_test)

