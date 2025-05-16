# Stroke_Classifier
This code builds a predictive model to identify patterns in stroke occurrences 

Here’s what it does:
1. Loads a Dataset: It imports stroke-related healthcare data using pandas.
2. Explores the Dataset: Displays basic statistics and structure using .head() and .info().
3. Visualizes Data: Uses matplotlib and seaborn to show the distribution of stroke cases and continuous variables like age, glucose level, and BMI.
4. Cleans and Prepares Data:
       a. Removes unnecessary columns.
       b. Filters out patients under 18.
       c. Handles missing values in BMI.
       d. Applies one-hot encoding to categorical variables.
6. Performs Feature Analysis: Computes correlations between variables and stroke risk.
7. Splits Data for Training and Testing: Uses train_test_split() to divide the dataset
8. Trains a Machine Learning Model: Implements a RandomForestClassifier to predict stroke likelihood.
9. Evaluates Model Performance: Assesses accuracy, precision, recall, and AUC score
10. Analyzes Feature Importance: Identifies which variables contribute most to stroke prediction using SHAP values.
