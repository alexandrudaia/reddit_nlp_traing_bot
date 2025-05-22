class_weights = {0: 2, 1: 1}  # Giving more weight to class 0

# Initialize logistic regression with custom class weights
model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print( f1_score(y_test, y_pred))
 
 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

 
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, df['direction'], test_size=0.2, 
                                                    random_state=42)

class_weights = {0: 1, 1: 2.5}
# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': class_weights,
    'max_iter': [1000]
}

# Initialize logistic regression
log_reg = LogisticRegression()

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,                # 5-fold cross-validation
    scoring='accuracy',  # metric to optimize
    n_jobs=1,           # number of parallel jobs
    verbose=1
)

# Perform grid search
grid_search.fit(X_train, y_train)

# Print results
print("\nBest parameters found:")
print(grid_search.best_params_)
print("\nBest cross-validation score:")
print(f"{grid_search.best_score_:.4f}")

# Create DataFrame with all results
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values(by='rank_test_score')

# Display top 5 parameter combinations
print("\nTop 5 parameter combinations:")
cols_to_show = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
print(results[cols_to_show].head())

# Test best model on test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"\nTest set score with best model: {test_score:.4f}")
