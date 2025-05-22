from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

# Define parameter grid for comprehensive search
param_grid = {
    # Tree-specific parameters
    'max_depth': [1,2,3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    
    # Boosting parameters
    'learning_rate': [0.01, 0.1, 0.3,0.001],
    'n_estimators': [11,31,100, 217, 313,511, 711,911, 1011],
    
    # Sampling parameters
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
xgb = XGBClassifier(
    objective='binary:logistic',  # for classification
    random_state=42,           # for reproducibility
    n_jobs=-1                  # use all available cores
)
# Initialize XGBoost classifier


# Setup GridSearchCV with cross-validation
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,                      # 5-fold cross-validation
    scoring='accuracy',        # optimize for accuracy
    verbose=2,                 # print progress
    n_jobs=-1,                # use all available cores
    return_train_score=True   # to analyze stability
)

# Fit the grid search
grid_search.fit(X, df['direction'])

# Print results
print("\n=== Grid Search Results ===")
print(f"\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

print(f"\nBest Cross-Validation Score: {grid_search.best_score_:.4f}")

# Calculate stability metrics
cv_results = grid_search.cv_results_
best_index = grid_search.best_index_
cv_scores = cv_results['split0_test_score'][best_index], cv_results['split1_test_score'][best_index], \
           cv_results['split2_test_score'][best_index], cv_results['split3_test_score'][best_index], \
           cv_results['split4_test_score'][best_index]

print("\nStability Metrics:")
print(f"Standard Deviation of CV Scores: {np.std(cv_scores):.4f}")
print(f"Min CV Score: {np.min(cv_scores):.4f}")
print(f"Max CV Score: {np.max(cv_scores):.4f}")
