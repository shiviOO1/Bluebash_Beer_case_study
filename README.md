# Beer Dataset Analysis and Prediction

This project analyzes a beer review dataset and predicts overall ratings using machine learning models like Random Forest and Linear Regression. 

## Features
- Data cleaning and preprocessing.
- Text feature extraction using TF-IDF.
- Machine learning model training and evaluation.
- Data visualization for insights.

## Models and Metrics
1. **Random Forest Regressor**:
   - MAE: (Add actual value)
   - MSE: (Add actual value)
   - R²: (Add actual value)

2. **Linear Regression**:
   - MAE: (Add actual value)
   - MSE: (Add actual value)
   - R²: (Add actual value)

## Hyperparameter Tuning
If you'd like to try optimizing the models, use the following code:

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best Parameters for Random Forest:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)
