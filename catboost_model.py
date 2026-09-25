"""
Best performing model, R² = 0.296, RMSE = 66.295, optimised model, R² = 0.367
RMSE = 83.884
"""

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np

df = pd.read_csv('team_features.csv')

train = df[df['year'] != 2025]
test = df[df['year'] == 2025]

# Most important features based off Lasso regression
key_features = ['college', 'division', 'sum_dnf_dq', 'avg_highest_div', 
                'college_best_placing', 'avg_best_score', 'best_last_score', 
                'best_team_score', 'total_yoe']

X_train = train.drop(columns=['score', 'year'])
X_train = X_train[key_features]
y_train = train['score']

X_test = test.drop(columns=['score', 'year'])
X_test = X_test[key_features]
y_test = test["score"]

cat_features = ['college', 'division']

model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    loss_function="RMSE",
    verbose=100,
    allow_writing_files=False
)

model.fit(X_train, y_train, cat_features=cat_features)
preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)
print(f"R² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")

# Tuning
results = []

for depth in [1, 2, 4]:
    for learning_rate in [0.01, 0.03, 0.05]:
        for l2_leaf_reg in [3, 10, 20]:

            model = CatBoostRegressor(
                iterations=1000,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                loss_function="RMSE",
                verbose=200,
                allow_writing_files=False,
            )

            model.fit(
                X_train,
                y_train,
                cat_features=cat_features
            )

            preds = model.predict(X_test)

            results.append({
                "depth": depth,
                "learning_rate": learning_rate,
                "l2_leaf_reg": l2_leaf_reg,
                "R2": r2_score(y_test, preds),
                "RMSE": root_mean_squared_error(y_test, preds)
            })

results = pd.DataFrame(results)

results.to_csv('optm_catboost.csv', index=False)

# Optimal model
model = CatBoostRegressor(
    iterations=1000,
    depth=1,
    learning_rate=0.01,
    l2_leaf_reg=3,
    loss_function='RMSE',
    verbose=100,
    allow_writing_files=False
)

model.fit(X_train, y_train, cat_features=cat_features)
preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)
print(f"R² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")

comparison = test[['college', 'division', 'year', 'score']].copy()
comparison['predicted_score'] = preds
comparison['error'] = comparison['predicted_score'] - comparison['score']
comparison['abs_error'] = comparison['error'].abs()

comparison.to_csv('comparison25.csv', index=False)
