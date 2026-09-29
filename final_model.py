"""
Best performing model, R² = 0.308, RMSE = 87.756, optimised model, R² = 0.429
RMSE = 79.714
"""

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

df = pd.read_csv('team_features.csv')

train = df[df['year'] < 2025]
val = df[df['year'] == 2025]
test = df[df['year'] == 2026]

X_train = train.drop(columns=['score', 'year'])
y_train = train['score']

X_val = val.drop(columns=['score', 'year'])
y_val = val['score']

X_test = test.drop(columns=['score', 'year'])
y_test = test['score']

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
preds = model.predict(X_val)

r2 = r2_score(y_val, preds)
rmse = root_mean_squared_error(y_val, preds)
print(f"R² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")

importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.get_feature_importance()
}).sort_values('importance', ascending=False)

# Testing for features
importance_order = importance['feature'].tolist()

results = []

for n in [5, 10, 15, 20, 25, len(importance_order)]:

    features = importance_order[:n]

    X_train_sub = X_train[features]
    X_val_sub = X_val[features]

    cat_sub = [f for f in cat_features if f in features]

    model_sub = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        loss_function='RMSE',
        verbose=False,
        allow_writing_files=False,
        random_seed=42
    )

    model_sub.fit(
        X_train_sub,
        y_train,
        cat_features=cat_sub
    )

    preds = model_sub.predict(X_val_sub)

    results.append({
        'n_features': n,
        'R2': r2_score(y_val, preds),
        'RMSE': root_mean_squared_error(y_val, preds)
    })

results = pd.DataFrame(results)

# Tuning
# These are the 5 best features from above
key_features = ['sum_dnf_dq', 'college', 'division', 'avg_best_rog', 
                'best_rog']

X_train = train.drop(columns=['score', 'year'])
X_train = X_train[key_features]
y_train = train['score']

X_val = val.drop(columns=['score', 'year'])
X_val = X_val[key_features]
y_val = val['score']

results = []

for depth in [1, 2, 4]:
    for learning_rate in [0.01, 0.001, 0.005]:
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

            preds = model.predict(X_val)

            results.append({
                "depth": depth,
                "learning_rate": learning_rate,
                "l2_leaf_reg": l2_leaf_reg,
                "R2": r2_score(y_val, preds),
                "RMSE": root_mean_squared_error(y_val, preds)
            })

results = pd.DataFrame(results)

# Optimal model
model = CatBoostRegressor(
    iterations=1000,
    depth=2,
    learning_rate=0.005,
    l2_leaf_reg=3,
    loss_function='RMSE',
    verbose=100,
    allow_writing_files=False
)

model.fit(X_train, y_train, cat_features=cat_features)
preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)

results = X_test.copy()
results['actual'] = y_test
results['predicted'] = preds

rmse_by_division = (
    results.groupby('division')
    .apply(lambda x: root_mean_squared_error(x['actual'], x['predicted']))
    .reset_index(name='RMSE')
)

print(f"R² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")

comparison = test[['college', 'division', 'year', 'score']].copy()
comparison['predicted_score'] = preds
comparison['error'] = comparison['predicted_score'] - comparison['score']
comparison['abs_error'] = comparison['error'].abs()

comparison.to_csv('comparison25.csv', index=False)
