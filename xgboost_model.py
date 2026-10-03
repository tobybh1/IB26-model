"""
Worst performing, R² = 0.275, RMSE = 89.830
NOTE THAT THIS HAS NOT BEEN UPDATED SINCE CHANGING FROM PREDICTING SCORE TO 
PLACING
"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, root_mean_squared_error

df = pd.read_csv('team_features.csv')

college_enc = LabelEncoder()
div_enc = LabelEncoder()

df['college'] = college_enc.fit_transform(df['college'])
df['division'] = div_enc.fit_transform(df['division'])

train = df[~df['year'].isin([2025, 2026])]
test = df[df['year'] == 2025]

# XGBoost does better with all features
X_train = train.drop(columns=['score', 'year'])
#X_train = X_train[key_features]
y_train = train['score']

X_test = test.drop(columns=['score', 'year'])
#X_test = X_test[key_features]
y_test = test["score"]

model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)
print(f"RÂ² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")
