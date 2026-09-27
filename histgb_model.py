"""
2nd best, R² = 0.286, RMSE = 89.097
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

df = pd.read_csv('team_features.csv')

# Encode categoricals
college_enc = LabelEncoder()
div_enc = LabelEncoder()

df['college'] = college_enc.fit_transform(df['college'])
df['division'] = div_enc.fit_transform(df['division'])

train = df[df['year'] != 2025]
test = df[df['year'] == 2025]

# Most important features based off Lasso regression
key_features = ['college', 'division', 'sum_dnf_dq', 'avg_highest_div', 
                'college_best_placing', 'avg_best_score', 'best_last_score', 
                'best_team_score', 'total_yoe']

X_train = train.drop(columns=['score', 'year'])
#X_train = train[key_features]
y_train = train['score']

X_test = test.drop(columns=['score', 'year'])
#X_test = test[key_features]
y_test = test['score']

model = HistGradientBoostingRegressor(
    learning_rate=0.03,
    max_depth=2,
    max_iter=1000,
    min_samples_leaf=5,
    l2_regularization=3,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)

print(f"RÂ² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")
