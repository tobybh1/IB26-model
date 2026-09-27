# Baseline Lasso regression model

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV

# Load data
df = pd.read_csv('team_features.csv')

# Drop rows with missing values
df = df.dropna()

# Define predictors and response
X = df.drop(columns=['score', 'year'])
y = df['score']

# Identify categorical and numeric columns
categorical_features = ['college']
numeric_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# Lasso with automatic alpha selection
model = Pipeline(
    [('preprocessor', preprocessor),
     ('lasso', LassoCV(cv=5, random_state=42, max_iter=20000))]
)

# Fit model
model.fit(X, y)

# Model performance
r2 = model.score(X, y)
print(f'R²: {r2:.4f}')

# Best alpha chosen by cross-validation
print(f'Best alpha: {model.named_steps["lasso"].alpha_:.6f}')

# Get feature names after preprocessing
feature_names = model.named_steps['preprocessor'].get_feature_names_out()

# Get coefficients
coefficients = model.named_steps['lasso'].coef_
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
})

# Sort by absolute coefficient size
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)