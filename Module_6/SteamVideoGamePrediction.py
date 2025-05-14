# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('dataset.csv')

# Remove rows with any missing values
df = df.dropna()

# Preprocessing: Handle outliers in positivity_ratio (cap at 95th percentile)
positivity_cap = df['positivity_ratio'].quantile(0.95)
df['positivity_ratio'] = df['positivity_ratio'].clip(upper=positivity_cap)

# Feature Engineering: One-hot encode tags
# Clean tags column (replace '|' with ',')
df['tags'] = df['tags'].str.replace('|', ',')
# Split tags and create binary columns for top 10 most frequent tags
all_tags = df['tags'].str.split(',', expand=True).stack().str.strip()
top_tags = all_tags.value_counts().head(10).index
for tag in top_tags:
    df[f'tag_{tag}'] = df['tags'].apply(lambda x: 1 if tag in str(x).split(',') else 0)

# Rename columns for clarity
df = df.rename(columns={
    'to_beat_main': 'main_story_hours',
    'to_beat_extra': 'extra_hours',
    'to_beat_completionist': 'completionist_hours'
})

# Select features and target
features = ['main_story_hours', 'extra_content_length', 'metacritic_rating', 
            'reviewer_rating'] + [f'tag_{tag}' for tag in top_tags]
X = df[features]
y = df['positivity_ratio']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature Importance Plot
feature_importances = pd.Series(rf_model.feature_importances_, index=features)
feature_importances = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh', color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance for Predicting Game Popularity')
plt.gca().invert_yaxis()
plt.savefig('feature_importance.png')
plt.close()

# Identify 5 samples with largest prediction errors
test_indices = X_test.index
errors = np.abs(y_test - y_pred)
error_df = pd.DataFrame({
    'Game Title': df.loc[test_indices, 'name'],
    'Actual Positivity Ratio': y_test,
    'Predicted Positivity Ratio': y_pred,
    'Absolute Error': errors
})
top_5_errors = error_df.sort_values(by='Absolute Error', ascending=False).head(5)
print("\nTop 5 Incorrect Predictions:")
print(top_5_errors.to_string(index=False))

# Save top 5 errors to a CSV for reference
top_5_errors.to_csv('top_5_incorrect_predictions.csv', index=False)

# Save top 5 accurate predictions for comparison
top_5_accurate = error_df.sort_values(by='Absolute Error').head(5)
print("\nTop 5 Accurate Predictions:")
print(top_5_accurate.to_string(index=False))
top_5_accurate.to_csv('top_5_accurate_predictions.csv', index=False)