import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your dataset
df_red = pd.read_csv('../../resources/winequality-red.csv',delimiter=";")
df_white = pd.read_csv('../../resources/winequality-white.csv',delimiter=";")

# quality_label = low if value <= 5 quality_label = medium if 5 < value <= 7 quality_label = high if value > 7
df_red['quality_label'] = df_red['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
df_red['quality_label'] = pd.Categorical(df_red['quality_label'], categories=['low', 'medium', 'high'])

df_white['quality_label'] = df_white['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
df_white['quality_label'] = pd.Categorical(df_white['quality_label'], categories=['low', 'medium', 'high'])


# Concatenate the two datasets
df_wines = pd.concat([df_red, df_white])

# Feature engineering Encode the Labels
label_encoder = LabelEncoder()
df_wines['quality_label'] = label_encoder.fit_transform(df_wines['quality_label'])

# Define the model
rf = RandomForestClassifier()

X = df_wines.drop(['quality', 'quality_label'],axis=1)
y = df_wines['quality_label']

# Split datasets
X_train, X_test, y_train, y_test=train_test_split(X , y, test_size=0.30, random_state=42)

# Create a pipeline with a scaler and a placeholder for the model
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Scale features
    ('rf', RandomForestClassifier())  # Step 2: Random Forest classifier
])


# Create a parameter grid for GridSearchCV
param_grid = {
    'rf__n_estimators': [50, 100, 200],  # Number of trees in the forest
    'rf__max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'rf__min_samples_split': [2, 5, 10]  # Minimum number of samples required to split an internal node
}

# Perform Grid Search with accuracy as the scoring metric
grid_search = GridSearchCV(model_pipeline,
                           param_grid=param_grid,
                           cv=5,
                           verbose=2,
                           scoring='accuracy')
# train
grid_search.fit(X, y)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy score: {:.2f}".format(grid_search.best_score_))

# Get the best model from Grid Search
best_model = grid_search.best_estimator_

# Assuming best_model is your trained RandomForestClassifier
feature_importances = best_model.named_steps['rf'].feature_importances_

# Get feature names from the original dataset
feature_names = X.columns  # Assuming you have loaded iris dataset previously

# Create a list of (feature_name, feature_importance) pairs
feature_importance_list = list(zip(feature_names, feature_importances))

# Sort feature_importance_list by feature_importance in descending order
feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

# Print feature importances
for feature_name, importance in feature_importance_list:
    print(f"{feature_name}: {importance}")

# Save the trained pipeline model to a file using joblib
model_filename = '../deploy/wine_quality_predictor_model.pkl'
joblib.dump(best_model, model_filename)

print(f"Model has been saved to {model_filename}")


if __name__ == '__main__':
    pass
