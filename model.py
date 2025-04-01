import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Load the cleaned and updated data from the 'main' branch
data = pd.read_csv('updated_data.csv')

# Define features and target
X = data.drop('target', axis=1)
y = data['target']

# Load the pre-trained model (this is just for demonstration; replace with actual model)
model = joblib.load('model.pkl')

# Make predictions on the test data
y_pred = model.predict(X)

# Evaluate the model's performance
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Define the model
model = LogisticRegression(max_iter=200)

# Define the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Save the best model
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
