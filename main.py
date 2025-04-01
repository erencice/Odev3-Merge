import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('data.csv')

# Check for missing values
print("Missing Values Before Cleaning:\n", data.isnull().sum())

# Handling missing values by replacing them with the mean of the column
imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Check if missing values are handled
print("\nMissing Values After Cleaning:\n", data_filled.isnull().sum())

# Save the cleaned data
data_filled.to_csv('cleaned_data.csv', index=False)

# Creating a new feature 'log_transformed_feature' as the log of an existing feature 'feature_1'
data_filled['log_transformed_feature'] = data_filled['feature_1'].apply(lambda x: np.log(x + 1))

# Converting a categorical feature to numeric using label encoding
data_filled['category_encoded'] = data_filled['category'].astype('category').cat.codes

# Drop unnecessary columns
data_filled = data_filled.drop(['column_to_drop'], axis=1)

# Save the updated data with features
data_filled.to_csv('updated_data.csv', index=False)
