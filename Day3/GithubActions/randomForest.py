# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import time
import datetime


print("Random Forest Regressor Start: " + datetime.datetime.fromtimestamp(time.time()).strftime("%H:%M:%S"))

# Load the Wine Quality dataset
wine_data = load_wine()

# Create a Pandas DataFrame from the dataset
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Add the target values to the DataFrame as a new column
df["target"] = wine_data.target

# Inspect the first few rows of the DataFrame
#df.head()


# Check the information about the dataset
#df.info()

# Separate features (X) and target variable (y)
X = df.drop("target", axis=1)
y = df["target"]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

# Create a StandardScaler object
scaler = StandardScaler()

# Apply standard scaling to selected columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.fit_transform(X_test[numerical_columns])
# Display the scaled dataset
#X_train



# Define the Random Forest Regressor model with random_state
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

# Define the search space for hyperparameters
param_space = {
    'n_estimators': Integer(50, 500),  # Number of trees
    'max_depth': Integer(1, 20)  # Maximum depth of the trees
}

# Perform Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=rf_model,
    search_spaces=param_space,
    scoring='r2',
    cv=5,
    n_jobs=-1
)

# Fit the model
bayes_search.fit(X_train, y_train)


print("Random Forest Regressor End: " + datetime.datetime.fromtimestamp(time.time()).strftime("%H:%M:%S"))

# Best hyperparameters
best_params = bayes_search.best_params_
best_rf_model = bayes_search.best_estimator_

# Predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Model evaluation
r2 = r2_score(y_test,y_pred)

# Display the results
print(f"\n[RF] Best Hyperparameters: {best_params}")
print(f"[RF] R-squared (r2) on Test Set: {r2:.4f}")