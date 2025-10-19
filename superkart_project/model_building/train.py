import pandas as pd
import sklearn
import numpy as np
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score # Import regression metrics
# for model serialization
import joblib
import os
import mlflow
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo, RepositoryNotFoundError

# Initialize HfApi for uploading the model file
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/sauravghosh2109/superkart-sales-predictor/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
columns_to_drop = ['Product_Id', 'Store_Id']
df.drop(columns=columns_to_drop, inplace=True)

if "Unnamed: 0" in df.columns:
  df = df.drop(columns=["Unnamed: 0"])

# Identify categorical and numerical columns based on the SuperKart data description
numerical_cols = ['Product_Weight', 'Product_MRP', 'Store_Establishment_Year', 'Product_Allocated_Area'] # Removed target column
nominal_categorical_cols = ['Product_Sugar_Content', 'Product_Type', 'Store_Size', 'Store_Location_City_Type', 'Store_Type']

# Separate target variable
target_col = 'Product_Store_Sales_Total'
X = df.drop(columns=[target_col])
y = df[target_col]

# Create a column transformer for one-hot encoding nominal categorical features and scaling numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_categorical_cols),
        ('scaler', StandardScaler(), numerical_cols) # Add StandardScaler for numerical features
    ],
    remainder='passthrough'
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Get feature names after preprocessing
all_feature_names = preprocessor.get_feature_names_out()

# Convert the processed data back to a DataFrame to maintain column names
X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

print(f"Shape of X_processed: {X_processed.shape}")
print(f"Number of feature names: {len(all_feature_names)}")
print("Feature names:", all_feature_names)

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_processed_df, y, test_size=0.2, random_state=42
)

# Save the processed data splits to CSV files
# Create the directory if it doesn't exist
os.makedirs("superkart_project/model_building", exist_ok=True)
Xtrain.to_csv("superkart_project/model_building/Xtrain.csv",index=False)
Xtest.to_csv("superkart_project/model_building/Xtest.csv",index=False)
ytrain.to_csv("superkart_project/model_building/ytrain.csv",index=False)
ytest.to_csv("superkart_project/model_building/ytest.csv",index=False)


files = ["superkart_project/model_building/Xtrain.csv","superkart_project/model_building/Xtest.csv","superkart_project/model_building/ytrain.csv","superkart_project/model_building/ytest.csv"]

# Define the Hugging Face repository ID for uploading processed data
processed_data_repo_id = "sauravghosh2109/superkart-sales-predictor" # Use the same repo as the raw data

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed_data/{file_path.split('/')[-1]}",  # Save in a 'processed_data' subfolder
        repo_id=processed_data_repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face Hub.")


# Define base XGBoost regressor model
xgb_model = xgb.XGBRegressor(random_state=42) # Changed to XGBRegressor

# Define hyperparameter grid for regression
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# Model training with GridSearchCV - using negative mean squared error as scoring
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1) # Changed scoring to neg_mean_squared_error

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i] # This is the mean negative mean squared error
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cross_val_neg_mse", mean_score) # Log negative MSE
            mlflow.log_metric("std_cross_val_neg_mse", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Evaluate on training and testing sets
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Evaluate using regression metrics
    train_mse = mean_squared_error(ytrain, y_pred_train)
    test_mse = mean_squared_error(ytest, y_pred_test)
    train_r2 = r2_score(ytrain, y_pred_train)
    test_r2 = r2_score(ytest, y_pred_test)


    # Log key metrics
    mlflow.log_metrics({
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

    # Log the model with MLflow
    mlflow.sklearn.log_model(best_model, "superkart_sales_prediction_model")

    print("Model training and evaluation complete. Check MLflow UI for results.")

 # Save the trained model to a joblib file
model_filename = "superkart_sales_prediction_model.joblib"
model_save_path = os.path.join("superkart_project/model_building", model_filename)
joblib.dump(best_model, model_save_path)
print(f"Model saved locally to {model_save_path}")

# Save the preprocessor to a joblib file
preprocessor_filename = "preprocessor.joblib"
preprocessor_save_path = os.path.join("superkart_project/model_building", preprocessor_filename)
joblib.dump(preprocessor, preprocessor_save_path)
print(f"Preprocessor saved locally to {preprocessor_save_path}")


# Upload the saved model file to Hugging Face Hub
# Define the repository ID and type for the model
model_repo_id = "sauravghosh2109/superkart-sales-predictor-model"
model_repo_type = "model"

try:
    api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
    print(f"Model space '{model_repo_id}' already exists. Uploading file.")
except RepositoryNotFoundError:
    print(f"Model space '{model_repo_id}' not found. Creating new space...")
    create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False)
    print(f"Model space '{model_repo_id}' created.")
except Exception as e:
    print(f"Error checking for model space: {e}")


# Upload the model file
try:
    api.upload_file(
        path_or_fileobj=model_save_path,
        path_in_repo=model_filename,
        repo_id=model_repo_id,
        repo_type=model_repo_type,
    )
    print(f"Model file '{model_filename}' uploaded to Hugging Face Hub.")
except Exception as e:
    print(f"Error uploading model file to Hugging Face Hub: {e}")

# Upload the preprocessor file
try:
    api.upload_file(
        path_or_fileobj=preprocessor_save_path,
        path_in_repo=preprocessor_filename,
        repo_id=model_repo_id,
        repo_type=model_repo_type,
    )
    print(f"Preprocessor file '{preprocessor_filename}' uploaded to Hugging Face Hub.")
except Exception as e:
    print(f"Error uploading preprocessor file to Hugging Face Hub: {e}")
