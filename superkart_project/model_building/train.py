import pandas as pd
import sklearn
import numpy as np # Import numpy
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer # Changed from make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
import os
import mlflow
# for handling class imbalance
from sklearn.utils.class_weight import compute_class_weight
from huggingface_hub import HfApi # Import HfApi

# Initialize HfApi for uploading the model file
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/sauravghosh2109/superkart-sales-predictor/SuperKart.csv" # Corrected path to the uploaded file
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
# Using the columns mentioned in the data description for dropping
columns_to_drop = ['Product_Id', 'Store_Id']
df.drop(columns=columns_to_drop, inplace=True)


# Identify categorical and numerical columns based on the SuperKart data description
numerical_cols = ['Product_Weight', 'Product_MRP', 'Store_Establishment_Year', 'Product_Store_Sales_Total', 'Product_Allocated_Area'] # Assuming Product_Allocated_Area is numerical

# Nominal categorical columns to be one-hot encoded
nominal_categorical_cols = ['Product_Sugar_Content', 'Product_Type', 'Store_Size', 'Store_Location_City_Type', 'Store_Type']

# Separate target variable
target_col = 'Product_Store_Sales_Total' # Corrected target column for sales prediction
X = df.drop(columns=[target_col])
y = df[target_col]

# Create a column transformer for one-hot encoding nominal categorical features
# Use remainder='passthrough' to keep numerical columns
# Set sparse_output=False to get a dense array
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_categorical_cols)
    ],
    remainder='passthrough'
)


# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Get feature names after preprocessing
# This will include one-hot encoded feature names and original numerical feature names
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
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


# Re-identify numerical features based on the processed data (excluding one-hot encoded columns)
# Assuming columns that are not one-hot encoded are the original numerical columns and CityTier
# We can get these by finding columns in Xtrain that don't have '_TypeofContact', etc. suffixes
# Or, more simply, reuse the numerical_cols list from the prep step if it's consistent.
# Let's assume the numerical_cols list from prep.py is correct for the raw data,
# and the one-hot encoding only applied to nominal_categorical_cols.
# After one-hot encoding and dropping the original nominal categorical columns, the remaining
# columns should be the original numerical columns plus the new one-hot encoded columns.

# We can identify numerical columns in Xtrain_processed by checking their dtype
numerical_features = Xtrain.select_dtypes(include=['float64', 'int64']).columns.tolist()

# The target column is 'ProdTaken'

# Set the class weight to handle class imbalance in the target variable 'ProdTaken'
# Since ytrain is a pandas Series, compute_class_weight can be applied directly
classes = np.array([0, 1]) # Assuming the target classes are 0 and 1, converted to numpy array
class_weights = compute_class_weight('balanced', classes=classes, y=ytrain)
# Convert class weights to a dictionary for XGBoost
class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
# XGBoost's scale_pos_weight expects the weight for the positive class (1) relative to the negative class (0)
scale_pos_weight_value = class_weight_dict[1] if 1 in class_weight_dict else 1 # Default to 1 if only one class exists


# Define base XGBoost model
# Use scale_pos_weight for binary classification imbalance
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight_value, random_state=42, use_label_encoder=False, eval_metric='logloss')


# Define hyperparameter grid - adjusted for a potentially smaller search space initially
# and focusing on parameters relevant for this problem.
param_grid = {
    'n_estimators': [100], # Reduced number of estimators
    'max_depth': [3, 5],     # Reduced max depth options
    'learning_rate': [0.05, 0.1], # Reduced learning rate options
    'subsample': [0.8, 0.9], # Reduced subsample options
    'colsample_bytree': [0.8, 0.9], # Reduced colsample_bytree options
    'gamma': [0], # Reduced gamma options
    'lambda': [1], # Reduced lambda options
    'alpha': [0] # Reduced alpha options
}

# Model training with GridSearchCV - directly on XGBoost model as preprocessing is already done
# We are now training the xgb_model directly on the preprocessed data
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='recall', n_jobs=-1) # Reduced CV folds

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i] # This is the mean recall score
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cross_val_recall", mean_score) # Log recall
            mlflow.log_metric("std_cross_val_recall", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Evaluate on training and testing sets
    # Use the best model to make predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Evaluate using classification report
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log key metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision_1": train_report['1']['precision'] if '1' in train_report else 0, # Handle case where class 1 has no predictions
        "train_recall_1": train_report['1']['recall'] if '1' in train_report else 0,
        "train_f1-score_1": train_report['1']['f1-score'] if '1' in train_report else 0,
        "test_accuracy": test_report['accuracy'],
        "test_precision_1": test_report['1']['precision'] if '1' in test_report else 0, # Handle case where class 1 has no predictions
        "test_recall_1": test_report['1']['recall'] if '1' in test_report else 0,
        "test_f1-score_1": test_report['1']['f1-score'] if '1' in test_report else 0
    })

    # Log the model with MLflow
    mlflow.sklearn.log_model(best_model, "/superkart_sales_prediction_model")

    print("Model training and evaluation complete. Check MLflow UI for results.")

 # Save the trained model to a joblib file
model_filename = "/superkart_sales_prediction_model.joblib"
model_save_path = os.path.join("superKart_project/model_building", model_filename)
joblib.dump(best_model, model_save_path)
print(f"Model saved locally to {model_save_path}")

# Save the preprocessor to a joblib file
preprocessor_filename = "preprocessor.joblib"
preprocessor_save_path = os.path.join("superKart_project/model_building", preprocessor_filename)
joblib.dump(preprocessor, preprocessor_save_path) # Make sure 'preprocessor' is defined and accessible in this scope
print(f"Preprocessor saved locally to {preprocessor_save_path}")


# Upload the saved model file to Hugging Face Hub
# Define the repository ID and type for the model
model_repo_id = "sauravghosh2109/superKart-sales-predictor-model"  # Use a distinct name or confirm the dataset repo name
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
    # Depending on the error, you might want to stop or try creating anyway

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
