from huggingface_hub import HfApi
import os
import shutil # Import shutil to potentially move files

# Define paths
model_building_dir = "superkart_project/model_building"
deployment_dir = "superkart_project/deployment"
preprocessor_filename = "preprocessor.joblib"
preprocessor_src_path = os.path.join(model_building_dir, preprocessor_filename)
preprocessor_dest_path = os.path.join(deployment_dir, preprocessor_filename)

# Move the preprocessor file from model_building to deployment before uploading
# This logic should ideally be in the script that prepares the deployment folder,
# but for this isolated step, we'll include it here.
if os.path.exists(preprocessor_src_path):
    shutil.move(preprocessor_src_path, preprocessor_dest_path)
    print(f"Moved {preprocessor_filename} from {model_building_dir} to {deployment_dir}")
else:
    print(f"Warning: {preprocessor_filename} not found in {model_building_dir}. Ensure it was saved during training.")


api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the Hugging Face Space repository ID
# Replace with your actual Space repo ID if different from the dataset repo
space_repo_id = "sauravghosh2109/superkart-sales-predictor"

api.upload_folder(
    folder_path="superKart_project/deployment",     # the local folder containing your files (app.py and preprocessor.joblib)
    repo_id=space_repo_id,          # the target Space repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
    commit_message="Upload updated Streamlit app and preprocessor for deployment"
)
print(f"Uploaded contents of superKart_project/deployment to Hugging Face Space '{space_repo_id}'.")
