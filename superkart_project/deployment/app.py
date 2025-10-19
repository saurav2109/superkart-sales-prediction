import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Define the Hugging Face repo and filename for the model and preprocessor
HF_MODEL_REPO_ID = "sauravghosh2109/superkart-sales-predictor-model"
MODEL_FILENAME = "superkart_sales_prediction_model.joblib"
PREPROCESSOR_FILENAME = "preprocessor.joblib"

# Get Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")

# Download and load the model
try:
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type="model", use_auth_token=hf_token)
    model = joblib.load(model_path)
    st.success("Model loaded successfully from Hugging Face Hub.")
except Exception as e:
    st.error(f"Error loading model from Hugging Face Hub: {e}")
    st.info(
        f"Attempted to download model '{MODEL_FILENAME}' from repo '{HF_MODEL_REPO_ID}'. Please ensure the model exists in your repo and your Hugging Face token is correctly set up."
    )
    st.stop()

# Download and load the preprocessor
try:
    preprocessor_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=PREPROCESSOR_FILENAME, repo_type="model", use_auth_token=hf_token)
    preprocessor = joblib.load(preprocessor_path)
    st.success("Preprocessor loaded successfully from Hugging Face Hub.")
except Exception as e:
    st.error(f"Error loading preprocessor from Hugging Face Hub: {e}")
    st.info(
        f"Attempted to download preprocessor '{PREPROCESSOR_FILENAME}' from repo '{HF_MODEL_REPO_ID}'. Please ensure the preprocessor exists in your repo and your Hugging Face token is correctly set up."
    )
    st.stop()


# Streamlit UI for Superkart Sales Prediction
st.title("Super Kart Sales Prediction")
st.write(
    """
This application predicts the sales total for a given product in a specific store based on their attributes.
Please enter the product and store details below to get a prediction.
"""
)

# User input fields based on the SuperKart.csv dataset description
st.header("Product and Store Details")

product_weight = st.number_input("Product Weight", min_value=0.0, value=10.0, step=0.1)
product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'No Sugar', 'Regular', 'reg'])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.1, step=0.01)
product_type = st.selectbox("Product Type", ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'])
product_mrp = st.number_input("Product MRP", min_value=0.0, value=100.0, step=0.1)
store_establishment_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2024, value=2000)
store_size = st.selectbox("Store Size", ['High', 'Medium', 'Small'])
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
store_type = st.selectbox("Store Type", ['Departmental Store', 'Food Mart', 'Supermarket Type1', 'Supermarket Type2'])


# Assemble input into DataFrame with original column names
# Ensure the column names match those expected by the preprocessor before transformation
input_data = pd.DataFrame(
    [
        {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Establishment_Year": store_establishment_year,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
        }
    ]
)


if st.button("Predict Sales"):
    if "model" in locals() and model is not None and "preprocessor" in locals() and preprocessor is not None:
        try:
            # Apply the loaded preprocessor to the input data
            # The preprocessor expects the raw data columns (excluding IDs)
            input_data_processed = preprocessor.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_processed)[0]

            # Display result
            st.subheader("Prediction Result:")
            st.success(f"The predicted sales total is: ${prediction:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure the input data format matches the model's expected input.")

    else:
        st.warning("Model or Preprocessor not loaded. Please check the loading steps.")
