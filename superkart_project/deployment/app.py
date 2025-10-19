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
This application predicts whether a customer will purchase the newly introduced SuperKart Sales based on their details and interaction data.
Please enter the customer details below to get a prediction.
"""
)

# User input fields based on the SuperKart.csv dataset description
# Customer Details
st.header("Customer Details")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox(
    "Occupation", ["Salaried", "Freelancer", "Large Business", "Small Business", "Other"] # Added 'Other' to handle potential unseen values
)
gender = st.selectbox("Gender", ["Male", "Female"]) # Assuming only Male and Female based on common data
numberofpersonvisiting = st.number_input(
    "Number of People Visiting", min_value=1, max_value=20, value=1
)
preferredpropertystar = st.selectbox(
    "Preferred Property Star", [1, 2, 3, 4, 5] # Added more options based on common hotel ratings
)
maritalstatus = st.selectbox(
    "Marital Status", ["Single", "Married", "Divorced", "Unmarried"] # Added Unmarried
)
numberoftrips = st.number_input(
    "Number of Trips Annually", min_value=0, max_value=100, value=1
)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
numberofchildrenvisiting = st.number_input(
    "Number of Children Visiting", min_value=0, max_value=10, value=0
)
designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP",
        "Director",
        "Cluster Manager",
        "Area Manager",
        "Senior Executive",
        "Junior Executive",
        "Associate", # Added 'Associate' based on common designations
        "Others" # Added 'Others' to handle potential unseen values
    ],
)
monthlyincome = st.number_input(
    "Monthly Income", min_value=0.0, value=50000.0, step=1000.0
)

# Customer Interaction Data
st.header("Customer Interaction Data")
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
productpitched = st.selectbox(
    "Product Pitched",
    [
        "Basic", # Correcting based on train.py
        "Deluxe",
        "King",
        "Standard",
        "Super Deluxe",
        "Duration", # These might be from a different column or a typo in the original description
        "Accommodation",
        "Food",
        "Transport",
        "Road trip",
        "Cultural",
        "Adventure",
        "Relaxation",
        "Museum",
        "City tour",
        "Cruise",
        "Nature",
        "Other" # Added 'Other' to handle potential unseen values
    ],
)
numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=1)
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0, step=0.1)


# Assemble input into DataFrame with original column names and dtypes
input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": typeofcontact,
            "CityTier": citytier,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": numberofpersonvisiting,
            "PreferredPropertyStar": preferredpropertystar,
            "MaritalStatus": maritalstatus,
            "NumberOfTrips": numberoftrips,
            "Passport": passport,
            "OwnCar": owncar,
            "NumberOfChildrenVisiting": numberofchildrenvisiting,
            "Designation": designation,
            "MonthlyIncome": monthlyincome,
            "PitchSatisfactionScore": pitchsatisfactionscore,
            "ProductPitched": productpitched,
            "NumberOfFollowups": numberoffollowups,
            "DurationOfPitch": durationofpitch,
        }
    ]
)


if st.button("Predict Purchase"):
    if "model" in locals() and model is not None and "preprocessor" in locals() and preprocessor is not None:
        try:
            # Apply the loaded preprocessor to the input data
            input_data_processed = preprocessor.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_processed)[0]

            # Display result
            result = "Yes, likely to purchase" if prediction == 1 else "No, not likely to purchase"
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success(f"The model predicts: **{result}**")
            else:
                st.info(f"The model predicts: **{result}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure the input data format matches the model's expected input.")

    else:
        st.warning("Model or Preprocessor not loaded. Please check the loading steps.")
