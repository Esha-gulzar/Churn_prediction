import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ✅ Load model AND scaler from pickle
model = pkl.load(open("Linear_Churn_model.pkl", "rb"))
scaler = pkl.load(open("scaler.pkl", "rb"))  # ✅ load the saved scaler, not a new one

# Build the app
st.title("Scikit-learn Logistic Regression Model for Churn Prediction")

gender = st.selectbox("Select Gender", options=['Female', 'Male'])
seniorCitizen = st.selectbox("Select Senior Citizen", options=['Yes', 'No'])
partner = st.selectbox("Do you have a Partner?", options=['Yes', 'No'])
dependents = st.selectbox("Do you have Dependents?", options=['Yes', 'No'])
tenure = st.text_input("Enter your Tenure (months)")
phoneService = st.selectbox("Phone Service?", options=['Yes', 'No'])
multipleLines = st.selectbox("Multiple Lines?", options=['Yes', 'No', 'No phone service'])  # ✅ fixed option text
contract = st.selectbox("Select Contract", options=['Month-to-month', 'One year', 'Two year'])  # ✅ fixed option text
totalCharges = st.text_input("Enter your Total Charges")


# ✅ Predictive function using manual encoding (not LabelEncoder on single rows)
def predictive(gender, seniorCitizen, partner, dependents, tenure,
               phoneService, multipleLines, contract, totalCharges):

    # ✅ Manual encoding - consistent with training data
    gender_enc = 1 if gender == 'Male' else 0
    seniorCitizen_enc = 1 if seniorCitizen == 'Yes' else 0
    partner_enc = 1 if partner == 'Yes' else 0
    dependents_enc = 1 if dependents == 'Yes' else 0
    phoneService_enc = 1 if phoneService == 'Yes' else 0
    multipleLines_enc = {'Yes': 1, 'No': 0, 'No phone service': 2}.get(multipleLines, 0)
    contract_enc = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}.get(contract, 0)

    # ✅ Safe conversion of text inputs
    try:
        tenure_val = float(tenure)
    except ValueError:
        st.error("Please enter a valid number for Tenure")
        return None

    try:
        totalCharges_val = float(totalCharges)
    except ValueError:
        st.error("Please enter a valid number for Total Charges")
        return None

    data = [[gender_enc, seniorCitizen_enc, partner_enc, dependents_enc,
             tenure_val, phoneService_enc, multipleLines_enc, contract_enc, totalCharges_val]]

    df1 = pd.DataFrame(data, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                                      'tenure', 'PhoneService', 'MultipleLines', 'Contract', 'TotalCharges'])

    df1 = scaler.transform(df1)  # ✅ transform only, not fit_transform
    result = model.predict(df1)
    return result[0]  # ✅ removed wrong reshape


# Churn prevention tips
churn_data = {
    "Churn Prevention Tips": [
        "Improve customer service and support",
        "Offer personalized plans and promotions",
        "Enhance network quality and coverage",
        "Provide value-added services",
        "Monitor usage patterns for early warning signs",
        "Engage customers through targeted communications",
        "Offer loyalty rewards or retention incentives",
        "Gather and act on customer feedback",
        "Simplify billing and payment processes",
        "Proactively address customer issues"
    ]
}

# Retention tips
retention_data = {
    "Customer Retention Tips": [
        "Build strong customer relationships",
        "Deliver consistent value to customers",
        "Personalize interactions and communications",
        "Reward loyal customers",
        "Respond promptly to customer issues",
        "Offer relevant upsell or cross-sell opportunities",
        "Use feedback to improve customer experience",
        "Maintain regular but not intrusive communication",
        "Show appreciation for customer loyalty",
        "Continuously improve service quality"
    ]
}

churn_df = pd.DataFrame(churn_data)
retention_df = pd.DataFrame(retention_data)

# Predict button
if st.button("Let's Predict"):
    if not tenure or not totalCharges:
        st.warning("Please fill in all fields before predicting.")  # ✅ input validation
    else:
        result = predictive(gender, seniorCitizen, partner, dependents,
                            tenure, phoneService, multipleLines, contract, totalCharges)
        if result is not None:
            if result == 1:
                st.error("Sorry, the customer has churned! 😟")
                st.write("Here are some churn prevention tips for you:")
                st.dataframe(churn_df, height=400, width=800)
            else:
                st.success("Great! The customer has NOT churned! 😊")
                st.write("Here are some retention tips to keep them happy:")
                st.dataframe(retention_df, height=400, width=800)