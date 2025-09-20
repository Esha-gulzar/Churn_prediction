import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl
import sklearn
from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()
ss=StandardScaler()

model=pkl.load(open("C:\\Users\\A I TECH\Desktop\\folder\\Linear_Churn_model","rb"))
#build an app
st.title("Scikit-learn Logistic Regression Model for Churn prediction")
gender=st.selectbox("Select Gender",options=['Female','Male'])
seniorCitizen=st.selectbox("Select Senior-Citizen",options=['Yes','No'])
partner=st.selectbox("Select Have Partner at work",options=['Yes','No'])
dependents=st.selectbox("Select Have Dependents",options=['Yes','No'])
tenure=st.text_input("Enter your tenure")
phoneService=st.selectbox("Select PhoneService",options=['Yes','No'])
multipleLines=st.selectbox("Select MultipleLines",options=['Yes','No','NoPhoneService'])
contract=st.selectbox("Select Contract",options=['One-year','Two-year','Month-to-month'])
totalCharges=st.text_input("Enter your TotalCharges")
#predictive function
def predictive(gender, seniorCitizen, partner, dependents, tenure, phoneService, multipleLines, contract, totalCharges):
    data = {
       'gender': [gender],
       'SeniorCitizen': [seniorCitizen],
       'Partner': [partner],
       'Dependents': [dependents],
       'tenure': [tenure],
       'PhoneService': [phoneService],
       'MultipleLines': [multipleLines],
       'Contract': [contract],
       'TotalCharges': [totalCharges] 
    }
    # Fixed indentation - df1 definition was on same line as dictionary closing brace
    df1 = pd.DataFrame(data)
    
    bin_col = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'Contract'] # lists of column to be encoded
    for col in bin_col: # for loop to encode all the columns and avoid all the more code of statements
        df1[col] = le.fit_transform(df1[col]) 
    
    df1 = ss.fit_transform(df1)
    result = model.predict(df1).reshape(1, -1)
    return result[0]
#churn prevention tips
Churrn_data={
 "churn_prevention_tips" : [
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
#retention tips
retention_data = {
    "customer_retention_tips": [
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
Churrn_data_df=pd.DataFrame(Churrn_data)
retention_data_df=pd.DataFrame(retention_data)



#button
if st.button("Let's Predict"):
 result=predictive(gender,seniorCitizen,partner,dependents,tenure,phoneService,multipleLines,contract,totalCharges)
 if result==1:
    st.write("Churn")
    st.title('Sorry, the customer has churned')
    st.write("Here are tips for you :)")
    st.dataframe(Churrn_data_df,height=400,width=800)

 else:
    
    st.write("Not Churn")
    st.title("ooh thats cool!Our customer has not churned")
    st.write("Here are tips for retention")
    st.dataframe(retention_data_df,height=400,width=800)
    