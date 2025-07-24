import os
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------
# Custom Transformer Definition
# ----------------------------
class TotalVisitingAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['Total_visiting'] = X['NumberOfPersonVisiting'] + X['NumberOfChildrenVisiting']
        return X

# ----------------------------
# Load the Model Pipeline
# ----------------------------
@st.cache_resource
def load_model():
    model_path = 'decision_tree_pipeline.joblib'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: `{model_path}`")
        st.stop()
    return load(model_path)

model = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧳 Travel Product Purchase Prediction")
st.markdown("Enter customer information to predict whether they will take the travel product.")

def user_input_features():
    Age = st.number_input('Age', min_value=18, max_value=100, value=30)
    TypeofContact = st.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'])
    CityTier = st.selectbox('City Tier', [1, 2, 3])
    DurationOfPitch = st.number_input('Duration Of Pitch', min_value=0, max_value=20, value=5)
    Occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Free Lancer'])
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    NumberOfPersonVisiting = st.number_input('Number Of Person Visiting', min_value=0, max_value=10, value=1)
    NumberOfFollowups = st.number_input('Number Of Followups', min_value=0, max_value=10, value=1)
    ProductPitched = st.selectbox('Product Pitched', ['Basic', 'Deluxe'])
    PreferredPropertyStar = st.selectbox('Preferred Property Star', [1, 2, 3, 4, 5])
    MaritalStatus = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    NumberOfTrips = st.number_input('Number Of Trips', min_value=0, max_value=50, value=1)
    Passport = st.selectbox('Passport', [0, 1])
    PitchSatisfactionScore = st.number_input('Pitch Satisfaction Score', min_value=1, max_value=5, value=3)
    OwnCar = st.selectbox('Own Car', [0, 1])
    NumberOfChildrenVisiting = st.number_input('Number Of Children Visiting', min_value=0, max_value=10, value=0)
    Designation = st.selectbox('Designation', ['Executive', 'Manager'])
    MonthlyIncome = st.number_input('Monthly Income', min_value=0, value=20000)

    data = {
        'Age': Age,
        'TypeofContact': TypeofContact,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': Occupation,
        'Gender': Gender,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': ProductPitched,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus,
        'NumberOfTrips': NumberOfTrips,
        'Passport': Passport,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': OwnCar,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': Designation,
        'MonthlyIncome': MonthlyIncome
    }

    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# ----------------------------
# Make Prediction
# ----------------------------
if st.button('🔮 Predict'):
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"✅ Prediction: Likely to take the product (Confidence: {proba:.2%})")
        else:
            st.warning(f"❌ Prediction: Not likely to take the product (Confidence: {1 - proba:.2%})")

    except Exception as e:
        st.error(f"An error occurred during prediction:\n\n{e}")
