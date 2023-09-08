
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


# Load the model
model = joblib.load("final_model/rta_model.joblib")

# Load the preprocessor
preprocessor = joblib.load("final_model/preprocessor.joblib")

# Define the main function
def main():
    # Set page title and layout
    st.set_page_config(page_title="Accident Severity Prediction App", layout="wide")
    st.title("Accident Severity Prediction App")

    # Define dropdown options
    driver_age_options = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
    vehicle_owner_options = ['Owner', 'Governmental', 'Organization', 'Other']
    vehicle_defect_options = ['No defect', '7', '5']
    accident_area_options = ['Other', 'Office areas', 'Residential areas', 'Church areas',
                             'Industrial areas', 'School areas', 'Recreational areas',
                             'Outside rural areas', 'Hospital areas', 'Market areas', 'Rural village areas',
                             'Unknown', 'Rural village areas', 'Office areas', 'Recreational areas']
    
    lanes_options = ['Two-way', 'Undivided Two way', 'other', 'Double carriageway', 'One way', 'Two-way', 'Unknown']
    surface_type_options = ['Asphalt roads', 'Earth roads', 'Gravel roads', 'Other', 'Asphalt roads with some distress']
    light_condition_options = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit']
    casualty_sex_options = ['Male', 'na', 'Female']
    casualty_work_options = ['Driver', 'Self-employed', 'Employee', 'Other', 'Student', 'Unemployed', 'Unknown']
    pedestrian_movement_options = ["Not a Pedestrian", 
                                    "Crossing from nearside - masked by parked or stationary vehicle", 
                                    "Unknown or other",  
                                    "Crossing from driver's nearside", 
                                    "Crossing from offside - masked by parked or stationary vehicle", 
                                    "In carriageway, stationary - not crossing  (standing or playing)", 
                                    "Walking along in carriageway, back to traffic", 
                                    "In carriageway, stationary - not crossing  (standing or playing) - masked by parked or stationary vehicle",
                                    "Walking along in carriageway, facing traffic"]

    # Define the form
    with st.form("accident_severity_form"):
        # Add form inputs
        st.subheader("Please enter the following inputs:")
        driver_age = st.selectbox("Driver's Age", options=driver_age_options)
        vehicle_owner = st.selectbox("Vehicle Owner", options=vehicle_owner_options)
        vehicle_defect = st.selectbox("Vehicle Defect", options=vehicle_defect_options)
        accident_area = st.selectbox("Accident Areaa", options=accident_area_options)
        lanes = st.selectbox("Lanes", options=lanes_options)
        surface_type = st.selectbox("Surface Type", options=surface_type_options)
        light_condition = st.selectbox("Light Conditions", options=light_condition_options)
        casualty_sex = st.selectbox("Casualty Sex", options=casualty_sex_options)
        casualty_work = st.selectbox("Casualty Work", options=casualty_work_options)
        pedestrian_movement = st.selectbox("Pedestrian Movement", options=pedestrian_movement_options)

        # Add submit button
        submit_button = st.form_submit_button(label='Predict')

    # If submit button is clicked
    if submit_button:
        # Create a DataFrame with the selected input features
        input_data = pd.DataFrame({
            'driver_age': [driver_age],
            'vehicle_owner': [vehicle_owner],
            'vehicle_defect': [vehicle_defect],
            'accident_area': [accident_area],
            'lanes': [lanes],
            'surface_type': [surface_type],
            'light_condition': [light_condition],
            'casualty_sex': [casualty_sex],
            'casualty_work': [casualty_work],
            'pedestrian_movement': [pedestrian_movement]
        })

        # Encode categorical features
        categorical_cols = input_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            input_data[col] = input_data[col].astype('category').cat.codes

        # Make the prediction
        prediction = model.predict(input_data)

        # Show the prediction
        st.subheader("Prediction:")
        st.write("The predicted severity of the accident is:", prediction[0])

# Run the main function
if __name__ == '__main__':
    main()
