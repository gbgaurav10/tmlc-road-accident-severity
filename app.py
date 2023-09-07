
import os
import streamlit as st
import pandas as pd
from data_transformation import DataTransformation, DataTransformationConfig  # Import your data transformation classes
from sklearn.model_selection import train_test_split
from accident_severity.pipeline.predictions import PredictionPipeline

# Create a Streamlit web app
st.title('Road Accident Severity Prediction App')
st.write('Choose how you want to input data for accident severity prediction.')

# Radio button to choose input method
input_method = st.radio("Select Input Method:", ("Upload CSV File", "Enter Data Manually"))

if input_method == "Upload CSV File":
    # File Upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Configuration for data transformation
        data_transformation_config = DataTransformationConfig(data_path=uploaded_file, root_dir='output')

        # Create an instance of DataTransformation
        data_transformer = DataTransformation(data_transformation_config)

        # Button to trigger data transformation
        if st.button('Transform Data'):
            # Perform data transformation
            try:
                data_transformer.get_data_transformation()
                data_transformer.handle_data_imbalance()
                data_transformer.train_test_split()
                st.success('Data transformation completed successfully.')

                # Display a button to download transformed data
                if st.button('Download Transformed Data'):
                    st.write('Download the transformed data:')
                    st.markdown(data_transformer.config.root_dir + "/train_resampled.csv")

            except Exception as e:
                st.error(f'Data transformation failed: {str(e)}')

        # Show a sample of the uploaded data
        if uploaded_file is not None:
            st.subheader('Sample of the Uploaded Data')
            df = pd.read_csv(uploaded_file)
            selected_features = ['driver_age', 'vehicle_owner', 'vehicle_defect', 'accident_area',
                                 'lanes', 'surface_type', 'light_condition', 'casualty_sex',
                                 'casualty_work', 'pedestrian_movement']
            st.write(df[selected_features].head())

elif input_method == "Enter Data Manually":
    st.subheader('Enter Data Manually:')

    # Create input fields for selected features
    driver_age = st.slider('Driver Age', min_value=0, max_value=100, step=1)
    vehicle_owner = st.selectbox('Vehicle Owner', ['Private', 'Commercial'])
    vehicle_defect = st.selectbox('Vehicle Defect', ['Yes', 'No'])
    accident_area = st.text_input('Accident Area')
    lanes = st.selectbox('Lanes', ['1', '2', '3+'])
    surface_type = st.selectbox('Surface Type', ['Dry', 'Wet', 'Ice', 'Snow', 'Other'])
    light_condition = st.selectbox('Light Condition', ['Daylight', 'Dark (No Street Lights)',
                                                       'Dark (Street Lights)', 'Dusk', 'Dawn'])
    casualty_sex = st.selectbox('Casualty Sex', ['Male', 'Female'])
    casualty_work = st.selectbox('Casualty Work', ['Employed', 'Unemployed', 'Student', 'Other'])
    pedestrian_movement = st.selectbox('Pedestrian Movement', ['Crossing Road', 'Not Crossing Road', 'Other'])

    # Create a button to trigger prediction
    if st.button('Predict Accident Severity'):
        # Create a DataFrame from the manually entered data
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

        # You can use the input_data for prediction
        # Replace this with your actual prediction logic
        prediction = "Slight Injury"  # Placeholder, replace with actual prediction

        st.subheader('Prediction:')
        st.write('Predicted Accident Severity:', prediction)

# Additional features can be added for model training and prediction if needed



