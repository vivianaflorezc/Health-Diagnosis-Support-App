import pickle
import streamlit as st
import pandas as pd
import numpy as np
import joblib

def metabolic_app():
    st.markdown("<h1 style='text-align: center;'>Metabolic Syndrome prediction</h1>", unsafe_allow_html=True)
    st.write('https://www.kaggle.com/datasets/antimoni/metabolic-syndrome/data')
    st.markdown('---')
    texto ='The Metabolic Syndrome Module is a crucial component of the app, dedicated to identifying and evaluating risk factors associated with metabolic syndrome. Healthcare professionals can enter patient data, including waist circumference, blood pressure, lipid profiles, glucose levels, among others. This module employs a SVM classification model to calculate the likelihood of metabolic-syndrome. If you want to determine if someone could have metabolic-syndrome, please enter his information below.'
    st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)
    model = joblib.load('MetabolicSyndrome_model.pkl')

    Age = st.slider('Age of a person', min_value=10, max_value=100, value=21, step=1)
    Sex = st.selectbox('Sex', ('Female', 'Male'))
    Marital = st.selectbox('Marital status', ('Single', 'Married', 'Widowed', 'Divorced', 'Separated'))
    Income = st.slider('Income', min_value=0, max_value=1000000, value=4000, step=1)
    Race = st.selectbox('Race', ('White', 'Asian', 'Black', 'MexAmerican', 'Hispanic', 'Other'))
    WaistCirc = st.slider('Waist Circunference value', min_value=0.0, max_value=300.0, value=33.3, step=0.01)
    BMI = st.slider('BMI value', min_value=0.0, max_value=70.0, value=33.3, step=0.01)
    Albuminuria = st.slider('Albuminuria value', min_value=0.0, max_value=5.0, value=1.0, step=0.01)
    UrAlbCr = st.number_input('Urinary albumin-to-creatinine ratio', min_value=0.0, max_value=10000.0, value=10.0,step=0.1)
    UricAcid = st.slider('Uric acid levels in the blood.', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    BloodGlucose = st.slider('Blood glucose level.', min_value=0.0, max_value=500.0, value=20.0, step=0.1)
    HDL = st.slider('High-Density Lipoprotein cholesterol level', min_value=0.0, max_value=1000.0, value=20.0, step=0.1)
    Triglycerides = st.number_input('Triglyceride levels in the blood.', min_value=0.0, max_value=10000.0, value=10.0,step=0.1)

        # Submit button to predict
    if st.button('Predict'):
        data = {
            'Age': Age,
            'Income': Income,
            'WaistCirc': WaistCirc,
            'BMI': BMI,
            'Albuminuria': Albuminuria,
            'UrAlbCr': UrAlbCr,
            'UricAcid': UricAcid,
            'BloodGlucose': BloodGlucose,
            'HDL': HDL,
            'Triglycerides': Triglycerides
            }

        data_cat = {
            'Sex': Sex,
            'Marital': Marital,
            'Race': Race
            }
        df = pd.DataFrame.from_dict([data])
        df_cat = pd.DataFrame.from_dict([data_cat])
        encoder = joblib.load('ohe.pkl')
        encoded_data = encoder.transform(df_cat)
        df_cat = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        # Unir las variables numéricas y categóricas codificadas
        data_full = pd.concat([df.reset_index(drop=True), df_cat], axis=1)
        with open('scaler.pkl', 'rb') as s:
            scaler1 = pickle.load(s)
        data = scaler1.transform(data_full)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = model.predict(input_data_reshaped)
        print(prediction)

        if prediction[0] == 0:
            st.success('Not Metabolic Syndrome')
        else:
            st.error('Metabolic Syndrome')
        #probabilities = model.predict_proba(input_data_reshaped)
            #st.write(probabilities)
            #st.write("0 = Not Metabolic Syndrome, 1 = Metabolic Syndrome")