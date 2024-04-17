import pickle
import streamlit as st
import pandas as pd
import numpy as np
import joblib

def diabetes_app():
    st.markdown("<h1 style='text-align: center;'>Diabetes prediction</h1>", unsafe_allow_html=True)
    st.write('https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes')
    st.markdown('---')
    texto = 'This  is a specialized module designed to assist healthcare professionals in diabetes diagnosing. It enables medical practitioners to input patient-specific data, including medical history, blood glucose levels, and lifestyle factors, to assess diabetes risk. This module employs a KNN classification model to calculate the likelihood of diabetes. If you want to determine if someone could have diabetes, please enter his information below.'
    st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)
    model_diab = joblib.load('diabetes_model.pkl')
    Pregnancies = st.slider('Number of Pregnancies', min_value=0, max_value=20, value=4, step=1)
    Glucose = st.slider('Glucose level', min_value=0, max_value=199, value=60, step=1)
    BloodPressure = st.slider('BloodPressure level in mm Hg', min_value=0, max_value=140, value=80, step=1)
    SkinThickness = st.slider('SkinThickness in mm', min_value=0, max_value=100, value=40, step=1)
    Insulin = st.slider('Insulin level in mu U/ml', min_value=0, max_value=1000, value=400, step=1)
    BMI = st.slider('BMI value', min_value=0.0, max_value=70.0, value=33.3, step=0.01)
    DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function value', min_value=0.000, step=0.001, max_value=3.0,value=0.045)
    Age = st.slider('Age of a person', min_value=10, max_value=100, value=21, step=1)

    if st.button('Predict Diabetes'):
        print("Button Diabetes")
        data_diab = {
            'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age
        }
        df_diab = pd.DataFrame.from_dict([data_diab])
        scaler_diab = joblib.load('scaler_diab.pkl')
        data_diab = scaler_diab.transform(df_diab)
        # changing the input_data to numpy array
        input_datadiab_as_numpy_array = np.asarray(data_diab)

        # reshape the array as we are predicting for one instance
        input_datadiab_reshaped = input_datadiab_as_numpy_array.reshape(1, -1)

        prediction_diab = model_diab.predict(input_datadiab_reshaped)
        print(prediction_diab)

        if prediction_diab[0] == 0:
            st.success('Not Diabetic')
        else:
            st.error('Diabetic')
        #probabilities_diab = model_diab.predict_proba(input_datadiab_reshaped)
        #st.write(probabilities_diab)
        #st.write("0 = Not Diabetic, 1 = Diabetic")

