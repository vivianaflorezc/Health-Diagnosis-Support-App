import streamlit as st
import pandas as pd
import numpy as np
import joblib

def heart_app():
    st.markdown("<h1 style='text-align: center;'>Heart Disease prediction</h1>", unsafe_allow_html=True)
    st.write('https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data')
    st.markdown('---')
    texto='The Heart Disease Module is an integral part of our app, designed to assess and predict the risk of heart disease by analyzing various critical factors. Healthcare professionals can input essential patient data, including age, gender, chest pain type, blood pressure, among others. This module employs a Random forest classification model to calculate the likelihood of having heart disease. If you want to determine if someone could have a heart disease, please enter his information below.'
    st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)
    model = joblib.load('Heart_model.pkl')
    Age = st.slider('Age of a person', min_value=10, max_value=100, value=21, step=1)
    Sex = st.selectbox('Sex', ('Female', 'Male'))
    ChestPainType = st.selectbox('Chest pain type',
                             ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'))
    RestingBP = st.slider('Resting blood pressure [mm Hg]', min_value=0.0, max_value=500.0, value=21.0, step=0.01)
    Cholesterol = st.slider('Serum cholesterol [mm/dl]', min_value=0.0, max_value=1000.0, value=21.0, step=0.01)
    FastingBS = st.selectbox('Fasting blood sugar', ('Higher than 120 mg/dl', 'Otherwise'))
    RestingECG = st.selectbox('Resting electrocardiogram results', (
        'Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy'))
    MaxHR = st.slider('Maximum heart rate achieved', min_value=60.0, max_value=202.0, value=21.0, step=0.01)
    ExerciseAngina = st.selectbox('Exercise-induced angina', ('Yes', 'No'))
    Oldpeak = st.number_input('Oldpeak', min_value=-10.0, max_value=10.0, value=5.0, step=0.01)
    ST_Slope = st.selectbox('The slope of the peak exercise ST segment', ('Upsloping', 'Flat', 'Downsloping'))

    # Submit button to predict
    if st.button('Predict'):
        data = {
            'Age': Age,
            'RestingBP': RestingBP,
            'Cholesterol': Cholesterol,
            'MaxHR': MaxHR,
            'Oldpeak': Oldpeak
        }

        data_cat = {
            'Sex': 'M' if Sex == 'Male' else 'F',
            'ChestPainType': 'TA' if ChestPainType == 'Typical Angina' else 'ATA' if 'ChestPainType'== 'Atypical Angina' else 'NAP' if ChestPainType == 'Non-Anginal Pain' else 'Asymptomatic',
            'FastingBS': 1 if FastingBS == 'Higher than 120 mg/dl' else 0,
            'RestingECG': 'Normal' if RestingECG == 'Normal' else 'ST' if RestingECG == 'Having ST-T wave abnormality' else 'LVH',
            'ExerciseAngina': 'Y' if ExerciseAngina == 'Yes' else 'N',
            'ST_Slope': 'Up' if  ST_Slope =='Upsloping' else 'Flat' if ST_Slope == 'Flat' else 'Down'
        }
        df = pd.DataFrame.from_dict([data])
        df_cat = pd.DataFrame.from_dict([data_cat])
        encoder = joblib.load('ohe_heart.pkl')
        encoded_data = encoder.transform(df_cat)
        df_cat = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        # Unir las variables numéricas y categóricas codificadas
        data_full = pd.concat([df.reset_index(drop=True), df_cat], axis=1)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(data_full)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = model.predict(input_data_reshaped)
        print(prediction)

        if prediction[0] == 0:
            st.success('Not Heart Disease')
        else:
            st.error('Heart Disease')
        #probabilities = model.predict_proba(input_data_reshaped)
        #st.write(probabilities)
        #st.write("0 = Not Heart Disease, 1 = Heart Disease")
