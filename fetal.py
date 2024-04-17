import streamlit as st
import pandas as pd
import numpy as np
import joblib
def fetal_app():
    texto='The Fetal Health Module is an essential component of our application, dedicated to evaluating the well-being of the unborn child. This module allows healthcare professionals to input crucial data related to fetal health, such as uterine contractions, accelerations per second, fetal movement, among others key parameters. This module employs a XGBoost classification model to assess the health status of the fetus and predict potential risks. By providing accurate and timely information, the Fetal Health Assessment Module empowers healthcare providers to make informed decisions and ensure the best possible outcomes for both mothers and their unborn children. If you wish to assess the fetal health and receive valuable insights, please enter the information below.'
    # Justificar y alinear el texto
    st.markdown("<h1 style='text-align: center;'>Fetal Health Prediction</h1>", unsafe_allow_html=True)
    st.write('https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification')
    st.markdown('---')
    st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)
    model = joblib.load('Fetal_model.pkl')
    baseline_value=st.number_input('baseline value - FHR baseline (beats per minute)', min_value=60.0, max_value=200.0, value=80.0, step=0.01)
    accelerations = st.slider('Number of accelerations per second', min_value=0.000, max_value=0.05, value=0.006, step=0.001)
    fetal_movement = st.slider('Number of fetal movements per second', min_value=0.0, max_value=0.100, value=0.020, step=0.01)
    uterine_contractions = st.number_input('Number of uterine contractions per second',min_value=0.00, max_value=0.03, value=0.01,step=0.01)
    light_decelerations = st.number_input('Number of light decelerations per second',min_value=0.00, max_value=0.03, value=0.01,step=0.01)
    severe_decelerations = st.number_input('Number of severe decelerations per second',min_value=0.00, max_value=0.03, value=0.01,step=0.01)
    prolongued_decelerations = st.number_input('Number of prolonged decelerations per second',min_value=0.00, max_value=0.03, value=0.003,step=0.001)
    abnormal_short_term_variability = st.slider('Percentage of time with abnormal short term variability',min_value=0.00, max_value=100.0, value=80.0,step=0.01)
    mean_value_of_short_term_variability = st.slider('Mean value of short term variability',min_value=0.00, max_value=20.0, value=2.0,step=0.01)
    percentage_of_time_with_abnormal_long_term_variability = st.number_input('Percentage of time with abnormal long term variability',min_value=0.00, max_value=100.0, value=20.0,step=0.01)
    mean_value_of_long_term_variability = st.slider('Mean value of long term variability',min_value=0.00, max_value=500.0, value=15.0,step=0.01)
    histogram_max = st.number_input('Maximum (high frequency) of FHR histogram',min_value=0.00, max_value=300.0, value=100.0,step=0.01)
    histogram_number_of_peaks = st.number_input('Number of histogram peaks',min_value=0.00, max_value=50.0, value=10.0,step=0.01)
    histogram_number_of_zeroes = st.number_input('Number of histogram zeros',min_value=0.00, max_value=20.0, value=10.0,step=0.01)
    histogram_mode = st.number_input('Histogram mode',min_value=50.00, max_value=300.0, value=100.0,step=0.01)
    histogram_mean = st.number_input('Histogram mean',min_value=50.00, max_value=300.0, value=100.0,step=0.01)
    histogram_median = st.number_input('Histogram median',min_value=50.00, max_value=300.0, value=100.0,step=0.01)
    histogram_variance = st.number_input('Histogram variance',min_value=0.00, max_value=300.0, value=15.0,step=0.01)
    histogram_tendency = st.slider('Histogram tendency',min_value=-1.0, max_value=1.0, value=0.5,step=0.01)

    # Submit button to predict
    if st.button('Predict'):
        data = {
            'baseline value': baseline_value,
            'accelerations': accelerations,
            'fetal_movement': fetal_movement,
            'uterine_contractions': uterine_contractions,
            'light_decelerations': light_decelerations,
            'severe_decelerations': severe_decelerations,
            'prolongued_decelerations':prolongued_decelerations,
            'abnormal_short_term_variability':abnormal_short_term_variability,
            'mean_value_of_short_term_variability':mean_value_of_short_term_variability,
            'percentage_of_time_with_abnormal_long_term_variability':percentage_of_time_with_abnormal_long_term_variability,
            'mean_value_of_long_term_variability':mean_value_of_long_term_variability,
            'histogram_max':histogram_max,
            'histogram_number_of_peaks':histogram_number_of_peaks,
            'histogram_number_of_zeroes':histogram_number_of_zeroes,
            'histogram_mode':histogram_mode,
            'histogram_mean':histogram_mean,
            'histogram_median':histogram_median,
            'histogram_variance':histogram_variance,
            'histogram_tendency':histogram_tendency
        }

        df = pd.DataFrame.from_dict([data])
        # changing the input_data to numpy array
        #input_data_as_numpy_array = np.asarray(data)

        # reshape the array as we are predicting for one instance
        #input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        probabilities = model.predict_proba(df)

        predicted_class = np.argmax(probabilities)

        if predicted_class == 0:
            st.success('Fetal Health: Normal')
        elif predicted_class == 1:
            st.warning('Fetal Health: Suspect')
        else:
            st.error('Fetal Health: Pathological')

        #st.write(
         #   f'Class Probabilities: Normal: {probabilities[0, 0]}, Suspect: {probabilities[0, 1]}, Pathological: {probabilities[0, 2]}')


