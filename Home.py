import streamlit as st

def home_app():
    html_temp = """
    <div style="background-color:lightblue;padding:10px">
    <h2 style="color:white;text-align:center;">Health Diagnosis Support App</h2>
    </div>"""

    st.markdown(html_temp, unsafe_allow_html=True)

    st.image('logo.jpg')
    st.markdown('---')
    texto = ('The Health Diagnosis Support App is a powerful tool designed to provide invaluable assistance to medical professionals in the diagnosis and identification of risk factors for a range of diseases. With a user-friendly interface and specialized modules for diseases such as Diabetes, Metabolic Syndrome, Fetal Health, and Heart Disease, this application offers comprehensive support and data analysis for healthcare professionals.\n\n'
             'To access any of the available modules, simply click on the left panel of the screen. Then select the name of the module you want, enter the corresponding information and finally click on the predict button to obtain the results.')

    # Justificar y alinear el texto
    st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)







