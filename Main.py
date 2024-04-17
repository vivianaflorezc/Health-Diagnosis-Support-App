import streamlit as st
from metabolic import metabolic_app
from diabetes import diabetes_app
from heart import heart_app
from fetal import fetal_app
from Home import home_app
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title='Home',
    page_icon='house'
)

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:
            app = option_menu(
                menu_title='Select an option ',
                options=['Home', 'Diabetes', 'Metabolic Syndrome', 'Fetal health', 'Heart disease'],
                icons=['house', 'clipboard2-pulse', 'activity', 'clipboard-heart', 'heart-pulse'],
                menu_icon='check2-circle',
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "black", "font-size": "15px"},
                    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                                 "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "lightblue"},}
            )
            with st.sidebar:
                st.write('Viviana Florez Camacho.\n\n')
                st.image('logoie.png')

        if app == 'Home':
            home_app()
        if app == 'Diabetes':
            diabetes_app()

        elif app == 'Metabolic Syndrome':
            metabolic_app()

        elif app == 'Fetal health':
            fetal_app()

        elif app == 'Heart disease':
            heart_app()

    run()

