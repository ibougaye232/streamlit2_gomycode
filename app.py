import pandas as pd
import numpy as np
import joblib
import streamlit as st

model=joblib.load('C:/Users/ass85/PycharmProjects/checkpoint_streamlit_2.py/.venv/Scripts/trained_classifier_model.joblib')
open("C:/fii_db.csv")
fii_db=pd.read_csv("C:/fii_db.csv")
st.title("Expresso_CHURN_predictor")
st.header("Expresso_CHURN_predictor")
st.text("Enter the REGULARITY and the ON_NET, and we will predict the CHURN")

# Utilisez st.number_input au lieu de st.int_input pour les valeurs numériques
aor = st.number_input("age_of_respondent", min_value=0)
hs = st.number_input("household_size", min_value=0)
country = st.number_input("country",min_value=0)
year = st.number_input("year",min_value=0)
el = st.number_input("education_level",min_value=0)
gor = st.number_input("gender_of_respondent",min_value=0)

# Créez un tableau numpy avec les valeurs entrées
data = np.array([aor, hs, country, year, el, gor]).reshape(1, -1)

# Utilisez st.write() au lieu de print() pour afficher les résultats dans Streamlit
st.write("Prediction:", model.predict(data))

#lien google collab: https://colab.research.google.com/drive/19uVIZRmwD2oneViWTtK2rWreP4OAcX9G#scrollTo=OGMFpk5Bhqm7