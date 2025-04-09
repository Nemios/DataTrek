
import streamlit as st
import pandas as pd

st.title("DataTrek")
st.write("Analyse d'un Dataet Kaggle : Indian Student Depression")

data = pd.read_csv(r"Data/Raw/student-depression-dataset.csv")
df = data.copy()
st.write(df)
