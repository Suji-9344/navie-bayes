import streamlit as st
import pandas as pd

st.title("📤 Upload Naive Bayes Dataset")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully ✅")
    st.dataframe(df)
else:
    st.info("Please upload naive_bayes_students_100.xlsx")
