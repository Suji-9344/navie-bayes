import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Naive Bayes Student Prediction", layout="wide")
st.title("🎓 Student Result Prediction (Naive Bayes)")

# ---------------- DATASET (BUILT-IN) ----------------
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10]*10,
    "Attendance": ["Low","Low","Low","Medium","Medium","High","High","High","High","High"]*10,
    "Internet_Access": ["Yes","No"]*50,
    "Result": ["Fail","Fail","Fail","Fail","Pass","Pass","Pass","Pass","Pass","Pass"]*10
}

df = pd.DataFrame(data)

# ---------------- ENCODE CATEGORICAL DATA ----------------
le_att = LabelEncoder()
le_net = LabelEncoder()
le_res = LabelEncoder()

df["Attendance"] = le_att.fit_transform(df["Attendance"])
df["Internet_Access"] = le_net.fit_transform(df["Internet_Access"])
df["Result"] = le_res.fit_transform(df["Result"])

X = df[["Hours_Studied", "Attendance", "Internet_Access"]]
y = df["Result"]

# ---------------- TRAIN MODEL ----------------
model = GaussianNB()
model.fit(X, y)

# ---------------- USER INPUT ----------------
st.sidebar.header("🔧 Enter Student Details")

hours = st.sidebar.slider("Hours Studied", 1, 10, 5)
attendance = st.sidebar.selectbox("Attendance", ["Low", "Medium", "High"])
internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])

# Encode input
attendance_enc = le_att.transform([attendance])[0]
internet_enc = le_net.transform([internet])[0]

# ---------------- PREDICTION ----------------
prediction = model.predict([[hours, attendance_enc, internet_enc]])
result = le_res.inverse_transform(prediction)[0]

# ---------------- OUTPUT ----------------
st.subheader("📊 Prediction Result")

if result == "Pass":
    st.success("✅ Student will PASS")
else:
    st.error("❌ Student will FAIL")

# ---------------- SHOW DATA ---------
