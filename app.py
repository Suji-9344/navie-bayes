import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Naive Bayes App", layout="wide")
st.title("📊 Naive Bayes Student Result Prediction")

# ---------------- LOAD DATA ----------------
st.subheader("📂 Dataset")

# File must be in same folder as app.py
df = pd.read_excel("naive_bayes_students_100.xlsx")
st.dataframe(df)

# ---------------- ENCODING ----------------
le_attendance = LabelEncoder()
le_internet = LabelEncoder()
le_result = LabelEncoder()

df["Attendance"] = le_attendance.fit_transform(df["Attendance"])
df["Internet_Access"] = le_internet.fit_transform(df["Internet_Access"])
df["Result"] = le_result.fit_transform(df["Result"])

# ---------------- FEATURES & TARGET ----------------
X = df[["Hours_Studied", "Attendance", "Internet_Access"]]
y = df["Result"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL TRAINING ----------------
model = GaussianNB()
model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)

# ---------------- ACCURACY ----------------
accuracy = accuracy_score(y_test, y_pred)
st.subheader("✅ Model Accuracy")
st.write(f"Accuracy: *{accuracy:.2f}*")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le_result.classes_)
disp.plot(ax=ax)
st.pyplot(fig)

# ---------------- USER INPUT PREDICTION ----------------
st.subheader("🔮 Predict Student Result")

hours = st.number_input("Hours Studied", min_value=1, max_value=10, value=6)
attendance = st.selectbox("Attendance", ["Low", "Medium", "High"])
internet = st.selectbox("Internet Access", ["Yes", "No"])

if st.button("Predict"):
    input_data = [[
        hours,
        le_attendance.transform([attendance])[0],
        le_internet.transform([internet])[0]
    ]]

    prediction = model.predict(input_data)
    result = le_result.inverse_transform(prediction)[0]

    st.success(f"🎯 Predicted Result: *{result}*")
