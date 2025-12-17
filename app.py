import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Pass/Fail Prediction", layout="wide")
st.title("📊 Student Pass/Fail Prediction App")

# ---------- USER INPUT ----------
st.sidebar.header("🔧 Enter Student Data")
study_hours = st.sidebar.number_input("Study Hours", min_value=0.0, value=5.0)
attendance = st.sidebar.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)

# ---------- SAMPLE TRAINING DATA ----------
# For demo, we create a small dataset
data = {
    'StudyHours': [2, 4, 6, 8, 10, 1, 3, 7, 9, 5],
    'Attendance': [50, 60, 70, 80, 90, 40, 55, 85, 95, 65],
    'PassFail': [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

# ---------- MODEL TRAINING ----------
X = df[['StudyHours', 'Attendance']]
y = df['PassFail']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# ---------- USER INPUT PREDICTION ----------
user_input = pd.DataFrame({'StudyHours':[study_hours], 'Attendance':[attendance]})
user_input_scaled = scaler.transform(user_input)

prediction = model.predict(user_input_scaled)[0]
prediction_proba = model.predict_proba(user_input_scaled)[0][prediction]

# ---------- DISPLAY RESULTS ----------
st.write("### Student Data")
st.table(user_input)

st.write("### Prediction Result")
result = "✅ Pass" if prediction == 1 else "❌ Fail"
st.write(f"Prediction: **{result}** with probability {prediction_proba:.2f}")

