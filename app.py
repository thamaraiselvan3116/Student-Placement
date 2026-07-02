import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to bottom,#eef7ff,#dceeff);
}

.title {
    text-align:center;
    font-size:40px;
    color:#0B5ED7;
    font-weight:bold;
}

.sub {
    text-align:center;
    font-size:18px;
    color:#555;
}

.stButton>button {
    width:100%;
    background:linear-gradient(90deg,#0d6efd,#00c6ff);
    color:white;
    font-size:20px;
    border-radius:12px;
    height:55px;
    border:none;
    font-weight:bold;
}

.stButton>button:hover {
    background:linear-gradient(90deg,#0056b3,#0099ff);
}

.footer {
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:18px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("students.csv")

X = data[["CGPA", "IQ"]]
y = data["Placed"]

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- TRAIN MODEL ----------------
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_scaled, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test_scaled)
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Model Details")

st.sidebar.success("Algorithm : KNN")

st.sidebar.info("Features Used")

st.sidebar.write("✔ CGPA")

st.sidebar.write("✔ IQ")

st.sidebar.write("Neighbours : 5")

st.sidebar.metric("Accuracy", f"{accuracy*100:.2f}%")

# ---------------- TITLE ----------------
st.markdown("<div class='title'>🎓 Student Placement Prediction</div>", unsafe_allow_html=True)

st.markdown("<div class='sub'>Machine Learning using K-Nearest Neighbors (KNN)</div>", unsafe_allow_html=True)

st.write("")

# ---------------- CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

cgpa = st.slider(
    "📚 Select CGPA",
    0.0,
    10.0,
    7.0,
    0.1
)

iq = st.slider(
    "🧠 Select IQ",
    50,
    150,
    100
)

st.write("")

if st.button("🔍 Predict Placement"):

    input_data = pd.DataFrame(
        [[cgpa, iq]],
        columns=["CGPA","IQ"]
    )

    input_scaled = scaler.transform(input_data)

    result = model.predict(input_scaled)

    st.write("---")

    if result[0] == 1:

        st.balloons()

        st.success("🎉 Congratulations!")

        st.markdown(
            """
            ## ✅ Student is Likely to be **PLACED**
            """
        )

    else:

        st.error("❌ Student is Not Likely to be Placed")

        st.warning("Improve CGPA and Skills for Better Placement Chances.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
"""
<hr>
<div class="footer">
💻 Developed by <b>Navnith</b><br>
Student Placement Prediction System using KNN
</div>
""",
unsafe_allow_html=True
)