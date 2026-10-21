# =====================================================================
# 81. STUDENT PLACEMENT PREDICTION (LOGISTIC REGRESSION)
# =====================================================================

import streamlit as st                                # Minimalist web UI
import pandas as pd                                   # Loads CSV dataset table
from sklearn.model_selection import train_test_split  # Splits data into Train (80%) and Test (20%)
from sklearn.linear_model import LogisticRegression   # Binary Classification ML algorithm
from sklearn.metrics import accuracy_score            # Calculates model accuracy percentage

# =====================================================================
# PART 1: BACKEND MACHINE LEARNING (TEACHER QUESTIONS COME FROM HERE)
# =====================================================================
@st.cache_resource  # Keeps model trained in memory so UI clicks are instant
def train_backend_model():
    # 1. Load dataset table
    df = pd.read_csv("placement.csv")
    
    # 2. X = Input Features (CGPA, skills rating, internships count)
    X = df[["cgpa", "skills", "internships"]]
    
    # 3. y = Target Output (1 = Placed, 0 = Not Placed)
    y = df["placed"]
    
    # 4. 80/20 train-test split for fair evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Initialize the Logistic Regression model
    model = LogisticRegression()
    
    # 6. Fit (train) model on the training set
    model.fit(X_train, y_train)
    
    # 7. Evaluate on unseen test data
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, acc

# Train the model once
model, accuracy = train_backend_model()

# =====================================================================
# PART 2: MINIMALIST FRONTEND DASHBOARD
# =====================================================================
st.set_page_config(page_title="Placement Predictor", layout="centered")

# Hide extra web menus for a clean, minimalist UI
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Student Placement Prediction")
st.caption(f"Backend: Logistic Regression  |  Model Test Accuracy: {accuracy * 100:.1f}%")
st.divider()

# 3 clean input columns side-by-side
col1, col2, col3 = st.columns(3)

with col1:
    cgpa = st.number_input("CGPA (5.0 - 10.0)", min_value=5.0, max_value=10.0, value=7.5, step=0.1)

with col2:
    skills = st.number_input(
        "Skills Rating (1 - 10)",
        min_value=1,
        max_value=10,
        value=6,
        step=1,
        help="Based on college mock coding tests, LeetCode proficiency, and project complexity (1=Beginner, 5=Average, 8+=Placement Ready)"
    )

with col3:
    internships = st.number_input("Internships (0 - 5)", min_value=0, max_value=5, value=1, step=1)

# Prediction button
if st.button("Predict Placement & Career Guidance", use_container_width=True, type="primary"):
    student_input = pd.DataFrame([[cgpa, skills, internships]], columns=["cgpa", "skills", "internships"])
    
    prediction = model.predict(student_input)[0]
    probability = model.predict_proba(student_input)[0][1] * 100
    
    st.divider()
    st.metric(label="Placement Probability", value=f"{probability:.1f}%")
    
    if prediction == 1:
        st.success("Outcome: LIKELY TO BE PLACED")
        st.write("**Career Guidance:** Strong profile. Focus on mock technical interviews and DSA revision.")
    else:
        st.error("Outcome: AT RISK / NEEDS IMPROVEMENT")
        st.write("**Career Guidance:** Aim for CGPA > 7.0, complete 1–2 internships, and build practical projects.")