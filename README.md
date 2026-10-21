# 🎓 Student Placement Prediction & Career Guidance System

An end-to-end Machine Learning web application that predicts campus placement outcomes based on academic performance and technical credentials. Powered by a **Logistic Regression** classification model and wrapped in a minimalist **Streamlit** dashboard.

---

## 📌 Project Overview
* **Problem Type:** Supervised Binary Classification ($y \in \{0, 1\}$)
* **Dataset:** 300 student records modeled after Kaggle placement benchmarks
* **Core Algorithm:** Logistic Regression with Sigmoid Activation
* **Frontend:** Streamlit
* **Deployment:** Streamlit Community Cloud

---

## ⚙️ Mathematical Architecture

Logistic Regression models the probability of placement via the Sigmoid function:

$$z = w_1 \cdot \text{CGPA} + w_2 \cdot \text{Skills} + w_3 \cdot \text{Internships} + b$$

$$P(Y = 1 | X) = \frac{1}{1 + e^{-z}}$$

Where:
* **CGPA:** Academic foundation (5.0 to 10.0 scale)
* **Skills:** Normalized composite technical assessment score (1 to 10 scale)
* **Internships:** Practical industrial exposure (0 to 5)

---

## 🚀 How to Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
