import streamlit as st
import joblib
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# -------------------------
# Load model + data
# -------------------------

model = joblib.load("model/disease_model.pkl")
symptoms_list = joblib.load("model/symptoms.pkl")

description_df = pd.read_csv("dataset/symptom_description.csv")
precaution_df = pd.read_csv("dataset/symptom_precaution.csv")

st.set_page_config(page_title="Medical AI Assistant")

st.title("🩺 AI Medical Assistant")

st.write("You can either type symptoms OR upload a medical report.")

# ----------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------

def extract_symptoms(text):

    text = text.lower().replace(" ", "_")

    found = []

    for symptom in symptoms_list:
        if symptom in text:
            found.append(symptom)

    return found


def predict(symptoms):

    input_vector = [0] * len(symptoms_list)

    for symptom in symptoms:
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            input_vector[index] = 1

    prediction = model.predict([input_vector])

    return prediction[0]


def get_description(disease):

    row = description_df[description_df["Disease"] == disease]

    if len(row) > 0:
        return row.iloc[0]["Description"]

    return "No description available."


def get_precautions(disease):

    row = precaution_df[precaution_df["Disease"] == disease]

    if len(row) > 0:

        precautions = row.iloc[0][1:].dropna().values

        return precautions

    return []


# ----------------------------------------------------
# OCR FOR IMAGE
# ----------------------------------------------------

def image_to_text(image):

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)

    return text


# ----------------------------------------------------
# OCR FOR PDF
# ----------------------------------------------------

def pdf_to_text(file):

    pages = convert_from_bytes(file.read())

    text = ""

    for page in pages:
        text += pytesseract.image_to_string(page)

    return text


# ----------------------------------------------------
# SYMPTOM INPUT SECTION
# ----------------------------------------------------

st.header("🔎 Symptom Checker")

user_input = st.text_area(
    "Example: I have fever, headache and vomiting"
)

if st.button("Predict from Symptoms"):

    symptoms_found = extract_symptoms(user_input)

    if len(symptoms_found) == 0:

        st.error("No symptoms detected")

    else:

        disease = predict(symptoms_found)

        st.success(f"Predicted Disease: {disease}")

        description = get_description(disease)

        st.markdown("## Disease Description")

        st.markdown(
            f"""
            <div style="font-size:20px;padding:15px;border-radius:10px;background:#111827;">
            {description}
            </div>
            """,
            unsafe_allow_html=True
        )

        precautions = get_precautions(disease)

        with st.expander("⚕️ View Precautions"):

            for p in precautions:
                st.write("•", p)


# ----------------------------------------------------
# REPORT ANALYZER
# ----------------------------------------------------

st.header("📄 Upload Medical Report")

uploaded_file = st.file_uploader(
    "Upload report (PDF or Image)",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    if uploaded_file.type == "application/pdf":

        extracted_text = pdf_to_text(uploaded_file)

    else:

        image = Image.open(uploaded_file)

        extracted_text = image_to_text(image)

    st.subheader("Extracted Report Text")

    st.text(extracted_text[:1000])

    # Extract symptoms from report
    report_symptoms = extract_symptoms(extracted_text)

    if len(report_symptoms) > 0:

        disease = predict(report_symptoms)

        st.success(f"Diagnosis from Report: {disease}")

        description = get_description(disease)

        st.markdown("## Disease Description")

        st.markdown(
            f"""
            <div style="font-size:20px;padding:15px;border-radius:10px;background:#111827;">
            {description}
            </div>
            """,
            unsafe_allow_html=True
        )

        precautions = get_precautions(disease)

        with st.expander("⚕️ View Precautions"):

            for p in precautions:
                st.write("•", p)

    else:

        st.warning("No recognizable symptoms found in report.")

# ----------------------------------------------------
# DOCTOR CONTACT SECTION 1
# ----------------------------------------------------

st.markdown("## 👨‍⚕️ Contact Our Doctor")

doctor_name = "Dr. Sahil Singh"
specialization = "Neurologist"
hospital = "City Care Hospital"
doctor_number = "+911111111111"

st.markdown(
    f"""
<div style="
display:flex;
align-items:center;
gap:20px;
padding:20px;
border-radius:12px;
background-color:#1f2937;
margin-top:10px;
">

<img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png"
width="80"
style="border-radius:50%;">

<div style="flex:1">

<div style="font-size:20px;font-weight:600;">
{doctor_name}
</div>

<div style="font-size:15px;color:#9ca3af;">
{specialization}
</div>

<div style="font-size:14px;margin-top:4px;">
🏥 {hospital}
</div>

<div style="margin-top:12px;display:flex;gap:10px;">

<a href="tel:{doctor_number}">
<button style="
background-color:#ef4444;
color:white;
padding:8px 16px;
border:none;
border-radius:8px;
cursor:pointer;">
📞 Call
</button>
</a>

<a href="https://wa.me/{doctor_number.replace('+','')}">
<button style="
background-color:#22c55e;
color:white;
padding:8px 16px;
border:none;
border-radius:8px;
cursor:pointer;">
💬 WhatsApp
</button>
</a>

</div>

</div>

</div>
""",
unsafe_allow_html=True
)


# ----------------------------------------------------
# DOCTOR CONTACT SECTION 2
# ----------------------------------------------------

doctor_name = "Dr. Ishita Bhatt"
specialization = "Physician"
hospital = "City Care Hospital"
doctor_number = "+912222222222"

st.markdown(
    f"""
<div style="
display:flex;
align-items:center;
gap:20px;
padding:20px;
border-radius:12px;
background-color:#1f2937;
margin-top:10px;
">

<img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png"
width="80"
style="border-radius:50%;">

<div style="flex:1">

<div style="font-size:20px;font-weight:600;">
{doctor_name}
</div>

<div style="font-size:15px;color:#9ca3af;">
{specialization}
</div>

<div style="font-size:14px;margin-top:4px;">
🏥 {hospital}
</div>

<div style="margin-top:12px;display:flex;gap:10px;">

<a href="tel:{doctor_number}">
<button style="
background-color:#ef4444;
color:white;
padding:8px 16px;
border:none;
border-radius:8px;
cursor:pointer;">
📞 Call
</button>
</a>

<a href="https://wa.me/{doctor_number.replace('+','')}">
<button style="
background-color:#22c55e;
color:white;
padding:8px 16px;
border:none;
border-radius:8px;
cursor:pointer;">
💬 WhatsApp
</button>
</a>

</div>

</div>

</div>
""",
unsafe_allow_html=True
)

st.info("Disclaimer: This AI assistant is for informational purposes only and should not replace professional medical advice. Always consult a healthcare provider for accurate diagnosis and treatment.")