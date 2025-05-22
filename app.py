import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from home_page import home_page
from about_cdss import about_cdss 

from streamlit_option_menu import option_menu

# Load trained RNN model
rnn_model = load_model('model/CDSS_RNN.h5')
# Load features and class labels
features = pd.read_csv('data/features.csv')
classes = pd.read_csv('data/classes.csv')

num_features = len(features) - 1

# Prepare encoder for class labels
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(classes.values.reshape(-1, 1))

# Set page configuration
st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",  
)

st.markdown("""
    <style>
        /* Remove top and bottom padding */
        .block-container {
            padding-top: 4rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# Store number of features in session state for use in about_cdss page
if 'num_features' not in st.session_state:
    st.session_state.num_features = num_features

# Place your disease_info dictionary here
# ===========================================
disease_info = {
    "AIDS": {
        "Symptoms": ["Muscle Wasting", "Patches in Throat", "High Fever", "Extra Marital Contacts"],
        "Consulting Specialist": "Infectious Disease Specialist",
        "Diagnostic Tests": ["HIV Antibody/Antigen Test", "CD4 Count", "Viral Load Test"],
        "Treatments": [
            "Antiretroviral Therapy (ART)",
            "Tenofovir",
            "Emtricitabine",
            "Dolutegravir"
        ],
        "Precautions": [
            "Practice Safe Sex - Use condoms, limit partners.",
            "Avoid Sharing Needles - Use sterilized equipment.",
            "Get Regular HIV Testing - For you and your partner(s).",
            "Prevent Mother-to-Child Transmission - ART during pregnancy.",
            "Handle Blood Safely - Use gloves, dispose of sharps properly.",
            "Boost Immunity - Healthy diet and lifestyle."
        ],
        "Resource Link": "https://www.who.int/health-topics/hiv-aids#tab=tab_1",
        "Specialist": [
            "Dr. Mahalakshmi Saravanan",
            "ARC Fertility Hospitals",
            "KK Nagar, Madurai",
            "Phone: 04471967014"
        ]
    },
    "Acne": {
        "Symptoms": [
            "Skin Rash",
            "Pus Filled Pimples",
            "Blackheads",
            "Scarring"
        ],
        "Consulting Specialist": "Dermatologist",
        "Diagnostic Tests": [
            "Patch Testing (for allergies)",
            "Skin Swab or Culture",
            "Lipid Profile"
        ],
        "Treatments": [
            "Hormonal Therapy",
            "Chemical Peels",
            "Laser Therapy",
            "Corticosteroid Injections",
            "Skincare Products"
        ],
        "Precautions": [
            "Avoid picking or squeezing acne.",
            "Use non-comedogenic skincare products.",
            "Wash your face twice daily with a gentle cleanser.",
            "Avoid excessive scrubbing or exfoliation.",
            "Limit sun exposure and use sunscreen.",
            "Avoid oily or greasy cosmetics and hair products."
        ],
        "Resource Link": "https://www.emro.who.int/emhj-volume-12-2006/volume-12-issue-6/factors-believed-by-jordanian-acne-patients-to-affect-their-acne-condition.html",
        "Specialist": [
            "Dr. Senthil Kumar. K",
            "Gem Skin, Hair and Laser Centre",
            "203, Vaigai Colony, Landmark: Near Ambiga Theatre, Madurai",
            "Phone: 01142246375"
        ]
    },
    "Alcoholic Hepatitis": {
        "Symptoms": [
            "Vomiting",
            "Yellowish Skin",
            "Abdominal Pain",
            "Swelling of Stomach",
            "Distention of Abdomen",
            "History of Alcohol Consumption",
            "Fluid Overload"
        ],
        "Consulting Specialist": ["Gastroenterologist", "Hepatologist"],
        "Diagnostic Tests": [
            "Liver Function Tests (LFTs)",
            "Complete Blood Count (CBC)",
            "Coagulation Profile",
            "Ultrasound",
            "CT Scan"
        ],
        "Treatments": [
            "Alcohol Abstinence",
            "Nutritional Support",
            "Corticosteroids",
            "Pentoxifylline",
            "Liver Transplantation"
        ],
        "Precautions": [
            "Avoid alcohol.",
            "Follow a balanced and nutrient-rich diet.",
            "Regularly monitor liver function through follow-ups.",
            "Stay up-to-date on hepatitis A and B vaccinations.",
            "Use medications only under medical supervision."
        ],
        "Resource Link": "https://www.mayoclinic.org/diseases-conditions/achalasia/symptoms-causes/syc-20352850",
        "Specialist": [
            "Dr. A.C. Arun",
            "Lily Mission Hospital",
            "Madurai, Tamil Nadu",
            "Phone: 07947133110"
        ]
    },
    "Allergy": {
        "Symptoms": [
            "Continuous Sneezing",
            "Shivering",
            "Chills",
            "Watering from Eyes"
        ],
        "Consulting Specialist": ["Allergist", "Immunologist", "Pulmonologist"],
        "Diagnostic Tests": [
            "Skin Prick Test",
            "Patch Test",
            "Blood Test (IgE Levels)",
            "Spirometry (for asthma-related allergies)",
            "Allergy-Specific IgE Testing"
        ],
        "Treatments": [
            "Antihistamines",
            "Decongestants",
            "Corticosteroids",
            "Immunotherapy (Allergy Shots)",
            "Epinephrine"
        ],
        "Precautions": [
            "Identify and avoid allergens.",
            "Keep prescribed medications readily available.",
            "Use air purifiers and maintain clean surroundings.",
            "Wear protective clothing or masks when exposed to allergens.",
            "Follow the allergist’s treatment plan consistently."
        ],
        "Resource Link": "https://www.worldallergy.org/",
        "Specialist": [
            "Dr. Meena Priyadarshini P",
            "ENT",
            "23+ Years Experience",
            "MBBS, DNB (ENT), DLO",
            "Apollo Speciality Hospitals, Madurai",
            "Phone: 1860-500-1066"
        ]
    },
    "Arthritis": {
        "Symptoms": [
            "Muscle Weakness",
            "Stiff Neck",
            "Swelling Joints",
            "Movement Stiffness",
            "Painful Walking"
        ],
        "Consulting Specialist": ["Rheumatologist", "Orthopedic Specialist"],
        "Diagnostic Tests": [
            "Blood Test",
            "Erythrocyte Sedimentation Rate",
            "C-Reactive Protein Test",
            "X-Ray",
            "MRI"
        ],
        "Treatments": [
            "Nonsteroidal Anti-Inflammatory Drugs",
            "Corticosteroids",
            "Disease-Modifying Antirheumatic Drugs",
            "Biologic Response Modifiers",
            "Physical Therapy"
        ],
        "Precautions": [
            "Maintain a healthy weight.",
            "Engage in regular, low-impact exercises.",
            "Use assistive devices if needed to reduce joint strain.",
            "Follow a balanced diet rich in anti-inflammatory foods.",
            "Adhere to prescribed medications and treatment plans."
        ],
        "Resource Link": "https://www.arthritis.org/",
        "Specialist": [
            "Dr. N. Ramesh",
            "Apollo Speciality Hospitals",
            "Madurai, Tamil Nadu",
            "Phone: 1860-500-1066"
        ]
    },
    "Bronchial Asthma": {
        "Symptoms": [
            "Fatigue",
            "Cough",
            "High Fever",
            "Breathlessness",
            "Family History",
            "Mucoid Sputum"
        ],
        "Consulting Specialist": ["Pulmonologist", "Allergist"],
        "Diagnostic Tests": [
            "Spirometry",
            "Peak Flow Measurement",
            "Chest X-ray",
            "Arterial Blood Gas Test",
            "Exhaled Nitric Oxide Test"
        ],
        "Treatments": [
            "Inhaled Bronchodilators",
            "Inhaled Corticosteroids",
            "Leukotriene Modifiers",
            "Long-Acting Beta Agonists",
            "Oral Corticosteroids"
        ],
        "Precautions": [
            "Identify and avoid asthma triggers.",
            "Use a peak flow meter to monitor lung function.",
            "Follow prescribed inhaler and medication schedule.",
            "Keep the environment clean.",
            "Get vaccinated for respiratory infections."
        ],
        "Resource Link": "https://www.nhlbi.nih.gov/health/asthma",
        "Specialist": [
            "Dr. R. Rajendran",
            "Meenakshi Mission Hospital & Research Centre",
            "Madurai, Tamil Nadu",
            "Phone: +91 452 435 8888"
        ]
    },
    "Cervical Spondylosis": {
        "Symptoms": [
            "Back Pain",
            "Weakness in Limbs",
            "Neck Pain",
            "Dizziness",
            "Loss of Balance"
        ],
        "Consulting Specialist": ["Orthopedic Specialist", "Neurologist"],
        "Diagnostic Tests": [
            "X-ray",
            "MRI",
            "CT Scan",
            "Nerve Conduction Study",
            "Electromyography"
        ],
        "Treatments": [
            "Pain Relievers",
            "Muscle Relaxants",
            "Physical Therapy",
            "Corticosteroid Injections",
            "Surgery"
        ],
        "Precautions": [
            "Maintain proper posture while sitting and standing.",
            "Use ergonomic furniture to reduce neck strain.",
            "Perform neck and shoulder exercises regularly.",
            "Avoid heavy lifting or sudden neck movements.",
            "Follow prescribed medications and physical therapy."
        ],
        "Resource Link": "https://www.mayoclinic.org/diseases-conditions/cervical-spondylosis/symptoms-causes/syc-20370787",
        "Specialist": [
            "Dr. Divaker James Fenn",
            "MBBS, Orthopedic Surgeon",
            "Fenn Hospital",
            "Phone: 04440115382"
        ]
    },
    "Chickenpox": {
        "Symptoms": [
            "Itching",
            "Skin Rash",
            "Fatigue",
            "Lethargy",
            "High Fever",
            "Headache",
            "Loss of Appetite",
            "Mild Fever",
            "Swelled Lymph Nodes",
            "Malaise",
            "Red Spots Over Body"
        ],
        "Consulting Specialist": ["General Physician", "Dermatologist"],
        "Diagnostic Tests": [
            "Blood Test",
            "Polymerase Chain Reaction (PCR) Test",
            "Direct Fluorescent Antibody (DFA) Test"
        ],
        "Treatments": [
            "Antiviral Medications",
            "Antihistamines",
            "Calamine Lotion",
            "Fever Reducers"
        ],
        "Precautions": [
            "Avoid scratching blisters to prevent scarring and infections.",
            "Isolate to avoid spreading the infection.",
            "Maintain proper hygiene and keep skin clean.",
            "Use cool compresses to relieve itching.",
            "Follow the doctor’s instructions for medications."
        ],
        "Resource Link": "https://www.cdc.gov/chickenpox/index.html",
        "Specialist": [
            "Dr. R. Mohan",
            "Meenakshi Mission Hospital & Research Centre",
            "Madurai, Tamil Nadu",
            "Phone: +91 452 435 8888"
        ]
    },
    "Chronic Cholestasis": {
        "Symptoms": [
            "Itching",
            "Vomiting",
            "Yellowish Skin",
            "Nausea",
            "Loss of Appetite",
            "Abdominal Pain",
            "Yellowing of Eyes"
        ],
        "Consulting Specialist": ["Gastroenterologist", "Hepatologist"],
        "Diagnostic Tests": [
            "Liver Function Tests",
            "Alkaline Phosphatase Test",
            "Bilirubin Test",
            "Ultrasound",
            "MRI"
        ],
        "Treatments": [
            "Ursodeoxycholic Acid",
            "Bile Acid Sequestrants",
            "Vitamin Supplements",
            "Cholestyramine",
            "Liver Transplantation"
        ],
        "Precautions": [
            "Avoid alcohol to reduce liver stress.",
            "Follow a low-fat diet to ease digestion.",
            "Take prescribed vitamin supplements to address deficiencies.",
            "Stay hydrated and maintain overall nutrition.",
            "Regularly monitor liver function with follow-ups."
        ],
        "Resource Link": "https://liverfoundation.org/",
        "Specialist": [
            "Dr. Madhusudhanan J",
            "Surgical Gastroenterologist",
            "Apollo Speciality Hospitals KK Nagar",
            "80 Feet Road, KK Nagar, Madurai",
            "Phone: 8069305511"
        ]
    },
    "Common Cold": {
        "Symptoms": [
            "Continuous Sneezing",
            "Chills",
            "Fatigue",
            "Cough",
            "High Fever",
            "Headache",
            "Swelled Lymph Nodes",
            "Malaise",
            "Phlegm",
            "Throat Irritation",
            "Redness of Eyes",
            "Sinus Pressure",
            "Runny Nose",
            "Congestion",
            "Chest Pain",
            "Loss of Smell",
            "Muscle Pain"
        ],
        "Consulting Specialist": ["General Physician", "Pediatrician (for children)"],
        "Diagnostic Tests": [
            "Physical Examination",
            "Nasal Swab Test (in some cases)"
        ],
        "Treatments": [
            "Antihistamines",
            "Decongestants",
            "Pain Relievers",
            "Cough Syrups"
        ],
        "Precautions": [
            "Wash hands frequently with soap and water.",
            "Avoid close contact with sick individuals.",
            "Stay hydrated with warm fluids.",
            "Rest adequately to boost recovery.",
            "Use tissues or cover mouth while sneezing or coughing."
        ],
        "Resource Link": "https://www.mayoclinic.org/diseases-conditions/common-cold/symptoms-causes/syc-20351605",
        "Specialist": [
            "Dr. S. Aravind",
            "Velammal Medical College Hospital",
            "Madurai, Tamil Nadu",
            "Phone: +91 452 248 2666"
        ]
    },
    "Dengue": {
        "Symptoms": [
            "Skin Rash",
            "Chills",
            "Joint Pain",
            "Vomiting",
            "Fatigue",
            "High Fever",
            "Headache",
            "Nausea",
            "Loss of Appetite",
            "Pain Behind the Eyes",
            "Back Pain",
            "Malaise",
            "Muscle Pain",
            "Red Spots Over Body"
        ],
        "Consulting Specialist": ["General Physician", "Infectious Disease Specialist"],
        "Diagnostic Tests": [
            "Dengue NS1 Antigen Test",
            "Dengue IgM and IgG Antibody Test",
            "Complete Blood Count (CBC)",
            "Platelet Count"
        ],
        "Treatments": [
            "Pain Relievers",
            "Fluid Replacement Therapy",
            "Fever Management"
        ],
        "Precautions": [
            "Avoid mosquito bites by using repellents and nets.",
            "Wear protective clothing to reduce skin exposure.",
            "Eliminate stagnant water to prevent mosquito breeding.",
            "Stay hydrated by drinking plenty of fluids.",
            "Monitor symptoms and seek medical help if severe signs appear."
        ],
        "Resource Link": "https://www.who.int/health-topics/dengue-and-severe-dengue#tab=tab_1",
        "Specialist": [
            "Dr. Jeya Suriya",
            "General Physician/ Internal Medicine Specialist",
            "Apollo Speciality Hospitals KK Nagar",
            "80 Feet Road, KK Nagar, Madurai",
            "Phone: 8069305511"
        ]
    },
    "Diabetes": {
        "Symptoms": [
            "Fatigue",
            "Weight Loss",
            "Restlessness",
            "Lethargy",
            "Irregular Sugar Level",
            "Blurred and Distorted Vision",
            "Obesity",
            "Excessive Hunger",
            "Increased Appetite",
            "Polyuria"
        ],
        "Consulting Specialist": ["Endocrinologist", "Diabetologist"],
        "Diagnostic Tests": [
            "Fasting Blood Sugar Test",
            "HbA1c Test",
            "Oral Glucose Tolerance Test",
            "Random Blood Sugar Test"
        ],
        "Treatments": [
            "Insulin Therapy",
            "Oral Hypoglycemic Agents",
            "Lifestyle Modifications",
            "Dietary Management"
        ],
        "Precautions": [
            "Follow a balanced diet with controlled carbohydrate intake.",
            "Engage in regular physical activity.",
            "Monitor blood sugar levels consistently.",
            "Avoid skipping medications or insulin doses.",
            "Manage stress and get adequate sleep."
        ],
        "Resource Link": "https://diabetes.org/",
        "Specialist": [
            "Dr. M. Aravind Kumar",
            "Meenakshi Mission Hospital & Research Centre",
            "Madurai, Tamil Nadu",
            "Phone: +91 452 435 8888"
        ]
    },
    "Dimorphic Hemorrhoids (Piles)": {
        "Symptoms": [
            "Constipation",
            "Pain During Bowel Movements",
            "Pain in Anal Region",
            "Bloody Stool",
            "Irritation in Anus"
        ],
        "Consulting Specialist": ["General Surgeon", "Gastroenterologist"],
        "Diagnostic Tests": [
            "Digital Rectal Examination",
            "Anoscopy",
            "Proctoscopy",
            "Colonoscopy"
        ],
        "Treatments": [
            "Fiber Supplements",
            "Stool Softeners",
            "Topical Treatments",
            "Rubber Band Ligation",
            "Hemorrhoidectomy"
        ],
        "Precautions": [
            "Increase dietary fiber intake.",
            "Drink plenty of water to avoid constipation.",
            "Avoid straining during bowel movements.",
            "Exercise regularly to improve bowel function.",
            "Maintain proper hygiene in the anal area."
        ],
        "Resource Link": "https://www.nhs.uk/conditions/piles-haemorrhoids/",
        "Specialist": [
            "Dr. Madhusudhanan J",
            "Surgical Gastroenterologist",
            "Apollo Speciality Hospitals KK Nagar",
            "80 Feet Road, KK Nagar, Madurai",
            "Phone: 8069305511"
        ]
    },
    "Drug Reaction": {
        "Symptoms": [
            "Itching",
            "Skin Rash",
            "Stomach Pain",
            "Burning Micturition",
            "Spotting Urination"
        ],
        "Consulting Specialist": ["Dermatologist", "Allergist", "Immunologist"],
        "Diagnostic Tests": [
            "Skin Patch Test",
            "Blood Test",
            "Drug Provocation Test",
            "Biopsy (in severe cases)"
        ],
        "Treatments": [
            "Antihistamines",
            "Corticosteroids",
            "Epinephrine (for severe reactions)",
            "Discontinuation of the Drug"
        ],
        "Precautions": [
            "Inform your doctor about any known drug allergies.",
            "Always read and follow prescription instructions.",
            "Avoid self-medication or over-the-counter drugs without consultation.",
            "Carry an allergy card or medical ID for known drug allergies.",
            "Seek immediate medical attention for any unusual symptoms."
        ],
        "Resource Link": "https://www.mayoclinic.org/diseases-conditions/drug-allergy/symptoms-causes/syc-20371835",
        "Specialist": [
            "Dr. M. Padmavathy",
            "Dermatologist",
            "Apollo Hospital in Madurai East, Madurai",
            "Phone: 04448132958"
        ]
    },
    "Fungal Infection": {
        "Symptoms": [
            "Itching",
            "Skin Rash",
            "Nodal Skin Eruptions",
            "Dischromic Patches"
        ],
        "Consulting Specialist": ["Dermatologist", "Infectious Disease Specialist"],
        "Diagnostic Tests": [
            "Skin Scraping and Microscopy",
            "Fungal Culture",
            "KOH Test",
            "Biopsy (in severe cases)"
        ],
        "Treatments": [
            "Antifungal Creams",
            "Oral Antifungal Medications",
            "Medicated Shampoos",
            "Antifungal Powders"
        ],
        "Precautions": [
            "Keep skin clean and dry, especially in moist areas.",
            "Avoid sharing personal items like towels and combs.",
            "Wear breathable clothing to reduce sweat buildup.",
            "Use antifungal powders if prone to infections.",
            "Complete the full course of prescribed treatment."
        ],
        "Resource Link": "https://www.cdc.gov/fungal/index.html",
        "Specialist": [
            "Dr. Senthil Kumar. K",
            "Gem Skin, Hair and Laser Centre",
            "203, Vaigai Colony, Landmark: Near Ambiga Theatre, Madurai",
            "Phone: 01142246375"
        ]
    },
    "GERD": {
        "Symptoms": [
            "Stomach Pain",
            "Acidity",
            "Ulcers on Tongue",
            "Vomiting",
            "Cough",
            "Chest Pain"
        ],
        "Consulting Specialist": ["Gastroenterologist"],
        "Diagnostic Tests": [
            "Upper Endoscopy",
            "Esophageal pH Monitoring",
            "Barium Swallow Test",
            "Esophageal Manometry"
        ],
        "Treatments": [
            "Antacids",
            "Proton Pump Inhibitors",
            "H2 Receptor Blockers",
            "Lifestyle Modifications"
        ],
        "Precautions": [
            "Avoid spicy, fatty, or acidic foods that trigger symptoms.",
            "Eat smaller meals and avoid overeating.",
            "Avoid lying down immediately after meals.",
            "Elevate the head of your bed while sleeping.",
            "Maintain a healthy weight to reduce abdominal pressure."
        ],
        "Resource Link": "https://www.niddk.nih.gov/health-information/digestive-diseases/acid-reflux-ger-gerd-adults",
        "Specialist": [
            "Dr. A.C. Arun",
            "Lily Mission Hospital",
            "Madurai, Tamil Nadu",
            "Phone: 07947133110"
        ]
    },
    "Gastroenteritis": {
        "Symptoms": [
            "Vomiting",
            "Sunken Eyes",
            "Dehydration",
            "Diarrhoea"
        ],
        "Consulting Specialist": ["Gastroenterologist", "General Physician"],
        "Diagnostic Tests": [
            "Stool Culture",
            "Blood Test",
            "Electrolyte Panel",
            "Abdominal Ultrasound"
        ],
        "Treatments": [
            "Oral Rehydration Solutions",
            "Antidiarrheal Medications",
            "Antibiotics (in bacterial infections)",
            "Anti-nausea Medications"
        ],
        "Precautions": [
            "Maintain good hand hygiene, especially after using the bathroom.",
            "Drink plenty of fluids to prevent dehydration.",
            "Avoid consuming contaminated food or water.",
            "Rest to allow the body to recover.",
            "Avoid dairy and fatty foods until symptoms resolve."
        ],
        "Resource Link": "https://www.mayoclinic.org/diseases-conditions/arteriovenous-malformation/care-at-mayo-clinic/mac-20350547",
        "Specialist": [
            "Dr. Madhusudhanan J",
            "Surgical Gastroenterologist",
            "Apollo Speciality Hospitals KK Nagar",
            "80 Feet Road, KK Nagar, Madurai",
            "Phone: 8069305511"
        ]
    },
    "Heart Attack": {
        "Symptoms": [
            "Vomiting",
            "Breathlessness",
            "Sweating",
            "Chest Pain"
        ],
        "Consulting Specialist": ["Cardiologist", "General Physician"],
        "Diagnostic Tests": [
            "Electrocardiogram (ECG)",
            "Blood Tests (Troponin, CK-MB)",
            "Echocardiogram",
            "Coronary Angiography",
            "Stress Test"
        ],
        "Treatments": [
            "Antiplatelet Medications",
            "Anticoagulants",
            "Beta-Blockers"
        ],
        "Precautions": [
            "Maintain a healthy diet, low in fats and cholesterol.",
            "Exercise regularly to improve heart health.",
            "Control blood pressure and blood sugar levels.",
            "Avoid smoking and excessive alcohol consumption.",
            "Take prescribed medications regularly and attend follow-up appointments."
        ],
        "Resource Link": "https://www.heart.org/en/health-topics/heart-attack",
        "Specialist": [
            "Dr. Madhavan's Heart Centre",
            "2nd & 3rd Floor (Vikram Hospital Premises), Ring Road, Pandikovil Junction, Sivagangai Main Road, Madurai",
            "Phone: 9095211123"
        ]
    },
    "Hepatitis A": {
        "Symptoms": [
            "Joint Pain",
            "Vomiting",
            "Yellowish Skin",
            "Dark Urine",
            "Nausea",
            "Loss of Appetite",
            "Abdominal Pain",
            "Diarrhoea",
            "Mild Fever",
            "Yellowing of Eyes",
            "Muscle Pain"
        ],
        "Consulting Specialist": ["Hepatologist", "General Physician"],
        "Diagnostic Tests": [
            "Hepatitis A Virus (HAV) Antibody Test",
            "Liver Function Tests",
            "PCR (Polymerase Chain Reaction) Test",
            "Abdominal Ultrasound"
        ],
        "Treatments": [
            "Supportive Care (Rest, Hydration)",
            "Antiemetics (for nausea)",
            "Pain Relievers (for discomfort)"
        ],
        "Precautions": [
            "Avoid alcohol and drugs that strain the liver.",
            "Practice good hygiene, especially hand washing.",
            "Avoid sharing utensils, towels, or personal items.",
            "Consume a balanced, liver-friendly diet.",
            "Avoid contaminated food and water sources."
        ],
        "Resource Link": "https://www.cdc.gov/hepatitis-a/",
        "Specialist": [
            "Dr. Jagadeesapandian P",
            "General Surgeon",
            "Apollo Hospital in Madurai East, Madurai",
            "Phone: 04446315859"
        ]
    },
    "Hepatitis B": {
        "Symptoms": [
            "Itching",
            "Fatigue",
            "Lethargy",
            "Yellowish Skin",
            "Dark Urine",
            "Loss of Appetite",
            "Abdominal Pain",
            "Yellow Urine",
            "Yellowing of Eyes",
            "Malaise",
            "Receiving Blood Transfusion",
            "Receiving Unsterile Injections"
        ],
        "Consulting Specialist": ["Hepatologist", "Gastroenterologist"],
        "Diagnostic Tests": [
            "Hepatitis B Surface Antigen (HBsAg) Test",
            "Hepatitis B Surface Antibody (Anti-HBs) Test",
            "Hepatitis B e Antigen (HBeAg) Test",
            "Liver Function Tests",
            "Abdominal Ultrasound"
        ],
        "Treatments": [
            "Antiviral Medications (Tenofovir, Entecavir)",
            "Interferon Therapy",
            "Liver Transplantation (in severe cases)"
        ],
        "Precautions": [
            "Avoid sharing needles or personal hygiene items.",
            "Get vaccinated for Hepatitis B if not already done.",
            "Regular monitoring of liver function.",
            "Avoid alcohol consumption and hepatotoxic drugs.",
            "Use protection during sexual activity to prevent transmission."
        ],
        "Resource Link": "https://www.who.int/news-room/fact-sheets/detail/hepatitis-b",
        "Specialist": [
            "Dr. Madhusudhanan J",
            "Surgical Gastroenterologist",
            "Apollo Speciality Hospitals KK Nagar, 80 Feet Road, KK Nagar, Madurai",
            "Phone: 8069305511"
        ]
    },
    "Hepatitis C": {
        "Symptoms": [
            "Fatigue",
            "Yellowish Skin",
            "Nausea",
            "Loss of Appetite",
            "Yellowing of Eyes",
            "Family History"
        ],
        "Consulting Specialist": ["Hepatologist", "Gastroenterologist"],
        "Diagnostic Tests": [
            "Hepatitis C Virus (HCV) Antibody Test",
            "HCV RNA PCR Test",
            "Liver Function Tests",
            "Liver Biopsy",
            "Abdominal Ultrasound"
        ],
        "Treatments": [
            "Direct-Acting Antivirals (DAAs)",
            "Ribavirin (in some cases)",
            "Pegylated Interferon (in specific situations)",
            "Liver Transplantation (in severe cases)"
        ],
        "Precautions": [
            "Avoid alcohol consumption to protect liver health.",
            "Do not share needles or personal hygiene items.",
            "Use barrier methods during sexual activity to prevent transmission.",
            "Regular monitoring of liver function and viral load.",
            "Get vaccinated for Hepatitis A and B to prevent further liver complications."
        ],
        "Resource Link": "https://www.who.int/news-room/fact-sheets/detail/hepatitis-c",
        "Specialist": [
            "Dr. Jagadeesapandian P",
            "General Surgeon",
            "Apollo Hospital in Madurai East, Madurai",
            "Phone: 04446315859"
        ]
    },
    "Hepatitis D": {
        "Symptoms": [
            "Joint Pain",
            "Vomiting",
            "Fatigue",
            "Yellowish Skin",
            "Dark Urine",
            "Nausea",
            "Loss of Appetite",
            "Abdominal Pain",
            "Yellowing of Eyes"
        ],
        "Consulting Specialist": ["Hepatologist"],
        "Diagnostic Tests": [
            "Hepatitis D Virus Antibody (anti-HDV) Test",
            "Hepatitis D Virus RNA Test (HDV-RNA)",
            "Liver Function Tests",
            "Hepatitis B Surface Antigen (HBsAg) Test",
            "Abdominal Ultrasound",
            "Liver Biopsy or Fibroscan"
        ],
        "Treatments": [
            "Interferon Alfa",
            "Pegylated Interferon",
            "Lamivudine",
            "Adefovir",
            "Tenofovir"
        ],
        "Precautions": [
            "Get Vaccinated Against Hepatitis B",
            "Avoid Sharing Needles or Personal Items",
            "Practice Safe Sex",
            "Follow Medical Treatment and Care",
            "Avoid Alcohol and Hepatotoxic Substances"
        ],
        "Resource Link": "https://www.cdc.gov/hepatitis/d",
        "Specialist": [
            "Dr. S. Rajendran (Gastroenterologist & Hepatologist)",
            "Kavery Hospital Madurai",
            "93, North Veli Street, Madurai, Tamil Nadu 625001, India",
            "Phone: +91 452 2343101",
            "Specialty: Gastroenterology and Hepatology, specializing in liver diseases such as Hepatitis D, cirrhosis, and other gastrointestinal conditions."
        ]
    },
    "Hepatitis E": {
        "Symptoms": [
            "Joint Pain",
            "Vomiting",
            "Fatigue",
            "High Fever",
            "Yellowish Skin",
            "Dark Urine",
            "Nausea",
            "Loss of Appetite",
            "Abdominal Pain",
            "Yellowing of Eyes",
            "Acute Liver Failure",
            "Coma",
            "Stomach Bleeding"
        ],
        "Consulting Specialist": ["Hepatologist", "Infectious Disease Specialist"],
        "Diagnostic Tests": [
            "Hepatitis E Virus Antibody Test (anti-HEV)",
            "Hepatitis E Virus RNA Test (HEV-RNA)",
            "Liver Function Tests",
            "Abdominal Ultrasound",
            "Hepatitis A and Hepatitis B Tests",
            "Complete Blood Count (CBC)"
        ],
        "Treatments": [
            "Supportive Care",
            "Hydration",
            "Bed Rest",
            "Antiviral Treatment (e.g., Ribavirin in severe cases)",
            "Liver Transplant (in cases of acute liver failure)"
        ],
        "Precautions": [
            "Ensure Access to Clean Drinking Water",
            "Practice Good Hand Hygiene",
            "Avoid Consumption of Contaminated Food",
            "Avoid Unprotected Contact with Infected Individuals",
            "Seek Prompt Medical Care if Symptoms Occur"
        ],
        "Resource Link": "https://www.cdc.gov/hepatitis/e",
        "Specialist": [
            "Dr. S. Rajendran (Gastroenterologist & Hepatologist)",
            "Kavery Hospital Madurai",
            "93, North Veli Street, Madurai, Tamil Nadu 625001, India",
            "Phone: +91 452 2343101",
            "Specialty: Gastroenterology and Hepatology, specializing in liver diseases, including Hepatitis E, and providing medical management for hepatitis infections."
        ]
    },
     "Hypertension": {
        "Symptoms": [
            "Headache",
            "Chest Pain",
            "Dizziness",
            "Loss of Balance",
            "Lack of Concentration"
        ],
        "Consulting Specialist": ["Cardiologist"],
        "Diagnostic Tests": [
            "Blood Pressure Measurement",
            "Electrocardiogram (ECG/EKG)",
            "Blood Tests",
            "Urinalysis",
            "Echocardiogram",
            "Stress Test"
        ],
        "Treatments": [
            "Angiotensin-Converting Enzyme (ACE) Inhibitors",
            "Beta-Blockers",
            "Diuretics",
            "Calcium Channel Blockers",
            "Angiotensin II Receptor Blockers (ARBs)"
        ],
        "Precautions": [
            "Monitor Blood Pressure Regularly",
            "Maintain a Healthy Diet (Low Salt, Rich in Fruits and Vegetables)",
            "Engage in Regular Physical Activity",
            "Avoid Excessive Alcohol and Tobacco Use",
            "Manage Stress and Get Enough Sleep"
        ],
        "Resource Link": "https://www.cdc.gov/bloodpressure",
        "Specialist": [
            "Dr. S. Ramakrishnan (Consultant Cardiologist)",
            "Madurai Heart Centre",
            "75, South Veli Street, Madurai, Tamil Nadu 625001, India",
            "Phone: +91 452 4372920",
            "Specialty: Cardiology, specializing in hypertension, heart diseases, and cardiovascular health."
        ]
    },
    "Hyperthyroidism": {
        "Symptoms": [
            "Fatigue",
            "Mood Swings",
            "Weight Loss",
            "Restlessness",
            "Sweating",
            "Diarrhoea",
            "Fast Heart Rate",
            "Excessive Hunger",
            "Muscle Weakness",
            "Irritability",
            "Abnormal Menstruation"
        ],
        "Consulting Specialist": ["Endocrinologist"],
        "Diagnostic Tests": [
            "Thyroid Function Tests",
            "Free T4 and Free T3 Tests",
            "Thyroid Antibodies Test",
            "Ultrasound of the Thyroid",
            "Radioactive Iodine Uptake Test",
            "Thyroid Scan"
        ],
        "Treatments": [
            "Antithyroid Medications (e.g., Methimazole, Propylthiouracil)",
            "Radioactive Iodine Therapy",
            "Beta-Blockers",
            "Thyroidectomy (Surgical Removal of Thyroid)",
            "Iodine Supplements (in specific cases)"
        ],
        "Precautions": [
            "Follow Thyroid Medication as Prescribed",
            "Regularly Monitor Thyroid Hormone Levels",
            "Avoid Excessive Iodine Intake",
            "Manage Stress Effectively",
            "Maintain a Balanced Diet with Adequate Calcium and Vitamin D"
        ],
        "Resource Link": "https://www.thyroid.org/hyperthyroidism",
        "Specialist": [
            "Dr. V. R. Prabhu (Endocrinologist)",
            "Meenakshi Mission Hospital & Research Centre (MMHRC)",
            "15A, Aruppukkottai Road, Madurai, Tamil Nadu 625107, India",
            "Phone: +91 452 4355555",
            "Specialty: Endocrinology, specializing in thyroid disorders, including hyperthyroidism, and providing diagnosis, medication, and treatment options like radioactive iodine therapy or surgery."
        ]
    },
    "Hypoglycemia": {
        "Symptoms": [
            "Vomiting",
            "Fatigue",
            "Anxiety",
            "Sweating",
            "Headache",
            "Nausea",
            "Blurred and Distorted Vision",
            "Excessive Hunger",
            "Drying and Tingling Lips",
            "Slurred Speech",
            "Irritability",
            "Palpitations"
        ],
        "Consulting Specialist": ["Endocrinologist", "Diabetologist"],
        "Diagnostic Tests": [
            "Blood Glucose Test",
            "HbA1c Test",
            "Insulin Level Test",
            "C-Peptide Test",
            "Oral Glucose Tolerance Test (OGTT)",
            "Urine Ketone Test"
        ],
        "Treatments": [
            "Glucose Tablets or Gel",
            "Fast-Acting Carbohydrates (e.g., fruit juice, regular soda)",
            "Intravenous Dextrose (in severe cases)",
            "Glucagon Injection (for severe cases)",
            "Adjusting Medications (e.g., insulin dosage)"
        ],
        "Precautions": [
            "Monitor Blood Sugar Levels Regularly",
            "Carry Fast-Acting Carbohydrates (e.g., glucose tablets)",
            "Eat Small, Frequent Meals",
            "Avoid Skipping Meals or Overexertion",
            "Wear a Medical ID Bracelet for Emergency Situations"
        ],
        "Resource Link": "https://www.diabetes.org/diabetes/complications/hypoglycemia",
        "Specialist": [
            "Dr. R. K. Rajendran (Consultant Diabetologist & Endocrinologist)",
            "Madurai Diabetes Care Centre",
            "14, Ram Nagar, Madurai, Tamil Nadu 625020, India",
            "Phone: +91 452 4378293",
            "Specialty: Endocrinology and Diabetology, specializing in the management of diabetes and conditions like hypoglycemia, offering diagnosis, treatment plans, and monitoring."
        ]
    },
    "Hypothyroidism": {
        "Symptoms": [
            "Fatigue",
            "Weight Gain",
            "Cold Hands and Feet",
            "Mood Swings",
            "Lethargy",
            "Dizziness",
            "Puffy Face and Eyes",
            "Enlarged Thyroid",
            "Brittle Nails",
            "Swollen Extremities",
            "Depression",
            "Irritability",
            "Abnormal Menstruation"
        ],
        "Consulting Specialist": ["General Practitioner (GP)"],
        "Diagnostic Tests": [
            "Thyroid Function Tests",
            "Free T4 and Free T3 Tests",
            "Thyroid Antibodies Test",
            "Ultrasound of the Thyroid",
            "Radioactive Iodine Uptake Test",
            "Lipid Profile"
        ],
        "Treatments": [
            "Levothyroxine",
            "Liothyronine (in specific cases)",
            "Regular Thyroid Function Monitoring",
            "Nutritional Support (iodine, selenium if deficient)",
            "Lifestyle Adjustments (healthy diet, regular exercise)"
        ],
        "Precautions": [
            "Take Thyroid Medication as Prescribed",
            "Monitor Thyroid Function Regularly",
            "Maintain a Balanced Diet with Adequate Iodine",
            "Manage Stress and Ensure Adequate Sleep",
            "Avoid Extreme Temperatures and Stay Warm"
        ],
        "Resource Link": "https://www.thyroid.org/hypothyroidism",
        "Specialist": [
            "Dr. V. R. Prabhu (Endocrinologist)",
            "Meenakshi Mission Hospital & Research Centre (MMHRC)",
            "15A, Aruppukkottai Road, Madurai, Tamil Nadu 625107, India",
            "Phone: +91 452 4355555",
            "Specialty: Endocrinology, specializing in thyroid disorders including hypothyroidism, offering comprehensive diagnosis, hormone therapy, and long-term management."
        ]
    },
    "Impetigo": {
        "Symptoms": [
            "Skin Rash",
            "High Fever",
            "Blister",
            "Red Sore Around Nose",
            "Yellow Crust Ooze"
        ],
        "Consulting Specialist": ["Dermatologist"],
        "Diagnostic Tests": [
            "Clinical Diagnosis",
            "Bacterial Culture",
            "Gram Staining",
            "Skin Swab Test",
            "Blood Tests"
        ],
        "Treatments": [
            "Topical Antibiotics (e.g., Mupirocin, Retapamulin)",
            "Oral Antibiotics (e.g., Cephalexin, Dicloxacillin, Amoxicillin-Clavulanate)",
            "Gentle Cleaning of Affected Areas",
            "Avoiding Scratching to Prevent Spread",
            "Maintaining Good Hygiene Practices"
        ],
        "Precautions": [
            "Keep the Affected Area Clean and Dry",
            "Avoid Scratching or Touching Sores",
            "Wash Hands Frequently, Especially After Touching Infected Areas",
            "Use Prescribed Antibiotic Ointments as Directed",
            "Avoid Close Contact with Others Until Lesions Heal"
        ],
        "Resource Link": "https://www.cdc.gov/impetigo",
        "Specialist": [
            "Dr. S. Rajendran (Dermatologist)",
            "Kavery Hospital Madurai",
            "93, North Veli Street, Madurai, Tamil Nadu 625001, India",
            "Phone: +91 452 2343101",
            "Specialty: Dermatology, specializing in skin infections including impetigo, offering both medical and topical treatments."
        ]
    },
    "Jaundice": {
        "Symptoms": [
            "Itching",
            "Vomiting",
            "Fatigue",
            "Weight Loss",
            "High Fever",
            "Yellowish Skin",
            "Dark Urine",
            "Abdominal Pain"
        ],
        "Consulting Specialist": ["Gastroenterologist", "Hepatologist"],
        "Diagnostic Tests": [
            "Liver Function Tests",
            "Complete Blood Count (CBC)",
            "Abdominal Ultrasound",
            "Hepatitis A, B, and C Tests",
            "Coagulation Profile",
            "CT Scan or MRI of the Abdomen"
        ],
        "Treatments": [
            "Treating Underlying Cause",
            "Phototherapy",
            "Exchange Transfusion",
            "Hydration and Nutritional Support",
            "Medications",
            "Avoiding Alcohol and Hepatotoxic Substances"
        ],
        "Precautions": [
            "Follow Medical Treatment and Liver Care as Prescribed",
            "Avoid Alcohol Consumption",
            "Maintain a Healthy Diet, Low in Fat and Rich in Nutrients",
            "Get Adequate Rest and Sleep",
            "Avoid Exposure to Hepatotoxic Substances"
        ],
        "Resource Link": "https://www.cdc.gov/jaundice",
        "Specialist": [
            "Dr. S. Rajendran (Gastroenterologist & Hepatologist)",
            "Kavery Hospital Madurai",
            "93, North Veli Street, Madurai, Tamil Nadu 625001, India",
            "Phone: +91 452 2343101",
            "Specialty: Gastroenterology and Hepatology, specializing in liver diseases such as jaundice, cirrhosis, and hepatitis, providing medical management, diagnosis, and treatment for jaundice."
        ]
    },
    "Malaria": {
        "Symptoms": [
            "Chills",
            "Vomiting",
            "High Fever",
            "Sweating",
            "Headache",
            "Nausea",
            "Diarrhoea",
            "Muscle Pain"
        ],
        "Consulting Specialist": ["Infectious Disease Specialist"],
        "Diagnostic Tests": [
            "Blood Smear",
            "Rapid Diagnostic Test (RDT)",
            "PCR Test",
            "Complete Blood Count (CBC)",
            "Hemoglobin Level Test",
            "Liver Function Tests"
        ],
        "Treatments": [
            "Artemisinin-based Combination Therapies",
            "Chloroquine",
            "Quinine",
            "Mefloquine",
            "Primaquine",
            "Supportive Care"
        ],
        "Precautions": [
            "Use Insect Repellent and Sleep Under Mosquito Nets",
            "Take Antimalarial Medication as Prescribed, Especially When Traveling to Endemic Areas",
            "Avoid Stagnant Water Where Mosquitoes Breed",
            "Wear Protective Clothing to Prevent Mosquito Bites",
            "Seek Medical Attention Immediately if Symptoms Appear (e.g., fever, chills)"
        ],
        "Resource Link": "https://www.cdc.gov/malaria",
        "Specialist": [
            "Dr. S. S. Prabhakaran (Infectious Disease Specialist)",
            "Government Rajaji Hospital",
            "Madurai Main, Tamil Nadu 625020, India",
            "Phone: +91 452 2531170",
            "Specialty: Infectious Diseases, specializing in malaria, dengue, and other vector-borne diseases. The hospital offers diagnostic testing, medication, and supportive care for malaria."
        ]
    },
    "Migraine": {
        "Symptoms": [
            "Acidity",
            "Indigestion",
            "Headache",
            "Blurred and Distorted Vision",
            "Excessive Hunger",
            "Stiff Neck",
            "Depression",
            "Irritability",
            "Visual Disturbances"
        ],
        "Consulting Specialist": ["Neurologist"],
        "Diagnostic Tests": [
            "Clinical Diagnosis",
            "Neurological Examination",
            "MRI of the Brain",
            "CT Scan of the Brain",
            "Blood Tests",
            "Electroencephalogram (EEG)"
        ],
        "Treatments": [
            "Abortive Medications (e.g., Triptans, NSAIDs)",
            "Preventive Medications (e.g., Beta-blockers, Anticonvulsants)",
            "Lifestyle Modifications",
            "Stress Management",
            "Dietary Changes",
            "Biofeedback Therapy",
            "Caffeine (in some cases)"
        ],
        "Precautions": [
            "Identify and Avoid Migraine Triggers (e.g., certain foods, stress, bright lights)",
            "Maintain a Regular Sleep Schedule",
            "Stay Hydrated and Eat Balanced Meals",
            "Practice Stress Management Techniques (e.g., yoga, meditation)",
            "Follow Medication Plan as Prescribed by Your Doctor"
        ],
        "Resource Link": "https://americanmigrainefoundation.org",
        "Specialist": [
            "Dr. M. Kumaravel (Neurologist)",
            "Madurai Medical College & Government Rajaji Hospital",
            "Madurai Main, Tamil Nadu 625020, India",
            "Phone: +91 452 2531170",
            "Specialty: Neurology, specializing in headaches, migraines, and other neurological disorders. He provides diagnostic services, treatment plans, and preventive care for migraines."
        ]
    },
    "Osteoarthritis": {
        "Symptoms": [
            "Joint Pain",
            "Neck Pain",
            "Knee Pain",
            "Hip Joint Pain",
            "Swelling Joints",
            "Painful Walking"
        ],
        "Consulting Specialist": ["Rheumatologist"],
        "Diagnostic Tests": [
            "X-ray",
            "MRI",
            "Joint Fluid Analysis",
            "Blood Tests",
            "Physical Examination",
            "Ultrasound"
        ],
        "Treatments": [
            "Pain Relievers (e.g., Acetaminophen, NSAIDs)",
            "Physical Therapy",
            "Weight Management",
            "Joint Injections (e.g., Corticosteroids, Hyaluronic Acid)",
            "Surgery (e.g., Joint Replacement)",
            "Supportive Devices (e.g., braces, canes)",
            "Alternative Therapies (e.g., Glucosamine, Chondroitin)"
        ],
        "Precautions": [
            "Maintain a Healthy Weight to Reduce Joint Strain",
            "Engage in Low-Impact Exercises (e.g., swimming, walking)",
            "Use Joint Protection Techniques (e.g., proper posture, assistive devices)",
            "Apply Heat or Cold to Relieve Joint Pain",
            "Follow Medication and Physical Therapy Recommendations"
        ],
        "Resource Link": "https://www.cdc.gov/arthritis/basics/osteoarthritis.htm",
        "Specialist": [
            "Dr. S. K. Kumar (Orthopedic Surgeon)",
            "Meenakshi Mission Hospital & Research Centre (MMHRC)",
            "15A, Aruppukkottai Road, Madurai, Tamil Nadu 625107, India",
            "Phone: +91 452 4355555",
            "Specialty: Orthopedic Surgery, specializing in joint replacement, osteoarthritis, and musculoskeletal conditions. Offers medical management, physiotherapy, and surgical options such as joint replacement surgeries."
        ]
    },
    "Paralysis (Brain Hemorrhage)": {
        "Symptoms": [
            "Vomiting",
            "Headache",
            "Weakness of One Body Side",
            "Altered Sensorium"
        ],
        "Consulting Specialist": ["Neurologist", "Neurosurgeon"],
        "Diagnostic Tests": [
            "CT Scan of the Brain",
            "MRI of the Brain",
            "Neurological Examination",
            "Blood Tests",
            "Angiogram",
            "Electroencephalogram (EEG)"
        ],
        "Treatments": [
            "Emergency Medical Care",
            "Surgical Intervention (e.g., Craniotomy, Hematoma Evacuation)",
            "Antihypertensive Medications",
            "Anticonvulsants (if seizures occur)",
            "Physical Therapy",
            "Occupational Therapy",
            "Speech Therapy",
            "Pain Management",
            "Psychological Support"
        ],
        "Precautions": [
            "Follow Post-Stroke Rehabilitation and Therapy Plans",
            "Manage Blood Pressure and Prevent Further Hemorrhages",
            "Avoid Smoking and Excessive Alcohol Consumption",
            "Ensure a Healthy Diet to Support Recovery and Prevent Complications",
            "Monitor for Signs of Secondary Complications (e.g., infections, blood clots)"
        ],
        "Resource Link": "https://www.ninds.nih.gov/health-information/disorders/brain-hemorrhage",
        "Specialist": [
            "Dr. S. Rajasekaran (Neurologist)",
            "Madurai Meenakshi Mission Hospital & Research Centre (MMHRC)",
            "15A, Aruppukkottai Road, Madurai, Tamil Nadu 625107, India",
            "Phone: +91 452 4355555",
            "Specialty: Neurology, specializing in brain hemorrhage, stroke, and paralysis. Provides diagnostic services, medical treatment, rehabilitation, and long-term management for patients recovering from brain hemorrhages."
        ]
    },
    "Peptic Ulcer Disease": {
        "Symptoms": [
            "Vomiting",
            "Indigestion",
            "Loss of Appetite",
            "Abdominal Pain",
            "Passage of Gases",
            "Internal Itching"
        ],
        "Consulting Specialist": ["Gastroenterologist", "Hepatologist"],
        "Diagnostic Tests": [
            "Endoscopy (EGD)",
            "H. pylori Testing (Breath, Blood, or Stool Test)",
            "Barium Swallow X-ray",
            "Complete Blood Count (CBC)",
            "Urea Breath Test"
        ],
        "Treatments": [
            "Proton Pump Inhibitors (e.g., Omeprazole, Esomeprazole)",
            "H2-Receptor Antagonists (e.g., Ranitidine, Famotidine)",
            "Antibiotics (for H. pylori infection)",
            "Antacids",
            "Sucralfate",
            "Avoidance of Irritants (e.g., alcohol, smoking)",
            "Dietary Modifications",
            "Surgery (in severe or complicated cases)"
        ],
        "Precautions": [
            "Avoid Spicy, Acidic, and Fatty Foods that Can Irritate the Stomach",
            "Take Medications as Prescribed to Reduce Stomach Acid",
            "Limit Alcohol and Caffeine Consumption",
            "Avoid Smoking, as it Can Delay Healing",
            "Eat Smaller, More Frequent Meals to Reduce Stomach Stress"
        ],
        "Resource Link": "https://www.niddk.nih.gov/health-information/digestive-diseases/peptic-ulcer-disease",
        "Specialist": [
            "Dr. S. Rajendran (Gastroenterologist)",
            "Kavery Hospital Madurai",
            "93, North Veli Street, Madurai, Tamil Nadu 625001, India",
            "Phone: +91 452 2343101",
            "Specialty: Gastroenterology, specializing in digestive diseases, including peptic ulcers, acid reflux, and gastrointestinal disorders. Offers diagnosis, medication management, and lifestyle advice for managing ulcers."
        ]
    },
    "Pneumonia": {
        "Symptoms": [
            "Chills",
            "Fatigue",
            "Cough",
            "High Fever",
            "Breathlessness",
            "Sweating",
            "Malaise",
            "Phlegm",
            "Chest Pain",
            "Fast Heart Rate",
            "Rusty Sputum"
        ],
        "Consulting Specialist": ["Pulmonologist", "Infectious Disease Specialist"],
        "Diagnostic Tests": [
            "Chest X-ray",
            "Blood Cultures",
            "Sputum Culture",
            "Complete Blood Count (CBC)",
            "Pulse Oximetry"
        ],
        "Treatments": [
            "Antibiotics (for bacterial pneumonia)",
            "Antiviral Medications (for viral pneumonia)",
            "Antifungal Medications (for fungal pneumonia)",
            "Oxygen Therapy",
            "Pain Relievers",
            "Cough Medicine (for symptomatic relief)",
            "Hospitalization (in severe cases)",
            "Hydration and Nutritional Support"
        ],
        "Precautions": [
            "Follow Prescribed Antibiotics or Medications to Treat Infection",
            "Rest and Stay Hydrated to Support Recovery",
            "Practice Good Hygiene, Including Handwashing and Coughing Etiquette",
            "Avoid Smoking and Secondhand Smoke Exposure",
            "Get Vaccinated Against Pneumonia (e.g., pneumococcal vaccine)"
        ],
        "Resource Link": "https://www.cdc.gov/pneumonia",
        "Specialist": [
            "Dr. M. A. Joseph (Pulmonologist)",
            "Meenakshi Mission Hospital & Research Centre (MMHRC)",
            "15A, Aruppukkottai Road, Madurai, Tamil Nadu 625107, India",
            "Phone: +91 452 4355555",
            "Specialty: Pulmonology, specializing in respiratory infections like pneumonia, chronic respiratory diseases, and lung health. Provides diagnosis, medical treatment, and care for severe respiratory conditions."
        ]
    },
    "Psoriasis": {
        "Symptoms": [
            "Skin Rash",
            "Joint Pain",
            "Skin Peeling",
            "Silver-like Dusting",
            "Small Dents in Nails",
            "Inflammatory Nails"
        ],
        "Consulting Specialist": ["Dermatologist", "Rheumatologist"],
        "Diagnostic Tests": [
            "Physical Examination",
            "Skin Biopsy",
            "Blood Tests (to rule out other conditions)",
            "Scalp Biopsy (if scalp psoriasis is suspected)",
            "X-rays (to check for joint involvement in psoriatic arthritis)"
        ],
        "Treatments": [
            "Topical Steroids",
            "Topical Vitamin D Analogues",
            "Phototherapy",
            "Systemic Medications (e.g., Methotrexate, Cyclosporine)",
            "Biologic Drugs",
            "Moisturizers",
            "Lifestyle Modifications"
        ],
        "Precautions": [
            "Follow Prescribed Topical Treatments and Medications",
            "Moisturize Skin Regularly to Prevent Dryness and Flare-ups",
            "Avoid Stress, as It Can Trigger or Worsen Symptoms",
            "Protect Skin from Excessive Sun Exposure",
            "Maintain a Healthy Diet and Stay Hydrated"
        ],
        "Resource Link": "https://www.psoriasis.org",
        "Specialist": [
            "Dr. S. Rajendran (Dermatologist)",
            "Kavery Hospital Madurai",
            "93, North Veli Street, Madurai, Tamil Nadu 625001, India",
            "Phone: +91 452 2343101",
            "Specialty: Dermatology, specializing in skin conditions like psoriasis, offering diagnosis, topical treatments, phototherapy, and advanced care options for managing psoriasis."
        ]
    },
    "Tuberculosis": {
        "Symptoms": [
            "Chills",
            "Vomiting",
            "Fatigue",
            "Weight Loss",
            "Cough",
            "High Fever",
            "Breathlessness",
            "Sweating",
            "Loss of Appetite",
            "Mild Fever",
            "Yellowing of Eyes",
            "Swelled Lymph Nodes",
            "Malaise",
            "Phlegm",
            "Chest Pain",
            "Blood in Sputum"
        ],
        "Consulting Specialist": ["Infectious Disease Specialist", "Pulmonologist"],
        "Diagnostic Tests": [
            "Tuberculin Skin Test (TST)",
            "Chest X-ray",
            "Sputum Smear Microscopy",
            "Sputum Culture",
            "Molecular Tests (e.g., GeneXpert)"
        ],
        "Treatments": [
            "Antitubercular Medications (e.g., Rifampin, Isoniazid, Pyrazinamide, Ethambutol)",
            "Directly Observed Therapy (DOT)",
            "Nutritional Support",
            "Oxygen Therapy (if needed)",
            "Isolation (to prevent transmission)",
            "Regular Monitoring and Follow-up"
        ],
        "Precautions": [
            "Complete the Full Course of TB Medication as Prescribed",
            "Cover Your Mouth and Nose When Coughing or Sneezing",
            "Avoid Close Contact with Others, Especially Vulnerable Individuals",
            "Ensure Proper Ventilation in Living Spaces",
            "Get Regular Follow-up Check-ups to Monitor Treatment Progress"
        ],
        "Resource Link": "https://www.cdc.gov/tb",
        "Specialist": [
            "Dr. M. A. Joseph (Pulmonologist)",
            "Meenakshi Mission Hospital & Research Centre (MMHRC)",
            "15A, Aruppukkottai Road, Madurai, Tamil Nadu 625107, India",
            "Phone: +91 452 4355555",
            "Specialty: Pulmonology, specializing in respiratory diseases including tuberculosis (TB). Provides diagnostic services, TB treatment (including directly observed therapy), and management for respiratory infections."
        ]
    },
    "Typhoid": {
        "Symptoms": [
            "Chills",
            "Vomiting",
            "Fatigue",
            "High Fever",
            "Headache",
            "Nausea",
            "Constipation",
            "Abdominal Pain",
            "Diarrhoea",
            "Toxic Look (Typhos)",
            "Belly Pain"
        ],
        "Consulting Specialist": ["Infectious Disease Specialist", "Gastroenterologist"],
        "Diagnostic Tests": [
            "Blood Culture",
            "Widal Test",
            "Stool Culture",
            "Urine Culture",
            "Complete Blood Count (CBC)"
        ],
        "Treatments": [
            "Antibiotics",
            "Hydration Therapy",
            "Nutritional Support",
            "Antipyretics",
            "Rest",
            "Vaccination for Prevention"
        ],
        "Precautions": [
            "Drink Only Safe, Clean Water and Avoid Contaminated Food",
            "Practice Good Hand Hygiene, Especially Before Eating",
            "Follow the Full Course of Antibiotics as Prescribed",
            "Avoid Eating Raw or Undercooked Foods, Particularly in High-Risk Areas",
            "Get Vaccinated if Traveling to Typhoid-Endemic Regions"
        ],
        "Resource Link": "https://www.cdc.gov/typhoid-fever",
        "Specialist": [
            "Dr. R. V. Rajesh (Infectious Disease Specialist)",
            "Government Rajaji Hospital",
            "Madurai Main, Tamil Nadu 625020, India",
            "Phone: +91 452 2531170",
            "Specialty: Infectious Diseases, specializing in the diagnosis and treatment of bacterial infections like typhoid fever. Provides medical management, including antibiotics and supportive care."
        ]
    },
    "Urinary Tract Infection": {
        "Symptoms": [
            "Burning Micturition",
            "Bladder Discomfort",
            "Foul Smell of Urine",
            "Continuous Feel of Urine"
        ],
        "Consulting Specialist": ["Urologist", "Infectious Disease Specialist"],
        "Diagnostic Tests": [
            "Urinalysis",
            "Urine Culture",
            "Blood Tests (e.g., CBC, kidney function tests)",
            "Urine Microscopy",
            "Imaging (e.g., Ultrasound, CT scan if complicated UTI)"
        ],
        "Treatments": [
            "Antibiotics",
            "Pain Relievers",
            "Increased Fluid Intake",
            "Cranberry Products",
            "Probiotics",
            "Good Hygiene Practices"
        ],
        "Precautions": [
            "Drink Plenty of Water to Help Flush Bacteria from the Urinary Tract",
            "Urinate Regularly and Do Not Hold It In",
            "Wipe from Front to Back to Avoid Spreading Bacteria",
            "Avoid Using Harsh Feminine Products that Can Irritate the Urinary Tract",
            "Complete the Full Course of Antibiotics as Prescribed"
        ],
        "Resource Link": "https://www.niddk.nih.gov/health-information/urologic-diseases/urinary-tract-infections",
        "Specialist": [
            "Dr. R. R. Rajasekar (Urologist)",
            "Kumaran Hospital",
            "32, North Masi Street, Madurai, Tamil Nadu 625001, India",
            "Phone: +91 452 2349000",
            "Specialty: Urology, specializing in urinary tract infections, kidney issues, bladder conditions, and other urological disorders. Provides comprehensive diagnosis and treatment for UTIs, including medication and preventive advice."
        ]
    },
    "Varicose Veins": {
        "Symptoms": [
            "Fatigue",
            "Cramps",
            "Bruising",
            "Obesity",
            "Swollen Legs",
            "Swollen Blood Vessels",
            "Prominent Veins on Calf"
        ],
        "Consulting Specialist": ["Vascular Surgeon", "Phlebologist"],
        "Diagnostic Tests": [
            "Physical Examination",
            "Duplex Ultrasound",
            "Venography",
            "Magnetic Resonance Venography (MRV)",
            "Doppler Ultrasound"
        ],
        "Treatments": [
            "Compression Stockings",
            "Sclerotherapy",
            "Laser Treatment",
            "Endovenous Ablation Therapy",
            "Vein Stripping",
            "Ambulatory Phlebectomy",
            "Lifestyle Changes"
        ],
        "Precautions": [
            "Avoid Prolonged Standing or Sitting to Reduce Pressure on the Veins",
            "Elevate Your Legs Regularly to Improve Circulation",
            "Wear Compression Stockings as Recommended by Your Doctor",
            "Maintain a Healthy Weight to Reduce Strain on the Legs",
            "Engage in Regular Physical Activity, such as Walking or Swimming"
        ],
        "Resource Link": "https://www.heart.org/en/health-topics/vascular-health/varicose-veins",
        "Specialist": [
            "Dr. N. M. Senthil Kumar (Vascular Surgeon)",
            "Madurai Meenakshi Mission Hospital & Research Centre (MMHRC)",
            "15A, Aruppukkottai Road, Madurai, Tamil Nadu 625107, India",
            "Phone: +91 452 4355555",
            "Specialty: Vascular Surgery, specializing in the diagnosis and treatment of varicose veins, offering medical management, sclerotherapy, and advanced surgical options for vein disorders."
        ]
    },
    "Vertigo": {
        "Symptoms": [
            "Vomiting",
            "Headache",
            "Nausea",
            "Spinning Movements",
            "Loss of Balance",
            "Unsteadiness"
        ],
        "Consulting Specialist": ["Neurologist", "Otolaryngologist (ENT Specialist)"],
        "Diagnostic Tests": [
            "Physical Examination",
            "Dix-Hallpike Test",
            "Audiometry (Hearing Tests)",
            "MRI or CT Scan (to rule out neurological causes)",
            "Blood Tests (to check for infections or other conditions)"
        ],
        "Treatments": [
            "Vestibular Rehabilitation Therapy",
            "Antihistamines",
            "Anticholinergics",
            "Benzodiazepines",
            "Betahistine",
            "Lifestyle Modifications"
        ],
        "Precautions": [
            "Avoid Sudden Head Movements or Position Changes",
            "Stay Hydrated to Prevent Dizziness Caused by Dehydration",
            "Perform Balance Exercises as Recommended by a Healthcare Professional",
            "Avoid Driving or Operating Heavy Machinery When Experiencing Vertigo",
            "Follow the Treatment Plan Prescribed by Your Doctor to Manage Symptoms"
        ],
        "Resource Link": "https://www.nidcd.nih.gov/health/vertigo",
        "Specialist": [
            "Dr. R. V. Rajesh (Neurologist)",
            "Madurai Medical College & Government Rajaji Hospital",
            "Madurai Main, Tamil Nadu 625020, India",
            "Phone: +91 452 2531170",
            "Specialty: Neurology, specializing in balance disorders, vertigo, and dizziness, providing diagnostic evaluation and treatment for vertigo-related conditions."
        ]
    }
}

# ===========================================

def predict_symptoms(input_data):
    reshaped_data = input_data.values.reshape((1, 1, input_data.shape[1]))
    pred_probs = rnn_model.predict(reshaped_data)
    
    pred_classes = np.argmax(pred_probs, axis=1)
    predicted_labels = encoder.inverse_transform(np.eye(pred_probs.shape[1])[pred_classes])
    
    accuracy = np.max(pred_probs)
    
    return predicted_labels[0][0], accuracy

def display_disease_info(prediction):
    info = disease_info.get(prediction, {})

    st.title(f"{prediction} - Disease Overview")

    tabs = st.tabs(["Overview", "Tests & Treatment", "Safety Measures", "Specialist Contact Details"])

    with tabs[0]:
        st.subheader("Common Symptoms")
        symptoms = info.get('Symptoms', [])
        
        if symptoms:
            for item in symptoms:
                st.markdown(f"- {item}")
        else:
            st.warning("No symptoms available.")

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Consulting Specialist")
            specialist = info.get('Consulting Specialist', [])
            if specialist:
                for item in specialist:
                    st.markdown(f"- {item}")
            else:
                st.warning("No specialist available.")
        with col2:
            st.subheader("Suggested Tests")
            tests = info.get('Diagnostic Tests', [])
        
            if tests:
                for item in tests:
                    st.markdown(f"- {item}")
            else:
                st.warning("No tests available.")

        st.subheader("Common Treatments")
        treatments = info.get('Treatments', [])
        
        if treatments:
            for item in treatments:
                st.markdown(f"- {item}")
        else:
            st.warning("No treatments available.")

    with tabs[2]:
        st.subheader("Precautions")
        precautions = info.get('Precautions', [])
        
        if precautions:
            for item in precautions:
                st.markdown(f"⚠️ {item}")
        else:
            st.warning("No precautions available.")

    with tabs[3]:
        st.write(f"🔗 To Learn More about {prediction}: [Click Here]({info.get('Resource Link', '#')})")
        for line in info.get('Specialist', []):
            st.write(f"**{line}**")

# Streamlit app
def main():
    selected = option_menu(
            menu_title=None, 
            options=["Home", "About CDSS", "Use Tool"],  
            icons=["house", "info-circle", "gear"],  
            menu_icon="cast",  
            default_index=0, 
            orientation="horizontal",
        )

    if selected == "Home":
        home_page()
        st.markdown("---")
        st.markdown("### Get Started")
        st.markdown("Navigate to the **Use Tool** page to start predicting diseases based on symptoms.")

        st.markdown("""
        <hr style='border:1px solid gray'>
        <p style='text-align: center;'>© 2025 CDSS | Contact: <a href="mailto:arrchith26@gmail.com" style="text-decoration: none; color: gray;">arrchith26@gmail.com</a></p>
        """, unsafe_allow_html=True)

    elif selected == "About CDSS":
        about_cdss()

    elif selected == "Use Tool":
        st.write("Please select your symptoms below:")

        symptom_list = features[:-1].values.flatten().tolist() 
        selected_symptoms = st.multiselect("Symptoms:", symptom_list, default=[])

        input_data = pd.DataFrame(np.zeros((1, num_features)), columns=symptom_list)

        for symptom in selected_symptoms:
            input_data[symptom] = 1

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                disease, accuracy = predict_symptoms(input_data)
                st.success(f"""
                        **Predicted Disease:** {disease} 
                        
                        **Level of Confidence:** {accuracy:.2%}
                            
                        """)
                display_disease_info(disease)
            st.markdown("""
            <hr style='border:1px solid gray'>
            <p style='text-align: center;'>© 2025 CDSS | Contact: <a href="mailto:arrchith26@gmail.com" style="text-decoration: none; color: gray;">arrchith26@gmail.com</a></p>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()