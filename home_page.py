import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd


def home_page():
    st.markdown(
        """
        <h2 style="text-align: center;">Welcome to the Clinical Decision Support System (CDSS)</h2>
        <p style="text-align: center;">Empowering healthcare through advanced technology and AI-driven solutions.</p>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
    """
    Clinical Decision Support Systems (CDSS) play a vital role in modern healthcare by **facilitating accurate and timely predictions**. 
    These predictions are instrumental in enabling healthcare professionals to make informed decisions, improving patient outcomes, and 
    reducing diagnostic errors. By leveraging advanced machine learning models, CDSS ensures precision in identifying potential health 
    issues early, thereby paving the way for **personalized treatment plans** and enhancing the overall quality of care.
    """
    )

    st.markdown("### Key Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🔍 **Accurate Disease Prediction**")
        st.write("""
        Harness the power of cutting-edge machine learning models to predict diseases based on a comprehensive 
        dataset of 132 symptoms. Achieving a **precision of over 90%**, this feature ensures reliable and timely 
        identification of medical conditions, helping clinicians make informed decisions with confidence.
        """)

    with col2:
        st.info("📊 **Data-Driven Insights**")
        st.write("""
        Leverages extensive historical patient data and trends to derive meaningful insights. By analyzing patterns 
        and relationships in the data, the system aids in uncovering critical information that supports personalized 
        treatment planning and enhances diagnostic accuracy for diverse medical scenarios.
        """)

    with col3:
        st.info("⚙️ **Advanced Technology**")
        st.write("""
        Built on **Recurrent Neural Networks (RNNs)** and **Bidirectional LSTMs**, the system excels at processing 
        sequential data such as patient histories and symptom progression. This advanced architecture ensures 
        the model can effectively capture temporal dependencies and deliver unparalleled performance in clinical 
        decision support tasks.
        """)

    st.markdown("### Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction Accuracy", "90.16%")
    col2.metric("Number of Symptoms Analyzed", "132")
    col3.metric("Diseases Diagnosed", "40+")

    st.markdown("### Why CDSS?")
    col1, col2 = st.columns([2, 2])
    with col1:
        st.markdown("""
            - **Improve Diagnostic Precision:**  
            Provides accurate predictions by analyzing complex patterns in patient symptoms, ensuring precise diagnoses and reducing the likelihood of errors.

            - **Facilitate Treatment Planning:**  
            Helps clinicians make informed decisions about treatment strategies by offering insights based on historical data and predictive analytics.

            - **Enable Early Detection:**  
            Identifies potential diseases at an early stage, improving patient outcomes and enabling timely intervention to prevent complications.

            - **Leverage Machine Learning and Deep Learning Models:**  
            Utilizes advanced techniques like Recurrent Neural Networks (RNNs) and Bidirectional LSTMs to analyze sequential data, capturing temporal patterns critical for clinical decision-making.

            """)

    with col2:
        image = Image.open("images/architecture.png")
        st.image(image, caption="Architecture Diagram", use_column_width=True)

    # Interactive Feature Selector
    st.markdown("## Explore Features of CDSS")

    feature = st.radio(
        "Select a Feature to Learn More:",
        ["Disease Prediction", "Symptom Analysis", "Analytics Dashboard"]
    )

    if feature == "Disease Prediction":
        st.subheader("🔍 Disease Prediction")
        st.write("""
            The Disease Prediction module uses advanced Recurrent Neural Networks (RNN) and Bidirectional LSTMs to:
            - Analyze patient symptoms and historical medical data.
            - Provide predictions with over 90% accuracy.
            - Support clinicians in diagnosing complex and rare diseases effectively.
        """)
        st.image("images/disease_prediction.png", caption="Accurate Disease Prediction", use_column_width=True)

    elif feature == "Symptom Analysis":
        st.subheader("🩺 Symptom Analysis")
        st.write("""
            The Symptom Analysis module helps clinicians:
            - Examine correlations between symptoms and diseases.
            - Detect early warning signs for critical conditions.
            - Understand temporal patterns in symptom evolution for personalized care.
        """)
        st.image("images/symptom_analysis.png", caption="Symptom Analysis", use_column_width=True)

    elif feature == "Analytics Dashboard":
        st.subheader("📊 Analytics Dashboard")
        st.write("""
            The Analytics Dashboard provides:
            - Real-time data visualization of patient trends and disease predictions.
            - Insights into performance metrics, such as accuracy, F1-score, and recall of the prediction models.
            - A centralized hub to track model outputs and decision-support actions.
        """)
        st.image("images/analytics_dashboard.png", caption="Comprehensive Analytics Dashboard", use_column_width=True)


    st.markdown("### Model Accuracy Comparison")
    df = pd.DataFrame({
        "Models": ["Decision Tree", "SVM", "KNN", "Logistic Regression", "RNN", "Proposed" ],
        "Accuracy": [68.85, 60.66, 88.52, 77.05, 61.08, 93.16],
    })
    fig = px.bar(df, x="Models", y="Accuracy", color="Models", title="Model Accuracy Comparison")
    st.plotly_chart(fig)