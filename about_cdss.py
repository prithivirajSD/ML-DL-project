import streamlit as st

def about_cdss():
    """Display information about the Clinical Decision Support System."""
    st.title("About Clinical Decision Support System (CDSS)")
    
    st.markdown("""
    ## What is CDSS?
    
    A Clinical Decision Support System (CDSS) is a health information technology system designed to provide 
    physicians and other health professionals with clinical decision support - that is, assistance with 
    clinical decision-making tasks.
    
    Our CDSS uses machine learning (specifically Recurrent Neural Networks) to help medical professionals 
    predict possible diagnoses based on patient symptoms.
    
    ## How It Works
    
    1. **Input**: Healthcare providers enter patient symptoms into the system
    2. **Processing**: Our RNN model analyzes the symptoms
    3. **Output**: The system provides potential diagnoses ranked by probability
    
    ## Model Information
    
    Our system uses a Recurrent Neural Network (RNN) trained on a comprehensive dataset of medical symptoms 
    and their corresponding diagnoses. The model achieves high accuracy in predicting conditions based on 
    symptom patterns.
    
    ### Technical Details:
    - **Model Type**: Recurrent Neural Network
    - **Training Data**: Curated healthcare dataset with symptom-disease mappings
    - **Features**: {len(st.session_state.get('num_features', 0))} unique symptoms
    - **Diseases**: Multiple conditions across various medical specialties
    
    ## Benefits
    
    - **Improved Diagnosis**: Assists medical professionals in considering all possible conditions
    - **Efficiency**: Reduces time to diagnosis
    - **Education**: Helps trainees learn symptom-disease relationships
    - **Second Opinion**: Provides an algorithmic perspective to complement clinical judgment
    
    ## Limitations
    
    - This tool is designed to **assist** healthcare providers, not replace clinical judgment
    - Results should always be verified by qualified medical professionals
    - The system is based on statistical patterns and may not account for rare presentations
    
    ## Responsible Use
    
    This tool should be used by healthcare professionals as a supplement to clinical expertise and 
    standard diagnostic procedures. It is not intended for self-diagnosis by patients.
    """)
    
    st.markdown("---")
    
    st.info("""
    **Disclaimer**: This Clinical Decision Support System is intended for use by healthcare professionals only. 
    The predictions made by this system should not be considered as final diagnoses. 
    Always consult with qualified medical professionals for proper diagnosis and treatment.
    """)
    
    st.markdown("""
    <hr style='border:1px solid gray'>
    <p style='text-align: center;'>© 2025 CDSS | Contact: <a href="mailto:arrchith26@gmail.com" style="text-decoration: none; color: gray;">arrchith26@gmail.com</a></p>
    """, unsafe_allow_html=True)