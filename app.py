import streamlit as st
import joblib
from preProcess import ArabicTextPreprocessor

st.set_page_config(page_title="Arabic Emotion Analysis", page_icon=":smile:")

# Load the model
model_RF = joblib.load('RF_model.pkl')
model_SVM = joblib.load('SVM_model.pkl')
model_KNN = joblib.load('KNN_model.pkl')

# Initialize the text preprocessor
text_preprocessor = ArabicTextPreprocessor()

# Streamlit app
st.title("Arabic Text Emotion Analysis 🤔")

# Text input
user_input = st.text_area("Enter Arabic text for emotion analysis:")

if st.button("Analyze"):
    if user_input:
        # Preprocess the input text
        preprocessed_text = text_preprocessor.preprocess_text(user_input)
        
        # Predict the emotion
        result = model_RF.predict([preprocessed_text])
        result1 = model_SVM.predict([preprocessed_text])
        result2 = model_KNN.predict([preprocessed_text])
        
        # Display the result
        st.write(f"Predicted Emotion RF: {result[0]}")
        st.write(f"Predicted Emotion KNN: {result2[0]}")
        st.write(f"Predicted Emotion SVM: {result1[0]}")
        st.write(f"Original Text: {user_input}")
    else:
        st.write("Please enter some text for analysis.")

if st.checkbox("Show Preprocessed Text"):
    if user_input:
        preprocessed_text = text_preprocessor.preprocess_text(user_input)
        st.write(f"Preprocessed Text: {preprocessed_text}")
