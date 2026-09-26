import streamlit as st
from textblob import TextBlob
import spacy
import PyPDF2

# Load the NLP model once
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_md")

nlp = load_model()

# Build the Web UI
st.title("TextMiner: NLP Analysis Tool")
st.write("Upload a document or enter text manually to analyze sentiment and extract key entities.")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
user_input = ""

# 2. Extract text from the uploaded file
if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        user_input = uploaded_file.getvalue().decode("utf-8")
        st.info("Text file loaded successfully.")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                user_input += extracted_text + "\n"
        st.info("PDF file loaded successfully.")
else:
    # 3. Fallback to manual text input if no file is uploaded
    user_input = st.text_area("Or input text manually", height=200)

if st.button("Analyze"):
    if user_input.strip():
        # Sentiment Analysis
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity
        
        st.subheader("1. Sentiment Analysis")
        if sentiment > 0.1:
            st.success(f"Positive (Score: {sentiment:.2f})")
        elif sentiment < -0.1:
            st.error(f"Negative (Score: {sentiment:.2f})")
        else:
            st.info(f"Neutral (Score: {sentiment:.2f})")

        # Entity Extraction
        st.subheader("2. Named Entities")
        doc = nlp(user_input)
        
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        if entities:
            st.table({"Entity": [e[0] for e in entities], "Label": [e[1] for e in entities]})
        else:
            st.write("No named entities found.")
    else:
        st.warning("Please upload a file or enter some text to analyze.")