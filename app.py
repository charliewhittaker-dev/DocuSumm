import streamlit as st
from textblob import TextBlob
import spacy

# Load the NLP model once to optimize performance
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# Build the Web UI
st.title("TextMiner: NLP Analysis Tool")
st.write("Enter text below to analyze sentiment and extract key entities.")

# Text input box
user_input = st.text_area("Input Text", height=200)

if st.button("Analyze"):
    if user_input:
        #Sentiment Analysis
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity
        
        st.subheader("1. Sentiment Analysis")
        if sentiment > 0.1:
            st.success(f"Positive (Score: {sentiment:.2f})")
        elif sentiment < -0.1:
            st.error(f"Negative (Score: {sentiment:.2f})")
        else:
            st.info(f"Neutral (Score: {sentiment:.2f})")

        #Entity Extraction
        st.subheader("2. Named Entities")
        doc = nlp(user_input)
        
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        if entities:
            # Format as a table for Streamlit
            st.table({"Entity": [e[0] for e in entities], "Label": [e[1] for e in entities]})
        else:
            st.write("No named entities found.")
    else:
        st.warning("Please enter some text to analyze.")