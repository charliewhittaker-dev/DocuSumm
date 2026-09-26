import os
import re
import ssl
import pandas as pd
import streamlit as st
import spacy
import PyPDF2
import nltk
from collections import Counter
from textblob import TextBlob
from deep_translator import MyMemoryTranslator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nrclex import NRCLex
import time

# --- ENVIRONMENT CONFIGURATION ---
# Route NLTK downloads to a local directory to bypass strict network or IT group policies
local_nltk_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(local_nltk_dir, exist_ok=True)
nltk.data.path.append(local_nltk_dir)

# Bypass SSL certificate verification for local downloading
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Silently download required background dictionaries to the local folder
nltk.download('punkt', download_dir=local_nltk_dir, quiet=True)
nltk.download('punkt_tab', download_dir=local_nltk_dir, quiet=True)
nltk.download('wordnet', download_dir=local_nltk_dir, quiet=True)

# --- MODEL INITIALISATION ---
# Cache the model load so Streamlit doesn't reload the heavy NLP data on every button press
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_md")

nlp = load_model()

# --- MAIN USER INTERFACE ---
st.title("DocuSumm: Advanced NLP Tool")
st.write("Upload a document or enter text manually to run a multi-layered linguistic analysis.")

# Input handling: Supports file uploads (.txt, .pdf) or direct text entry
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
user_input = ""

if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        user_input = uploaded_file.getvalue().decode("utf-8")
        
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        # Iterate through all PDF pages and concatenate the text
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                user_input += extracted_text + "\n"
                
        # Error handling for image-based PDFs that return blank strings
        if not user_input.strip():
            st.error("Error: Could not extract text. This PDF appears to be a scanned image rather than a text document.")
else:
    user_input = st.text_area("Or input text manually", height=150)

# --- NLP PROCESSING PIPELINE ---
# Only execute the processing logic if valid text exists
if user_input.strip():
    
    # Structure the dashboard into clean, logical sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Core Analysis", "Summariser", "Translator", "Emotion", "CV Parser"
    ])

    # --- TAB 1: Sentiment & Entity Extraction ---
    with tab1:
        # Calculate baseline sentiment polarity (-1.0 to 1.0)
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity
        
        st.subheader("Sentiment Analysis")
        if sentiment > 0.1:
            st.success(f"Positive (Score: {sentiment:.2f})")
        elif sentiment < -0.1:
            st.error(f"Negative (Score: {sentiment:.2f})")
        else:
            st.info(f"Neutral (Score: {sentiment:.2f})")

        # Extract Named Entities using the spaCy medium model
        st.subheader("Named Entities")
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        if entities:
            # Display extracted data in a structured table
            st.table({"Entity": [e[0] for e in entities], "Label": [e[1] for e in entities]})
            
            # Visualise the frequency of entity categories
            st.subheader("Entity Frequency")
            label_counts = Counter([e[1] for e in entities])
            st.bar_chart(label_counts)

            # Export functionality for further data handling
            df = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.download_button(
                label="Download Entities as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="extracted_entities.csv",
                mime="text/csv"
            )
        else:
            st.write("No named entities found.")

    # --- TAB 2: Extractive Summarisation ---
    with tab2:
        st.subheader("Extractive Summarisation")
        num_sentences = st.slider("Select number of sentences for summary", 1, 10, 3)
        
        if st.button("Generate Summary"):
            # Process the text and rank sentences by importance using LexRank
            parser = PlaintextParser.from_string(user_input, Tokenizer("english"))
            summariser = LexRankSummarizer()
            summary = summariser(parser.document, num_sentences)
            
            for sentence in summary:
                st.write(f"- {sentence}")

    # --- TAB 3: Multi-Language Translation ---
    with tab3:
        st.subheader("Translation")
        
        # Map front-end display names to strict ISO language codes required by the API
        language_mapping = {
            "Spanish": "es-ES",
            "French": "fr-FR",
            "German": "de-DE",
            "Chinese (Simplified)": "zh-CN"
        }
        
        selected_lang = st.selectbox("Select Target Language", list(language_mapping.keys()))
        target_code = language_mapping[selected_lang]
        
        if st.button("Translate"):
            try:
                translator = MyMemoryTranslator(source='en-GB', target=target_code) 
                
                # Split text into chunks of 499 characters to bypass API length limits
                chunks = [user_input[i:i+499] for i in range(0, len(user_input), 499)]
                translated_text = ""
                
                # Visual indicator for large files
                st.info(f"Translating {len(chunks)} sections. This may take a moment...")
                progress_bar = st.progress(0)
                
                # Process each chunk sequentially
                for i, chunk in enumerate(chunks):
                    translated_text += translator.translate(chunk) + " "
                    # Update the progress bar
                    progress_bar.progress((i + 1) / len(chunks))
                    # Pause for 0.5 seconds to prevent rate-limit IP bans
                    time.sleep(0.5) 
                
                st.write(translated_text.strip())
                st.success("Full document translated successfully.")
                
            except Exception as e:
                st.error(f"Translation failed: {e}")

    # --- TAB 4: Custom Emotion Lexicon ---
    with tab4:
        st.subheader("Emotion Detection")
        if st.button("Analyse Emotion"):
            
            # Hardcoded psychological lexicon to bypass unstable third-party dictionary dependencies
            emotion_lexicon = {
                "Joy": ["happy", "delighted", "thrilled", "fantastic", "confident", "good", "great", "positive", "glad", "excellent", "proud", "cheerful", "satisfied", "success", "amazing"],
                "Anger": ["furious", "mad", "angry", "disaster", "terrible", "bad", "hate", "frustrated", "annoyed", "irritated", "rage", "hostile", "bitter", "awful"],
                "Fear": ["anxious", "terrified", "scared", "afraid", "overwhelmed", "panic", "nervous", "worried", "dread", "frightened", "threat", "danger", "stressed"],
                "Anticipation": ["upcoming", "proceed", "ready", "wait", "soon", "expect", "eager", "hope", "planning", "anticipate", "prepare"],
                "Sadness": ["sad", "depressed", "sorry", "upset", "crying", "lack", "grief", "heartbreak", "miserable", "sorrow", "loss", "disappointed", "gloomy"],
                "Trust": ["rely", "depend", "trust", "support", "secure", "safe", "honest", "truth", "reliable", "proven", "authentic"],
                "Surprise": ["shocked", "amazed", "astonished", "surprised", "sudden", "unexpected", "startled", "unbelievable"],
                "Disgust": ["disgusting", "gross", "vile", "repulsive", "sick", "nasty", "revolting", "appalling", "unacceptable"]
            }
            
            # Normalise text and strip punctuation to isolate pure alphabetical words
            words = re.findall(r'\b[a-z]+\b', user_input.lower())
            
            # Dynamically initialise the counter to 0 for all keys in the lexicon
            emotion_counts = {emotion: 0 for emotion in emotion_lexicon.keys()}
            emotions_found = False
            
            # Map the document vocabulary against the lexicon arrays
            for word in words:
                for emotion, keywords in emotion_lexicon.items():
                    if word in keywords:
                        emotion_counts[emotion] += 1
                        emotions_found = True
                        
            if emotions_found:
                # Filter out zero-value metrics to ensure the bar chart remains readable
                active_emotions = {k: v for k, v in emotion_counts.items() if v > 0}
                emo_df = pd.DataFrame(list(active_emotions.items()), columns=["Emotion", "Count"])
                st.bar_chart(emo_df.set_index("Emotion"))
            else:
                st.warning("No strong specific emotions detected in this text.")
                
    # --- TAB 5: Unstructured Data Parsing ---
    with tab5:
        st.subheader("Automated Data Extraction (CVs)")
        if st.button("Extract Data"):
            
            # Standardised Regex patterns for capturing contact information
            emails = set(re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', user_input))
            phones = set(re.findall(r'\(?\b[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', user_input))
            
            # Pre-defined array of industry-specific technical skills
            tech_skills = ["python", "java", "sql", "git", "machine learning", "excel", "c++", "react", "html", "javascript", "nlp"]
            found_skills = set([skill for skill in tech_skills if skill in user_input.lower()])

            # Render the extracted data into clean vertical columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Emails Found:**")
                for e in emails: st.write(e)
            with col2:
                st.markdown("**Phones Found:**")
                for p in phones: st.write(p)
            with col3:
                st.markdown("**Tech Skills:**")
                for s in found_skills: st.write(s.title())