# DocuSumm: Advanced NLP Dashboard

A robust, multi-layered Natural Language Processing dashboard built with Python and Streamlit. This application processes unstructured text and PDF documents to extract actionable linguistic data.

## Features
*   **Core Analysis:** Sentiment polarity scoring (TextBlob) and Named Entity Recognition (spaCy) with CSV export.
*   **Extractive Summarisation:** LexRank algorithms (Sumy) to condense large documents into key sentences.
*   **Machine Translation:** Automated text chunking to bypass API rate limits for seamless multi-language translation.
*   **Custom Emotion Engine:** A bespoke regex-driven lexicon mapping system built to bypass unstable third-party dependencies.
*   **Unstructured Data Parsing:** Automated regex extraction of contact details and technical skills from CVs.

## Engineering Highlights
*   **Environment Resilience:** Implemented SSL bypasses and local directory routing to ensure NLTK dictionaries download correctly on restricted corporate/educational networks.
*   **API Rate Limiting:** Built data chunking and time-delay loops to process large documents through free-tier translation endpoints without triggering IP bans.
*   **Defensive Programming:** Built input validation to safely catch and handle unreadable scanned image PDFs.

## Installation

1. Clone the repository:
`git clone https://github.com/yourusername/docusumm.git`

2. Install dependencies:
`pip install -r requirements.txt`

3. Download the spaCy language model:
`python -m spacy download en_core_web_md`

4. Run the application:
`streamlit run app.py`