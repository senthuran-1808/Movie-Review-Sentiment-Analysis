import streamlit as st
import joblib
import re
import string
import nltk

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Load saved model and TF-IDF vectorizer
model = joblib.load("models\sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Text preprocessing function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove emojis / non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Join tokens
    cleaned_text = " ".join(tokens)

    return cleaned_text


# Streamlit page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# Title
st.title("🎬 Movie Review Sentiment Analysis")


# Sidebar
st.sidebar.header("About Project")

st.sidebar.write("""
This NLP project uses:

- NLTK
- TF-IDF Vectorization
- Machine Learning
- Streamlit

Models Used:
- Logistic Regression
- Naive Bayes
- Random Forest
""")


# User input
user_review = st.text_area(
    "Enter Movie Review",
    height=100
)
# Custom CSS for green button
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #90EE90;
    color: black;
    font-size: 18px;
    height: 3em;
    width: 100%;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

div.stButton > button:first-child:hover {
    background-color: #77dd77;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# Predict button
if st.button("Predict Sentiment"):

    if user_review.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        # Loading spinner
        with st.spinner("Predicting sentiment..."):

            # Clean review
            cleaned_review = clean_text(user_review)

            # TF-IDF transformation
            vector_input = tfidf.transform(
                [cleaned_review]
            )

            # Prediction
            prediction = model.predict(
                vector_input
            )[0]

            # Probability score
            probability = model.predict_proba(
                vector_input
            )

            confidence = round(
                probability.max() * 100,
                2
            )

        # Final result
        st.subheader("Prediction Result")

        if prediction == "positive":

            st.success(
                f"😊 Positive Review\n\nConfidence Score: {confidence}%"
            )

        else:

            st.error(
                f"😠 Negative Review\n\nConfidence Score: {confidence}%"
            )

        # Extra analysis
        st.subheader("Review Analysis")

        st.write(
            f"**Character Count:** {len(user_review)}"
        )

        st.write(
            f"**Word Count:** {len(user_review.split())}"
        )

        st.write(
            f"**Cleaned Review:** {cleaned_review}"
        )
