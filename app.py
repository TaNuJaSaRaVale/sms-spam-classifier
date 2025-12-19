import streamlit as st
import pickle
import string
import nltk

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

download_nltk_data()



from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------- Page Config ----------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
body {
    background-color: #0f1117;
}
.main {
    background-color: #0f1117;
}
.card {
    background: #161b22;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 0 25px rgba(99,102,241,0.15);
}
.result-spam {
    color: #ff4b4b;
    font-size: 28px;
    font-weight: bold;
}
.result-ham {
    color: #4ade80;
    font-size: 28px;
    font-weight: bold;
}
.small-text {
    color: #9ca3af;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- NLP Setup ----------
ps = PorterStemmer()

@st.cache_resource
def load_models():
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model

tfidf, model = load_models()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = [i for i in tokens if i.isalnum()]
    y = [i for i in y if i not in stopwords.words("english")]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# ---------- UI ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("## SMS Spam Classifier")
st.markdown(
    "<p class='small-text'>Paste a message below and let the ML model decide whether it’s spam or not.</p>",
    unsafe_allow_html=True
)

input_sms = st.text_area(
    "Message",
    height=150,
    placeholder="Example: Congratulations! You've won a free gift card..."
)

predict_btn = st.button("🔍 Predict", use_container_width=True)

# ---------- Prediction ----------
if predict_btn:
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        prediction = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input).max() * 100

        st.markdown("---")

        if prediction == 1:
            st.markdown(
                "<div class='result-spam'>🚨 Spam Message</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-ham'>✅ Not Spam</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            f"<p class='small-text'>Confidence: {probability:.2f}%</p>",
            unsafe_allow_html=True
        )

        with st.expander("🧠 See Preprocessed Text"):
            st.code(transformed_sms)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown(
    "<p class='small-text' style='text-align:center;margin-top:20px;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
