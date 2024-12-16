import streamlit as st
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from PyPDF2 import PdfReader  # Import PDF reader

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Setup stopwords
reserved_stop_words = set(stopwords.words('english'))
extra_stop_words = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'using', 'sample', 'fig', 'figure', 'image', 'using'
]
all_stop_words = list(reserved_stop_words.union(extra_stop_words))

# Preprocessing function without stemming or lemmatization
def txt_preprocessing(txt):
    txt = txt.lower()  # Lowercase text
    txt = re.sub(r"<.*?>", " ", txt)  # Remove HTML tags
    txt = re.sub(r"[^a-zA-Z]", " ", txt)  # Remove special characters and digits
    txt = nltk.word_tokenize(txt)  # Tokenize text
    txt = [word for word in txt if word not in all_stop_words]  # Remove stopwords
    txt = [word for word in txt if len(word) >= 3]  # Remove words shorter than 3 characters
    return " ".join(txt)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract top keywords using TF-IDF
def extract_keywords(preprocessed_text, top_n=10):
    vectorizer = CountVectorizer(max_features=3000, ngram_range=(1, 2))
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    
    # Transform text to word count and compute TF-IDF
    word_count_vector = vectorizer.fit_transform([preprocessed_text])
    tfidf_transformer.fit(word_count_vector)
    tf_idf_vector = tfidf_transformer.transform(word_count_vector)
    
    # Extract keywords
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = sorted(
        zip(tf_idf_vector.toarray()[0], feature_names),
        key=lambda x: -x[0]
    )
    keywords = [(item[1], round(item[0], 3)) for item in sorted_items[:top_n]]
    return keywords

# Streamlit App
st.title("Keyword Extraction with Text or PDF")

# Option for user input type
input_type = st.radio("Choose input type:", ["Enter Text", "Upload PDF"])

if input_type == "Enter Text":
    user_input = st.text_area("Enter your text here:", height=200)
    
    if st.button("Extract Keywords"):
        if user_input.strip():
            try:
                # Preprocess the text
                preprocessed_text = txt_preprocessing(user_input)
                st.subheader("Preprocessed Text:")
                st.write(preprocessed_text)
                
                # Extract keywords
                keywords = extract_keywords(preprocessed_text)
                st.subheader("Top Keywords:")
                for keyword, score in keywords:
                    st.write(f"**{keyword}**: {score}")
            except Exception as e:
                st.error(f"Error occurred: {e}")
        else:
            st.error("Please enter some text to process.")

elif input_type == "Upload PDF":
    pdf_file = st.file_uploader("Upload your PDF file:", type=["pdf"])
    
    if pdf_file is not None:
        try:
            # Extract text from PDF
            raw_text = extract_text_from_pdf(pdf_file)
            st.subheader("Extracted Text from PDF:")
            st.write(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)  # Show a snippet of the PDF content
            
            # Preprocess the extracted text
            preprocessed_text = txt_preprocessing(raw_text)
            
            # Extract keywords
            keywords = extract_keywords(preprocessed_text)
            st.subheader("Top Keywords:")
            for keyword, score in keywords:
                st.write(f"**{keyword}**: {score}")
        except Exception as e:
            st.error(f"Error occurred while processing the PDF: {e}")
