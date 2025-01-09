import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import base64

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .stMarkdownContainer {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set up background
image_base64 = get_base64_image("img.jpg")  # Update the path to your background image
set_background(image_base64)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load all saved models and the vectorizer
models = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'K-Nearest Neighbors': 'k-nearest_neighbors_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'SVM': 'svm_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
    'LightGBM': 'lightgbm_model.pkl',
    'Naive Bayes': 'naive_bayes_model.pkl',
    'Voting Classifier': 'voting_classifier_model.pkl'
}

# Load each model into the dictionary
for model_name in models.keys():
    with open(models[model_name], 'rb') as file:
        models[model_name] = pickle.load(file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess the input text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

# Streamlit UI
st.title('Sentiment Analysis')

# Sidebar for model selection
st.sidebar.header("🔍 Model Selection")
st.sidebar.markdown(
    """
    ** 📝Instructions:**
    1. Enter a restaurant review in the main section.
    2. Select the model you want to use for sentiment analysis from the sidebar.
    3. Click on the "Predict Sentiment" button to get the sentiment of the review.
    4. The result will show if the sentiment is **Positive** or **Negative**, along with the model used.
    """
)

# Model selection in the sidebar
model_option = st.sidebar.selectbox(
    " 📊Choose a model:",
    list(models.keys())
)

# Main panel
st.markdown("Enter a restaurant review below:")

# Textbox for user input
review_input = st.text_area("Review:", height=100)

# Button to predict sentiment
if st.button("📈Predict Sentiment"):
    if review_input:
        # Preprocess the input text
        processed_review = preprocess_text(review_input)

        # Transform the input text using the vectorizer
        review_vector = vectorizer.transform([processed_review]).toarray()

        # Model prediction based on selected model
        model = models[model_option]
        sentiment = model.predict(review_vector)

        # Display the result
        if sentiment == 1:
            st.write(f"The sentiment of the review is: **Positive** 😊   using {model_option}")
        else:
            st.write(f"The sentiment of the review is: **Negative** 😞   using {model_option}")
    else:
        st.warning("Please enter a review to analyze!")
