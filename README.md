# Sentiment Analysis
---
## 📝 Project Description:
 - This project focuses on building a sentiment analysis application that can predict the sentiment of restaurant reviews as either positive or negative. The user can input a restaurant 
   review, select one of several machine learning models for sentiment classification, and receive a prediction of whether the sentiment is positive or negative. The application is 
   built using Streamlit for the frontend and various machine learning algorithms for the backend.
---
## 📜 Overview:
 - The goal of this project is to create a web-based sentiment analysis tool that allows users to input restaurant reviews and get predictions on whether the reviews are positive or 
   negative. The tool is powered by several machine learning models, including popular algorithms like Logistic Regression, K-Nearest Neighbors, Random Forest, SVM, XGBoost, and others. 
   The best-performing model for sentiment classification is chosen based on accuracy metrics and confusion matrices.
---
## 📦 Dataset:
 - **Dataset Name:** Restaurant Reviews Dataset
 - **Source:** The dataset contains a collection of restaurant reviews, where each review is labeled as either positive or negative.
 - **Dataset Format:** The dataset is stored in a tab-separated value (TSV) file, where each row contains:
     - **Review:** A text field containing the restaurant review.
     - **Sentiment:** A binary target value (1 for positive, 0 for negative).
  The dataset has 1000 entries, which were used to train and test the models.
---
## 🤖 Technologies Used:
 - `Python` - The primary programming language used for both backend and machine learning tasks.
 - `Streamlit` - Used to build the interactive frontend for the sentiment analysis tool.
 - `Scikit-learn` - Machine learning library used for building, training, and evaluating models.
 - `NLTK` - Natural Language Toolkit used for text preprocessing, including tokenization, stopword removal, and lemmatization.
 - `XGBoost` - An optimized gradient boosting library for machine learning.
 - `LightGBM` - A gradient boosting framework used to improve model performance.
 - `Pickle` - Used to save the trained models and the vectorizer.
---
## ⚙ Algorithms Used:
 - **Logistic Regression:** A linear model for binary classification.
   -  Accuracy: 74.5
   -  Confusion Matrix:
     
                      [[78, 19]
     
                      [32, 71]]

 - **K-Nearest Neighbors (KNN):** A non-parametric method used for classification based on distance to the nearest neighbors.
   -  Accuracy: 72
   -  Confusion Matrix:
     
                     [[73, 24]
      
                     [32, 71]]

 - **Random Forest:** An ensemble learning method using multiple decision trees.
    -  Accuracy: 69
    -  Confusion Matrix:

                     [91, 6]
                     [56, 47]]

 - **Decision Tree:** A model that splits data based on features to predict the outcome.
    -  Accuracy: 66.5
    -  Confusion Matrix:

                     [[73, 24]
                     [46, 60]]

 - **Support Vector Machine (SVM):** A classification algorithm that finds a hyperplane separating the classes with the largest margin.
    -  Accuracy: 72.5
    -  Confusion Matrix:

                     [[78, 19]
                     [36, 67]]

 - **XGBoost:** A gradient boosting algorithm that builds trees in a sequential manner to improve predictive accuracy.
    -  Accuracy: 66.5
    -  Confusion Matrix:
    
                     [[80, 17]
                     [50, 53]]

 - **LightGBM:** A gradient boosting method that is optimized for large datasets.
    -  Accuracy: 60.5
    -  Confusion Matrix:
    
                     [[79, 18]
                     [61, 42]]

 - **Naive Bayes:** A probabilistic classifier based on Bayes’ theorem, commonly used for text classification.
    -  Accuracy: 76
    -  Confusion Matrix:
    
                     [[72, 25]
                     [23, 80]]
---
## 🔍 Data Preprocessing:
  The data preprocessing pipeline consists of several key steps to prepare the text data for machine learning models:

  - **Text Cleaning:**
      - Removal of special characters and digits using regular expressions (e.g., `re.sub('[^a-zA-Z]', ' ', text)`).
      - Conversion of all text to lowercase to maintain uniformity.
  - **Tokenization:** Splitting the text into individual words or tokens.
  - **Stopword Removal:** Removal of common but non-informative words (like "the", "is", etc.) using NLTK's list of stopwords.
  - **Lemmatization:** Reducing words to their root form (e.g., "running" becomes "run") using NLTK's `WordNetLemmatizer`.
  - **TF-IDF Vectorization:** The `TfidfVectorizer` is used to transform the text data into numerical vectors that can be fed into machine learning models. TF-IDF (Term Frequency-Inverse 
    Document Frequency) helps in identifying important words in the text while down-weighting common words.
---
## 🌐 Streamlit Integration:
The frontend of the application is built using Streamlit, which provides a simple yet powerful interface for deploying machine learning models. The application includes:

- **Model Selection Sidebar:** Users can select different models from a sidebar to perform sentiment analysis.
- **Text Input Box:** Users can input a restaurant review that will be processed and classified.
- **Sentiment Display:** The predicted sentiment (Positive/Negative) is displayed with an appropriate emoji.
- **Background Customization:** A custom background image is added to enhance the user experience.
---
## 📈 Result:
After processing the reviews and applying the trained models, the application predicts whether the review's sentiment is positive or negative. Based on the selected model, users can view the sentiment analysis result along with the model name.

For example:

 - Input Review: "The food was fantastic, and the service was great!"
 - Predicted Sentiment: Positive 😊 (using Random Forest)
---
## 🎯 Conclusion:
 - The sentiment analysis application is capable of providing accurate predictions for restaurant reviews based on the selected machine learning model.
 - The project successfully integrates multiple algorithms, providing users with flexibility in choosing the best model for the task.
 - Among the models tested, Random Forest and Logistic Regression show the highest accuracy, indicating strong performance for text classification tasks.
 - Streamlit is an excellent tool for deploying this machine learning model, providing an interactive and user-friendly interface.
---
## streamlit app()
---
