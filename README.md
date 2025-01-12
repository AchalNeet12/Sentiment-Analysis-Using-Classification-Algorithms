# Sentiment Analysis Using Classification Algorithms
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
   -  Cross-validated Accuracy: 0.766
   -  Test Accuracy: 0.745
   -  Confusion Matrix:
      - True Negatives (TN): 78
      - False Positives (FP): 19
      - False Negatives (FN): 32
      - True Positives (TP): 71

     
                      [[78, 19]
     
                      [32, 71]]
   -  Precision:
      - Class 0: 0.71
      - Class 1: 0.79
   -  Recall:
      - Class 0: 0.80
      - Class 1: 0.69
   -  F1-Score:
      - Class 0: 0.75
      - Class 1: 0.74
        
**Summary:** Logistic Regression performs well, with high recall for Class 0 and balanced precision and recall for both classes. This makes it a reliable model for this classification task.

- **K-Nearest Neighbors (KNN):** A non-parametric method used for classification based on distance to the nearest neighbors.
   -  Cross-validated Accuracy: 0.73875
   -  Test Accuracy: 0.72
   -  Confusion Matrix:
      - True Negatives (TN): 73
      - False Positives (FP): 24
      - False Negatives (FN): 32
      - True Positives (TP): 71

                     [[73, 24]
      
                     [32, 71]]
   -  Precision:
      - Class 0: 0.70
      - Class 1: 0.75
   -  Recall:
      - Class 0: 0.75
      - Class 1: 0.69
   -  F1-Score:
      - Class 0: 0.72
      - Class 1: 0.72
        
**Summary:** KNN provides balanced performance but falls behind other models like Logistic Regression and Naive Bayes.

- **Random Forest:** An ensemble learning method using multiple decision trees.
   -  Cross-validated Accuracy: 0.735
   -  Test Accuracy: 0.69
   -  Confusion Matrix:
       - True Negatives (TN): 91
       - False Positives (FP): 6
       - False Negatives (FN): 56
       - True Positives (TP): 47

                     [[91, 6]
                     [56, 47]]
   -  Precision:
      - Class 0: 0.62
      - Class 1: 0.89
   -  Recall:
      - Class 0: 0.94
      - Class 1: 0.46
   -  F1-Score:
      - Class 0: 0.75
      - Class 1: 0.60
        
**Summary:** Random Forest has high recall for Class 0, but very low recall for Class 1, leading to a lower overall performance on the test set. It is not the best choice here.

- **Decision Tree:** A model that splits data based on features to predict the outcome.
    -  Cross-validated Accuracy: 0.73
    -  Test Accuracy: 0.665
    -  Confusion Matrix:
       - True Negatives (TN): 73
       - False Positives (FP): 24
       - False Negatives (FN): 43
       - True Positives (TP): 60

                     [[73, 24]
                     [46, 60]]
   -  Precision:
      - Class 0: 0.63
      - Class 1: 0.71
   - Recall:
     - Class 0: 0.75
     - Class 1: 0.58
   - F1-Score:
     - Class 0: 0.69
     - Class 1: 0.64
       
**Summary:** The Decision Tree model shows moderate performance but struggles with recall for Class 1, which makes it less effective compared to other models.

- **Support Vector Machine (SVM):** A classification algorithm that finds a hyperplane separating the classes with the largest margin.
   -  Cross-validated Accuracy: 0.776
   -  Test Accuracy: 0.725
   -  Confusion Matrix:
      - True Negatives (TN): 78
      - False Positives (FP): 19
      - False Negatives (FN): 36
      - True Positives (TP): 67

                     [[78, 19]
                     [36, 67]]
   -  Precision:
      - Class 0: 0.68
      - Class 1: 0.78
   -  Recall:
      - Class 0: 0.80
      - Class 1: 0.65
   -  F1-Score:
      - Class 0: 0.74
      - Class 1: 0.71
        
**Summary:** While SVM has a high cross-validation score, its test set accuracy is slightly lower than Logistic Regression. Precision for Class 1 is high, but recall for Class 1 could be improved.

- **XGBoost:** A gradient boosting algorithm that builds trees in a sequential manner to improve predictive accuracy.
   -  Cross-validated Accuracy: 0.710
   -  Test Accuracy: 0.665
   -  Confusion Matrix:
      - True Negatives (TN): 80
      - False Positives (FP): 17
      - False Negatives (FN): 50
      - True Positives (TP): 53

    
                     [[80, 17]
                     [50, 53]]
   -  Precision:
      - Class 0: 0.62
      - Class 1: 0.76
   -  Recall:
      - Class 0: 0.82
      - Class 1: 0.51
   -  F1-Score:
      - Class 0: 0.70
      - Class 1: 0.61
        
**Summary:** XGBoost has lower performance in terms of both cross-validation and test accuracy. While recall for Class 0 is high, recall for Class 1 is significantly lower, which negatively impacts its overall performance.

- **LightGBM:** A gradient boosting method that is optimized for large datasets.
   -  Cross-validated Accuracy: 0.60125
   -  Test Accuracy: 0.605
   -  Confusion Matrix:
      - True Negatives (TN): 79
      - False Positives (FP): 18
      - False Negatives (FN): 61
      - True Positives (TP): 42

                     [[79, 18]
                     [61, 42]]
   -  Precision:
      - Class 0: 0.56
      - Class 1: 0.70
   -  Recall:
      - Class 0: 0.81
      - Class 1: 0.41
   -  F1-Score:
      - Class 0: 0.67
      - Class 1: 0.52
        
**Summary:** LightGBM underperforms compared to other models, particularly with low recall for Class 1. It is not recommended for this task.

- **Naive Bayes:** A probabilistic classifier based on Bayes’ theorem, commonly used for text classification.
   -  Cross-validated Accuracy: 0.7625
   -  Test Accuracy: 0.76
   -  Confusion Matrix:
      - True Negatives (TN): 72
      - False Positives (FP): 25
      - False Negatives (FN): 23
      - True Positives (TP): 80
        
                    [[72, 25]
                     [23, 80]]
   -  Precision:
      - Class 0: 0.76
      - Class 1: 0.76
   -  Recall:
      - Class 0: 0.74
      - Class 1: 0.78
   - F1-Score:
     - Class 0: 0.75
     - Class 1: 0.77
       
 - **Summary:** Naive Bayes provides a strong balance between precision and recall for both classes, making it another good candidate, 
   with performance similar to Logistic Regression.    
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
