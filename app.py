import numpy as np
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')
nltk.download('wordnet')

# Importing the dataset
dataset = pd.read_csv(r"Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the texts with Lemmatization and custom stopwords
lemmatizer = WordNetLemmatizer()
corpus = [] 
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the TF-IDF model
cv = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Dictionary of classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=0),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=0),
    'Decision Tree': DecisionTreeClassifier(random_state=0),
    'SVM': SVC(kernel='linear', random_state=0),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'Naive Bayes': MultinomialNB()
}

# Hyperparameter tuning and cross-validation
for name, model in classifiers.items():
    # GridSearchCV for hyperparameter tuning
    if name == 'Logistic Regression':
        param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear']}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        print(f"{name} Best Params: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    elif name == 'K-Nearest Neighbors':
        param_grid = {'n_neighbors': [3, 5, 7, 9]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        print(f"{name} Best Params: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)

    # Cross-validation score
    accuracies = cross_val_score(model, X_train, y_train, cv=5)
    print(f'{name} Cross-validated accuracy: {accuracies.mean()}')

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    print(f'{name} Confusion Matrix:\n', cm)
    print(f'{name} Accuracy: {ac}')
    print(classification_report(y_test, y_pred))
    
    # Save the model
    with open(f'{name.replace(" ", "_").lower()}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Create an ensemble model (Voting Classifier)
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(random_state=0)),
    ('rf', RandomForestClassifier(random_state=0)),
    ('knn', KNeighborsClassifier())
], voting='hard')

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
cm_voting = confusion_matrix(y_test, y_pred_voting)
ac_voting = accuracy_score(y_test, y_pred_voting)
print(f'Voting Classifier Confusion Matrix:\n', cm_voting)
print(f'Voting Classifier Accuracy: {ac_voting}')
print(classification_report(y_test, y_pred_voting))

# Save the ensemble model
with open('voting_classifier_model.pkl', 'wb') as file:
    pickle.dump(voting_clf, file)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(cv, file)
