# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("cleaned_post_comment.csv")
df=df[:15000]
df["label"] = ""


st.write("Have a look at the data:")
st.write(df.head())

# +
from tqdm import tqdm

def Search_Label_AITA():
    for i in tqdm(range(len(df['comment']))):
        if "NTA" in df['comment'][i]:
            df["label"][i] = "NTA"
        elif "YTA" in df['comment'][i]:
            df["label"][i] = "YTA"
        elif "YWBTA" in df['comment'][i]:
            df["label"][i] = "YWBTA"
        elif "YWNBTA" in df['comment'][i]:
            df["label"][i] = "YWNBTA"
        elif "ESH" in df['comment'][i]:
            df["label"][i] = "ESH"
        elif "NAH" in df['comment'][i]:
            df["label"][i] = "NAH"
        elif "INFO" in df['comment'][i]:
            df["label"][i] = "INFO"


# -

Search_Label_AITA()

df = df.drop(df[df['label'] == ''].index)

df.drop('Unnamed: 0', axis=1, inplace=True)


# +
class_distribution = df['label'].value_counts(normalize=True)

st.write(print("Class Distribution:"))
st.write(print(class_distribution))
# -


# # Convert text data to a sparse matrix
# vectorizer = TfidfVectorizer()
# sparse_matrix = vectorizer.fit_transform(df['title_body'])

# dense_matrix = sparse_matrix.toarray()

# # +
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score

# X = dense_matrix
# y = df['label']
# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # -


# # +
# # Create a Gaussian Naive Bayes classifier
# clf = GaussianNB()


# # Train the classifier
# clf.fit(X_train, y_train)

# # +
# # Make predictions on the test set
# y_pred = clf.predict(X_test)

# # Calculate the accuracy of the classifier
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
# # -

#Make predictions on unseen data
temp = []
new_df = df[:1000]
for index, row in new_df.iterrows():
    temp.append((row['title_body'], row['label']))

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample dataset (replace this with your dataset)
data = temp

# Preprocess data
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Create feature set
all_words = FreqDist()
for text, _ in data:
    for word in preprocess_text(text):
        all_words[word] += 1

word_features = list(all_words.keys())

def extract_features(text):
    words = set(preprocess_text(text))
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

# Split data into training and test sets
featuresets = [(extract_features(text), label) for text, label in data]
train_set, test_set = train_test_split(featuresets, test_size=0.3, random_state=42)

# Train the Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
y_true = [label for _, label in test_set]
y_pred = [classifier.classify(features) for features, _ in test_set]
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Get input from user
user_input = input("Enter your scenario: ")

# Preprocess user input
user_features = extract_features(user_input)

# Use the trained classifier to predict the label
prediction = classifier.classify(user_features)

# Display the prediction
print(f"Based on your scenario, you are considered: {prediction}")



































































