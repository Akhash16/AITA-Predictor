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

print("Class Distribution:")
print(class_distribution)
# -


# Convert text data to a sparse matrix
vectorizer = TfidfVectorizer()
sparse_matrix = vectorizer.fit_transform(df['title_body'])

dense_matrix = sparse_matrix.toarray()

# +
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X = dense_matrix
y = df['label']
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# -


# +
# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()


# Train the classifier
clf.fit(X_train, y_train)

# +
# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
# -

#Make predictions on unseen data
user_input_title = st.text_area("Enter the title of the post")
user_input_comment = st.text_area("Enter the comment")




































































