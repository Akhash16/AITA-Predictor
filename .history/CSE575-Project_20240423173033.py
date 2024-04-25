import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.tree import DecisionTreeClassifier
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


label_meanings = {
    "NTA": "Not the Asshole",
    "YTA": "You're the Asshole",
    "YWBTA": "You Were Being the Asshole",
    "YWNBTA": "You Were Not Being the Asshole",
    "ESH": "Everyone Sucks Here",
    "NAH": "No Assholes Here",
    "INFO": "Not Enough Information"
}

# Display the dictionary using Streamlit
st.write("### Labels and Their Meanings")
for label, meaning in label_meanings.items():
    st.write(f"- **{label}**: {meaning}")
# -

Search_Label_AITA()

df = df.drop(df[df['label'] == ''].index)

df.drop('Unnamed: 0', axis=1, inplace=True)


# +
class_distribution = df['label'].value_counts(normalize=True)

st.write("### Class Distribution:")
st.write(class_distribution)
# -

#Make predictions on unseen data
temp = []
new_df = df[:1000]
for index, row in new_df.iterrows():
    temp.append((row['title_body'], row['label']))

import nltk
import pickle
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

# Pickle the model
with open('NBmodel.pkl', 'wb') as f:
    pickle.dump(classifier, f)

st.title("Am I the Asshole Detector")

# User input
user_input = st.text_input("Enter your scenario:")

# Preprocess user input
user_features = extract_features(user_input)

# Use the trained classifier to predict the label
if st.button("Predict"):
    prediction = classifier.classify(user_features)
    st.write(f"Based on your scenario, you are considered: {prediction}")



































































