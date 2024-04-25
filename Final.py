import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import pickle
import streamlit as st

neutral_ds = pd.read_csv("Datasets/Neutral_dataset.csv")

st.markdown("### Have a look at the data:")
st.write(neutral_ds.head())

label_meanings = {
    "NTA": "Not the Asshole",
    "YTA": "You're the Asshole",
    "ESH": "Everyone Sucks Here",
    "NAH": "No Assholes Here",
    "INFO": "Not Enough Information"
}

# Display the dictionary using Streamlit
st.write("### Labels and Their Meanings")
for label, meaning in label_meanings.items():
    st.write(f"- **{label}**: {meaning}")

# +
from tqdm import tqdm

def Search_Label_AITA(ds):
    for i in tqdm(range(len(ds['completion']))):
        if "NTA" in ds['completion'][i]:
            ds["label"][i] = "NTA"
        elif "YTA" in ds['completion'][i]:
            ds["label"][i] = "YTA"
        elif "ESH" in ds['completion'][i]:
            ds["label"][i] = "ESH"
        elif "NAH" in ds['completion'][i]:
            ds["label"][i] = "NAH"
        elif "INFO" in ds['completion'][i]:
            ds["label"][i] = "INFO"


neutral_ds["label"] = ""
Search_Label_AITA(neutral_ds)


class_distribution = neutral_ds['label'].value_counts(normalize=True)

st.write("### Class Distribution before balancing:")
st.write(class_distribution)


# +
majority_class = neutral_ds[neutral_ds['label'] == "NTA"]
minority_class = neutral_ds[neutral_ds['label'] == "YTA"]

# Undersample majority class
undersampled_majority = majority_class.sample(len(minority_class))

# Combine minority class with undersampled majority class
balanced_df = pd.concat([undersampled_majority, minority_class])

# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
# -

balanced_df['label'].value_counts()

class_distribution = balanced_df['label'].value_counts(normalize=True)

st.write("### Class Distribution after Balancing (Under Sampling):")
st.write(class_distribution)



# +
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(balanced_df['prompt'])
y = balanced_df['label']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model

# with open('Model.pkl', 'wb') as f:
#     pickle.dump(model, f)


# with open('Model.pkl', 'rb') as f:
#     model = pickle.load(f)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
accuracy_neu = accuracy_score(y_test, y_pred) * 100
print(f'Accuracy for Neutral Dataset: {accuracy_neu:.2f}')
st.markdown(f'#### Accuracy for Neutral Dataset: {accuracy_neu:.2f}%')

# 
# Example prediction for a new post


st.title("Am I the Asshole Detector")
user_input = st.text_area("Enter your scenario:")
new_post = []
new_post.append(user_input)
new_post_vectorized = vectorizer.transform(new_post)

if st.button("Predict"):
    prediction = model.predict(new_post_vectorized)
    st.write(f"Based on your scenario, you are considered: {prediction}")