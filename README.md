# Am I the Asshole Detector

This Streamlit application uses a Naive Bayes classifier to predict whether a scenario is considered "Am I the Asshole" (YTA) or not. The model is trained on a dataset of scenarios labeled as YTA or Not the Asshole (NTA).

## Dataset Information

The dataset used for training the model contains scenarios from the "r/AmITheAsshole" subreddit. Each scenario is labeled as one of the following categories:

- NTA: Not the Asshole
- YTA: You're the Asshole
- ESH: Everyone Sucks Here
- NAH: No Assholes Here
- INFO: Not Enough Information

## Model Training

The dataset is preprocessed to extract features from the text using TF-IDF vectorization. The class distribution is balanced using Synthetic Minority Over-sampling Technique (SMOTE). A Multinomial Naive Bayes classifier is trained on the balanced dataset.

## Usage

1. Run the Streamlit application using the command `streamlit run Final.py`.
2. Enter a scenario in the text area.
3. Click the "Predict" button to see the prediction.


## Dependencies

- pandas
- numpy
- scikit-learn
- tqdm
- streamlit
