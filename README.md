## AWS Sentiment Streamlit App

Check out the [AWS Sentiment Streamlit App](https://awsentiment.streamlit.app/)

# AWS Sentiment Analysis with Amazon Fashion Customer Reviews

This repository contains a project for sentiment analysis on Amazon Fashion customer reviews using various machine learning models. The project is implemented using Python and includes a Streamlit web application for model deployment and testing.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Streamlit Application](#streamlit-application)
- [Usage](#usage)


## Project Overview

The main goal of this project is to analyze customer reviews from the Amazon Fashion dataset to determine sentiment. The project involves several steps, including data collection, data cleaning, feature engineering, model training, and deployment of the model using a Streamlit application.

### Agenda
1. **MongoDB**
2. **Data Cleaning & Processing**
3. **Vectorizer**
4. **Topic Modeling**
5. **Training Data**
6. **Model Evaluation & Selection**
7. **Streamlit App**

## Data Pipeline

### MongoDB
- The dataset is created by merging the `raw_review_Amazon_Fashion` data with the `raw_meta_Amazon_Fashion` data from AWS Public Datasets.
- Reviews data can be fetched from MongoDB, while meta data is processed directly from the URL due to size limitations.

### Data Cleaning & Processing
- Select columns to be included in the model.
- Remove rows with null values.
- Classify users based on their ratings into three groups:
  - Ratings 4 and 5: Positive (1)
  - Ratings 3: Neutral (2)
  - Ratings 1 and 2: Negative (0)

### Review Processing Steps
1. **Stop Words Removal**: Convert all text to lowercase and remove stop words.
2. **Regex Cleaning**: Remove all numbers and non-alphabet characters.
3. **Correct Spelling**: Fix spelling errors.
4. **POS Tagging**: Tag parts of speech for lemmatization.
5. **Lemmatization**: Reduce words to their base forms.
6. **Get Sentiment**: Use the TextBlob library to include polarity and subjectivity scores in the model.

### Handling Data Imbalance
- Oversampling methods such as SMOTE can be used to handle data imbalance during model training.

### Vectorizer
- **TFIDF Tokenizer**: Applied with the following parameters:
  - `stop_words='english'`: Automatically remove common English stop words (e.g., "and", "the", "is").
  - `min_df=0.008`: A term must appear in at least 0.8% of the documents to be considered.
  - `ngram_range=(1,3)`: Create unigrams (single words), bigrams (two-word combinations), and trigrams (three-word combinations).
  - `token_pattern="\\b[a-z][a-z][a-z]+\\b"`: Select words that are at least three letters long. This regex pattern captures words with three or more lowercase letters.

### Feature Engineering
- **Topic Modeling**: Product titles are classified into three different topics using topic modeling.
- **Review Length**: The length of the review is included as a feature.
- Create dummy variables for string columns.

## Model Training
- Use GridSearchCV for model training.
- Evaluate different models including XGBClassifier, Random Forest, and Naive Bayes.

## Model Evaluation

### Results
- The best model is selected based on accuracy and f1-score.
- Model comparison:
  | Model                | Accuracy | f1  |
  |----------------------|----------|-----|
  | XGBClassifier        | 0.80     | 0.50|
  | XGBClassifier_SMOTE  | 0.91     | 0.53|
  | Random Forest        | 0.79     | 0.44|
  | Naive Bayes          | 0.26     | 0.19|

## Streamlit Application

The model is deployed using a Streamlit web application, which can be accessed [here](https://awsentiment.streamlit.app/).

## Usage

To run the Streamlit app locally:
```sh
streamlit run app3.py
