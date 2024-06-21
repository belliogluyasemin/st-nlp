import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import recall_score, log_loss, f1_score, accuracy_score, precision_recall_fscore_support
import shap
shap.initjs()
from joblib import load
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import xgboost as xgb

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

st.set_page_config(
    page_title="AWS Review Sentiment",
    page_icon= "sentiment-analysis-icon-vector.jpg",
)

# Sidebar inputs
st.sidebar.markdown("**Enter User Review & Other Information**")
input_data = {
    'text_review': [st.sidebar.text_area("User Review:", value="happy sock money waste buy wear tie stay foot well")],
    'title_product': [st.sidebar.text_area("Product Title:", value="month pair woman truly show socks black black size")],
    'rating_number': [st.sidebar.number_input("Rating Count:", value=58, step=1)],
}
input_df = pd.DataFrame(input_data)

st.markdown("""
    <h1 style='color: #876C6C;'>User Sentiment Analysis For AWS Fashion Customer</h1>
    """, unsafe_allow_html=True)
st.markdown("Click The Submit Button for Model Result's")

st.markdown("""
    <h2 style='color: #1b8bb4;'>About The Models</h2>
    <p>This sentiment analysis is performed on customer reviews from the AWS online store. The best results are achieved using the XGBoost algorithm. Additionally, the Google BERT model, developed by Google, is utilized to analyze user reviews on a scale of 1 to 5. The results from both models are displayed below.</p>
    <h3 style='color: #1b8bb4;'>XGBoost</h3>
    <p>XGBoost is an optimized version of the Gradient Boosting algorithm. As a tree-based machine learning algorithm, it builds predictive models sequentially. Each new model is created to correct the errors of the previous model. This approach enhances the overall performance of the model and achieves high accuracy.</p>
    <h3 style='color: #1b8bb4;'>Google BERT</h3>
    <p>BERT (Bidirectional Encoder Representations from Transformers) is a language model developed by Google. Based on the Transformer architecture, BERT captures the meaning of text by using bidirectional context. This allows the model to consider both previous and next words for more accurate predictions. BERT excels in various natural language processing tasks such as sentiment analysis, question answering, and text summarization, delivering high performance.Google BERT model scales sentiments from 1 to 5.</p>
    """, unsafe_allow_html=True)


if st.sidebar.button("Submit"):

    # Function List
    def predict_with_bert(text):
        tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        tokens = tokenizer.encode(text, return_tensors='pt')
        result = model(tokens)
        sentiment = int(torch.argmax(result.logits))+1
        return sentiment

    lower = lambda x: str(x.lower())

    def stop_words(text):
        text = [word for word in word_tokenize(text) if word not in stopwords.words('english')]
        text = ' '.join(text)
        return text 

    def regex(text):
        text = [re.sub(r'[^a-zA-Z\s]', '', word) for word in word_tokenize(text)]
        text = [word for word in text if word]
        text = ' '.join(text)
        return text

    def correct_spelling(text):
        text_blob = TextBlob(text)
        corrected_text = str(text_blob.correct())
        return corrected_text

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        pos_tagged = pos_tag(word_tokenize(text))
        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tagged]
        return ' '.join(lemmatized)

    def get_sentiment(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        return pd.Series([polarity, subjectivity])

    # Load dataset & AI Model
    all_smaller = pd.read_pickle("all_smaller.pkl")
    xgb_model = load("xgb_model.pkl")

    # Preprocess new input data
    input_df['text_review'] = input_df['text_review'].map(lower).map(correct_spelling).map(regex).map(stop_words).map(lemmatize)
    input_df['title_product'] = input_df['title_product'].map(lower).map(correct_spelling).map(regex).map(stop_words).map(lemmatize)
    input_df[['polarity', 'subjectivity']] = input_df['text_review'].apply(get_sentiment)
    input_df['polarity'] = input_df['polarity'] 
    input_df['subjectivity'] = input_df['subjectivity'] 

    # Vectorize new input data using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', min_df = 0.001, ngram_range=(1,3), token_pattern="\\b[a-z][a-z][a-z]+\\b")
    vectorizer.fit(all_smaller['text_review'])
    vectorized = vectorizer.transform(input_df['text_review'])
    tfidf_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())

    input_df = pd.concat([input_df, tfidf_df], axis=1)

    vectorizer2 = CountVectorizer(ngram_range=(1, 2), min_df=0.025, max_df=0.95, token_pattern="\\b[a-z][a-z]+\\b")
    vectorizer2.fit(all_smaller['title_product'])
    doc_word = vectorizer2.transform(input_df['title_product'])
    doc_word_df = pd.DataFrame(doc_word.toarray(), columns=vectorizer2.get_feature_names_out())

    optimal_n_components = 3
    lsa = TruncatedSVD(n_components=optimal_n_components)
    doc_topic = lsa.fit_transform(doc_word)
    dominant_topic = doc_topic.argmax(axis=1)
    topic_labels = ["Topic " + str(i+1) for i in dominant_topic]
    input_df['dominant_topic'] = topic_labels
    input_df['len_rew'] = input_df['text_review'].apply(lambda x: len(str(x).split()))
    dummy = pd.get_dummies(input_df['dominant_topic'], drop_first=True, dtype=int)
    input_df = pd.concat([input_df, dummy], axis=1)
    x = input_df.drop(['text_review', 'title_product', 'dominant_topic'], axis=1)

    # Fill missing columns
    missing_cols = set(xgb_model.feature_names_in_) - set(x.columns)
    for col in missing_cols:
        x[col] = 0
    x = x[xgb_model.feature_names_in_]
    x_data = x.T

    pred_class = xgb_model.predict(x_data.T)
    xgb_class = {0: 'Negative', 1: 'Confused', 2: 'Positive'}
    xgb_prediction = xgb_class[pred_class[0]]
    bert_prediction = predict_with_bert(input_df['text_review'][0])

    # Display Predictions in separated 2 Columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="background-color:#876C6C; padding: 10px; border-radius: 10px;">
                <h2 style='text-align: center; color: #202C5E;'>Class Prediction of XGBOOST</h2>
                <h1 style='text-align: center; color: #ecab53; font-size: 48px;'>{xgb_prediction}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="background-color:#876C6C; padding: 10px; border-radius: 10px;">
                <h2 style='text-align: center; color: #202C5E;'>Class Prediction of Google BERT</h2>
                <h1 style='text-align: center; color: #ecab53; font-size: 48px;'>{bert_prediction}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.header("Feature Importance XGBoost's")
    st.markdown("""
    **XGBoost's Built-in Feature Importance:** Shows how many times each feature is used by the model or the contribution of these features to the model's performance.
    """)

    st.subheader("Most Important Features for XGBOOST Model")
    fig, ax = plt.subplots()
    xgb.plot_importance(xgb_model, ax=ax, max_num_features=20)
    plt.title("xgboost.plot_importance(xgb_model)")
    st.pyplot(fig)
     
    st.header("Feature Importance Manuel")
    st.markdown("""
    **Manual Feature Importance Calculation:** Based on the importance values calculated and recorded during model training, which are typically based on the model's split gains.
    """)
    st.subheader("Most Important 20 Features Manuel Calculation")
    f_imp = xgb_model.feature_importances_
    indices = np.argsort(f_imp)[::-1][:20]  # Most Important 20 Features
    top_features = [x.columns[i] for i in indices]
    top_importances = f_imp[indices] 
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title("Top 20 Feature Importances")
    plt.bar(range(len(top_features)), top_importances, align="center")
    plt.xticks(range(len(top_features)), top_features, rotation=90)
    plt.xlim([-1, len(top_features)])
    st.pyplot(fig)

    st.header("SHAP Local Waterfall Plot - Positive")
    st.markdown("""
    The SHAP waterfall plot below class Positive
    """)
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(x)

    # SHAP waterfall plot Positive
    fig, ax = plt.subplots(figsize=(15, 8))
    shap.plots.waterfall(shap_values[-1][:,2])
    st.pyplot(fig)

    st.header("SHAP Local Waterfall Plot - Confused")
    st.markdown("""
    The SHAP waterfall plot below class Confused
    """)
    fig, ax = plt.subplots(figsize=(15, 8))
    shap.plots.waterfall(shap_values[-1][:,1])
    st.pyplot(fig)

    st.header("SHAP Local Waterfall Plot - Negative")
    st.markdown("""
    The SHAP waterfall plot below class Negative
    """)
    fig, ax = plt.subplots(figsize=(15, 8))
    shap.plots.waterfall(shap_values[-1][:,0])
    st.pyplot(fig)

    # Word Clouds Display
    st.header("WordClouds for Reviews")
    st.markdown("The WordClouds below show the most frequent words in positive, negative, and confused reviews.")

    positive_reviews = ' '.join(all_smaller[all_smaller.sentiment == 2].text_review)
    negative_reviews = ' '.join(all_smaller[all_smaller.sentiment == 0].text_review)
    confused_reviews = ' '.join(all_smaller[all_smaller.sentiment == 1].text_review)

    fig, ax = plt.subplots(3, 1, figsize=(18, 10))

    wordcloud = WordCloud(background_color='black').generate(positive_reviews)
    ax[0].imshow(wordcloud)
    ax[0].axis('off')
    ax[0].set_title('Positive Reviews')

    wordcloud = WordCloud(background_color='black').generate(confused_reviews)
    ax[1].imshow(wordcloud)
    ax[1].axis('off')
    ax[1].set_title('Confused Reviews')

    wordcloud = WordCloud(background_color='black').generate(negative_reviews)
    ax[2].imshow(wordcloud)
    ax[2].axis('off')
    ax[2].set_title('Negative Reviews')

    st.pyplot(fig)
