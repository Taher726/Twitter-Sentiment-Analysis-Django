# Twitter Sentiment Analysis Django App

## Overview
This Django web application performs sentiment analysis on Twitter tweets using a pre-trained machine learning model. Users can input a tweet, and the application will analyze the sentiment (positive or negative) of the text.

## Features
- **Sentiment Analysis:** The app uses a machine learning model trained on Twitter data to predict the sentiment of a given tweet.
- **Django Framework:** Built using the Django web framework for a scalable and maintainable structure.
- **User Interface:** Simple and intuitive web interface for users to input and analyze tweets.
- **Database Integration:** Tweets and their corresponding sentiments are stored in a database.

## Sentiment Analysis Model
The sentiment analysis model employed in this application is a logistic regression classifier trained on a dataset of Twitter messages. The model has been pre-trained using natural language processing (NLP) techniques and features an N-gram (bi-gram) vectorization approach to capture contextual information in the text.

### Model Files
- **Vectorizer:** The N-gram vectorizer is used to convert input text into a format suitable for the model. The vectorizer has been serialized and saved in `./app/utils/vectoriser-ngram-(1,2).pickle`.

- **Classifier Model:** The logistic regression model, responsible for sentiment prediction, has been serialized and saved in `./app/utils/Sentiment-LR.pickle`.

### Training Data
The model was trained on a labeled dataset of 1600000 Twitter messages, with positive and negative sentiments. The training data is not included in this repository due to its size, but you can find the dataset in this link: https://www.kaggle.com/code/stoicstatic/twitter-sentiment-analysis-for-beginners/data

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Taher726/Twitter-Sentiment-Analysis-Django.git
   cd Twitter-Sentiment-Analysis-Django
