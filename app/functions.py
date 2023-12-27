import pandas as pd
import re
import nltk
nltk.download('wordnet')
# nltk
from nltk.stem import WordNetLemmatizer
import pickle

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

def preprocessData(data):
    processData = []

    #Create Lemmatizer and Stemmer
    wordLemm = WordNetLemmatizer()

    #Defining regex patterns
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in data:
        #Lower Casing: Each text is converted to lowercase.
        tweet = tweet.lower()

        #Replacing URLs: Links starting with "http" or "https" or "www" are replaced with "URL".
        tweet = re.sub(urlPattern, "URL", tweet)

        #Replacing Emojis: Replace emojis by using a pre-defined dictionary containing emojis along with their meaning.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI"+emojis[emoji])

        #Replacing Usernames: Replace @Usernames with word "USER".
        tweet = re.sub(userPattern, "USER", tweet)

        #Removing Non-Alphabets: Replacing characters except Digits and Alphabets with a space.
        tweet = re.sub(alphaPattern, " ", tweet)

        #Removing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetWords = ""
        for word in tweet.split():
            if len(word) > 1:
                #Lemmatizing: Lemmatization is the process of converting a word to its base form. (e.g "Great" to "Good")
                word = wordLemm.lemmatize(word)
                tweetWords += (word + " ")
        processData.append(tweetWords)
    return processData

import os
def load_model():
    file_path = os.path.abspath("C:\\Users\\ASUS\\Desktop\\MLProjects\\TwitterSentimentAnalysisDjango\\app\\vectoriser-ngram-(1,2).pickle")
    file_path1 = os.path.abspath("C:\\Users\\ASUS\\Desktop\\MLProjects\\TwitterSentimentAnalysisDjango\\app\\Sentiment-LR.pickle")
    #Load the vectoriser
    file=open(file_path, "rb")
    vectoriser = pickle.load(file)
    file.close()
    
    #Load LR model
    file=open(file_path1, "rb")
    LRModel = pickle.load(file)
    file.close()

    return vectoriser, LRModel

def predict(vectoriser, model, text):
    #Predict the sentiment
    textdata = vectoriser.transform(preprocessData(text))
    sentiment = model.predict(textdata)

    #Make a list of text with sentiment
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

vectoriserLoaded, LRModel = load_model()