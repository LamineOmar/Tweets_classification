import re
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib


nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load("tweets_model.joblib")

lemma = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def contains_arabic(text):
    arabic_pattern = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\uFB50-\uFDFF]+')
    return arabic_pattern.search(text)

def clean_tweet(tweet):
    if contains_arabic(tweet):
        tweet = re.sub(r'https?://[^\s]+', ' ', tweet)  # Remove URLs
        tweet = re.sub(r'[^؀-ۿ]+', ' ', tweet)  # Remove non-Arabic characters
        tweet = re.sub(r'\W+', ' ', tweet)
        tweet = ' '.join([w for w in tweet.split() if len(w) > 1])  # Remove single character words
        tweet = ' '.join([lemma.lemmatize(x, nltk.corpus.reader.wordnet.VERB) for x in nltk.wordpunct_tokenize(tweet) if x not in stop_words])
    else:
        tweet = re.sub(r'https?://[^\s]+', ' ', tweet)  # Remove URLs
        tweet = re.sub(r'\$[^\s]+', ' ', tweet)        # Remove $symbols
        tweet = re.sub(r'\@[^\s]+', ' ', tweet)        # Remove @mentions
        tweet = re.sub(r'[^a-zA-Z\'\s]', ' ', tweet)  
        tweet = ' '.join([w for w in tweet.split() if len(w) > 1])  # Remove single character words
        tweet = ' '.join([lemma.lemmatize(x, nltk.corpus.reader.wordnet.VERB) for x in nltk.wordpunct_tokenize(tweet) if x not in stop_words])

    return tweet

def data_final(new_tweet):
    
    preprocessed_tweet = [clean_tweet(new_tweet)]
    
    #preprocessed_tweet = TweetTokenizer().tokenize(preprocessed_tweet)

    bow = joblib.load("bow.joblib")    
    #print(bow.get_feature_names_out())

    bow_df=bow.transform(preprocessed_tweet).toarray()
    return bow_df
