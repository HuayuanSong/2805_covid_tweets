import tweepy
import re
import requests
import pandas as pd
import collections
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

class Sentimentor:
    def __init__(self, twitter_data_path, sentiment_path):
        self.tweet_df = pd.read_csv(twitter_data_path)
        self.tweet_df = self.tweet_df.iloc[0:30]

        self.sentiment_df = pd.read_csv(sentiment_path, sep="\t")

    def fun(self, tweet):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(tweet)


        freqs = collections.Counter(tokens)
        score = 0
        n = 0
        df1 = self.sentiment_df[self.sentiment_df["word"].isin(tokens)]
        for _, row in df1.iterrows():
            word = row['word']
            happiness_average = row['happiness_average']
            freq = freqs[word]
            score += freq * happiness_average
            n += freq

        if n != 0:
            score = score / n
        else:
            score = 0.0
        return score

    def get_sentiments(self):
        tqdm.pandas()
        sentiments =  self.tweet_df['tweet'].progress_apply(self.fun)

        self.tweet_df['sentiment_score'] = sentiments
        return self.tweet_df


tweet_with_sentiment_df = Sentimentor("cleaned_twitter_data.csv", "Data_Set_S1.txt").get_sentiments()
tweet_with_sentiment_df.to_csv("twitter_data_with_sentiment.csv", index=False)
