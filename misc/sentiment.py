import tweepy
import re
import requests
import pandas as pd
import collections
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import os
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Sentimentor:
    def __init__(self, twitter_data_path, sentiment_path):
        self.tweet_df = pd.read_csv(twitter_data_path).dropna()

        idxs = random.sample(list(np.arange(len(self.tweet_df))), min(1*10**5, len(self.tweet_df)))
        self.tweet_df = self.tweet_df.iloc[idxs]

        self.sentiment_df = pd.read_csv(sentiment_path, sep="\t")

        self.run()

    def get_sentiment(self, tweet):
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

    def get_week_number(self, tweet_timestamp):
        day = tweet_timestamp.split(" ")[0].split("-")
        date_obj = datetime.date(int(day[0]), int(day[1]), int(day[2]))
        week_number = date_obj.isocalendar()[1]
        return week_number


    def run(self):
        tqdm.pandas()
        sentiments =  self.tweet_df['tweet'].progress_apply(self.get_sentiment)

        tqdm.pandas()
        week_numbers = self.tweet_df['tweet_timestamp'].progress_apply(self.get_week_number)

        self.tweet_df['sentiment_score'] = sentiments
        self.tweet_df['week_number'] = week_numbers
        return self.tweet_df

    def group(self):
        grouped_by = {"mean" : self.tweet_df.groupby("week_number")['sentiment_score'].mean(),
                      'std': self.tweet_df.groupby("week_number")['sentiment_score'].std(),
                      'N': self.tweet_df.groupby("week_number")['sentiment_score'].count()}
        return grouped_by


a = 2
sentimentor = Sentimentor("../../cleaned_twitter_data.csv", "../../Data_Set_S1.txt")
tweet_with_sentiment_df = sentimentor.tweet_df
tweet_with_sentiment_df.to_csv("../../twitter_data_with_sentiment.csv", index=False)


grouped_by = sentimentor.group()
week_numbers = grouped_by['mean'].keys()._data
means = grouped_by['mean'].values
stds = grouped_by['std'].values
Ns = grouped_by['N'].values


plt.plot(week_numbers, means, linewidth=9)

# lower confidence
lower = [stats.norm.interval(0.95, loc=means[i], scale=stds[i]/np.sqrt(Ns[i]))[0] for i in range(len(Ns))]
plt.plot(week_numbers, lower, color='black')


# upper confidence
higher = [stats.norm.interval(0.95, loc=means[i], scale=stds[i]/np.sqrt(Ns[i]))[1] for i in range(len(Ns))]
plt.plot(week_numbers, higher, color='black')

for month_label, month_number in [('April', 4), ('May', 5), ('June', 6) , ('July', 7)]:
    vline = datetime.date(2020, month_number, 1).isocalendar()[1]
    plt.vlines(vline, min(lower), max(higher), colors='black', label=month_label, linewidth=4, linestyles='dashed')
    plt.text(vline+0.1, min(lower), month_label, fontsize=20)
# plt.scatter(grouped_by.keys()._data, grouped_by.values, s=[300]*len(grouped_by), color='red')



plt.fill_between(week_numbers, lower, higher, alpha=0.2)



plt.xticks(week_numbers)
plt.title("Sentiment score through the weeks", fontsize=20)
plt.xlabel("Week number", fontsize=20)
plt.ylabel("Sentiment score", fontsize=20)
# plt.grid()
plt.show()

