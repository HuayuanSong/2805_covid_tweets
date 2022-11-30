import re
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
from collections import Counter
from copy import deepcopy


class Sentimentor:
    def __init__(self, twitter_data_path, sentiment_path, output_path, k=None):
        self.tweet_df = pd.read_csv(twitter_data_path).dropna()
        self.output_path = output_path

        if k != None:
            idxs = random.sample(list(np.arange(len(self.tweet_df))), min(k, len(self.tweet_df)))
            self.tweet_df = self.tweet_df.iloc[idxs]

        self.sentiment_df = pd.read_csv(sentiment_path, sep="\t")

        self.__run()
        self.__save()

    def __get_sentiment(self, tweet):
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

    def __get_week_number(self, tweet_timestamp):
        day = tweet_timestamp.split(" ")[0].split("-")
        date_obj = datetime.date(int(day[0]), int(day[1]), int(day[2]))
        week_number = date_obj.isocalendar()[1]
        return week_number

    def __get_hashtags(self, tweet):
        try:
            regex_code = "(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)"
            hashtags = re.findall(regex_code, tweet)
            return hashtags
        except TypeError:
            return []

    def __run(self):
        tqdm.pandas()
        sentiments = self.tweet_df['tweet'].progress_apply(self.__get_sentiment)

        tqdm.pandas()
        week_numbers = self.tweet_df['tweet_timestamp'].progress_apply(self.__get_week_number)

        self.tweet_df['sentiment_score'] = sentiments
        self.tweet_df['week_number'] = week_numbers
        return self.tweet_df

    def __group(self, df, column):
        grouped_by = {"mean": df.groupby("week_number")[column].mean(),
                      'std': df.groupby("week_number")[column].std(),
                      'N': df.groupby("week_number")[column].count()}
        return grouped_by

    def __save(self):
        self.tweet_df.to_csv(self.output_path, index=False)

    def plot_sentiment(self, title):
        grouped_by = self.__group(self.tweet_df, 'sentiment_score')
        week_numbers = grouped_by['mean'].keys()._data
        means = grouped_by['mean'].values
        stds = grouped_by['std'].values
        Ns = grouped_by['N'].values

        plt.plot(week_numbers, means, linewidth=9)

        # lower confidence
        lower = [stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[0] for i in range(len(Ns))]
        plt.plot(week_numbers, lower, color='black')

        # upper confidence
        higher = [stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[1] for i in range(len(Ns))]
        plt.plot(week_numbers, higher, color='black')

        for month_label, month_number in [('April', 4), ('May', 5), ('June', 6), ('July', 7)]:
            vline = datetime.date(2020, month_number, 1).isocalendar()[1]
            plt.vlines(vline, min(lower), max(higher), colors='black', label=month_label, linewidth=4,
                       linestyles='dashed')
            plt.text(vline + 0.1, min(lower), month_label, fontsize=20)

        plt.fill_between(week_numbers, lower, higher, alpha=0.2)

        plt.xticks(week_numbers)
        plt.title(title, fontsize=20)
        plt.xlabel("Week number", fontsize=20)
        plt.ylabel("Sentiment score", fontsize=20)
        # plt.grid()
        plt.show()

    def analyse_hashtags(self):
        hashtags = self.tweet_df['tweet'].apply(self.__get_hashtags)
        hashtags = [item for sublist in hashtags for item in sublist]
        values, counts = np.unique(hashtags, return_counts=True)
        sort_idx = np.argsort(-counts)

        print(values[sort_idx][0:100])
        print(counts[sort_idx][0:100])


    def plot_hashtags(self, hashtags, title):
        hashtags_column = self.tweet_df['tweet'].apply(self.__get_hashtags)
        df_temp = deepcopy(self.tweet_df)
        df_temp['hashtags'] = hashtags_column

        f = lambda row_hashtags: [hashtag in row_hashtags for hashtag in hashtags]
        idxs = np.array(list(df_temp['hashtags'].apply(f)))

        minn = 10 * 10 ** 6
        maxx = -10 * 10 ** 6

        for i, hashtag in enumerate(hashtags):
            df_temp_hashtag = df_temp.loc[idxs[:, i]]

            grouped_by = self.__group(df_temp_hashtag, 'sentiment')
            week_numbers = grouped_by['mean'].keys()._data
            means = grouped_by['mean'].values
            stds = grouped_by['std'].values
            Ns = grouped_by['N'].values

            plt.plot(week_numbers, means, linewidth=9, label=hashtag)
            # lower confidence
            lower = [stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[0] for i in range(len(Ns))]
            # upper confidence
            higher = [stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[1] for i in
                      range(len(Ns))]


            minn = min(minn, min(lower))
            maxx = max(maxx, max(higher))

        for month_label, month_number in [('April', 4), ('May', 5), ('June', 6), ('July', 7)]:
            vline = datetime.date(2020, month_number, 1).isocalendar()[1]
            plt.vlines(vline, minn, maxx, colors='black', linewidth=4,
                       linestyles='dashed')
            plt.text(vline + 0.1, minn, month_label, fontsize=20)

        plt.xticks(week_numbers)
        plt.title(title, fontsize=20)
        plt.xlabel("Week number", fontsize=20)
        plt.ylabel("Sentiment score", fontsize=20)
        plt.legend()
        # plt.grid()
        plt.show()

    def plot_emotions(self, title):
        emotions = ['fear', 'anger', 'happiness', 'sadness']
        minn = 10 * 10 ** 6
        maxx = -10 * 10 ** 6
        for emotion in emotions:
            grouped_by = self.__group(self.tweet_df, '{}_intensity'.format(emotion))

            week_numbers = grouped_by['mean'].keys()._data
            means = grouped_by['mean'].values
            stds = grouped_by['std'].values
            Ns = grouped_by['N'].values

            plt.plot(week_numbers, means, linewidth=9, label=emotion)

            # lower confidence
            lower = [stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[0] for i in range(len(Ns))]

            # upper confidence
            higher = [stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[1] for i in
                      range(len(Ns))]

            minn = min(minn, min(lower))
            maxx = max(maxx, max(higher))

        for month_label, month_number in [('April', 4), ('May', 5), ('June', 6), ('July', 7)]:
            vline = datetime.date(2020, month_number, 1).isocalendar()[1]
            plt.vlines(vline, minn, maxx, colors='black', linewidth=4,
                       linestyles='dashed')
            plt.text(vline + 0.1, minn, month_label, fontsize=20)

        plt.xticks(week_numbers)
        plt.title(title, fontsize=20)
        plt.xlabel("Week number", fontsize=20)
        plt.ylabel("Sentiment score", fontsize=20)
        plt.legend(fontsize=20)
        plt.show()


class Sentimentor_load(Sentimentor):

    def __init__(self, twitter_sentiment_path, sentiment_path):
        self.tweet_df = pd.read_csv(twitter_sentiment_path).dropna()
        self.tweet_df = self.tweet_df[self.tweet_df['tweet_timestamp'] > '2020-03-1 1:30:00']


        self.sentiment_df = pd.read_csv(sentiment_path, sep="\t")



if __name__ == "__main__":
    # sentimentor = Sentimentor("../../cleaned_twitter_data.csv",
    #                           "../../Data_Set_S1.txt",
    #                           "../../stråmand.csv",
    #                           "Sentiment score through the weeks",
    #                           k=100)

    # sentimentor = Sentimentor_load("../../UK_sentiment.csv", "../../Data_Set_S1.txt",
    #                                "UK sentiment score")
    # sentimentor.plot_hashtags(['lockdown', 'NHS', 'mentalhealth', 'remoteworking', 'pharma', 'Crypto'])
    # sentimentor.plot_hashtags(['lockdown', 'NHS'])
    # sentimentor.plot_emotions("UK emotions")
    # sentimentor.plot_sentiment()

    sentimentor = Sentimentor_load("../../CAN_sentiment.csv", "../../Data_Set_S1.txt")
    # sentimentor.plot_sentiment()
    # sentimentor.plot_emotions("hej")
    # sentimentor.plot_hashtags(['SocialDistancing', 'lockdown', 'realestate', 'mentalhealth', 'Trump'], "Canada Hashtags")
    # sentimentor.analyse_hashtags()
