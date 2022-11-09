import tweepy
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("cleaned_twitter_data.csv")
emotions = ["fear_intensity", 'anger_intensity', 'happiness_intensity','sadness_intensity']

for emotion in emotions:
    # plt.hist(df[emotion], bins=100, label=emotion, alpha=0.7)
    counts, edges =  np.histogram(df[np.isfinite(df[emotion])][emotion], bins=100)
    plt.plot(edges[:-1], counts, label=emotion.split("_")[0], linewidth=8)

plt.title("Emotion distribution", fontsize=20)
plt.xlabel("Intensity", fontsize=20)
plt.legend(fontsize=35)
plt.grid()
plt.show()