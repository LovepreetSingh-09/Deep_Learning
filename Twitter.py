# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:37:03 2019

@author: user
"""

import pandas as pd
import graphviz
import tweepy
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
consumer_key='rc3NJwXJHzEG2ymimusoMMwIs'
consumer_secret='0Qdzu92XE45KNMQphC3piT9GspjhGIYl1r6jPngN7ILvjsWiBn '
access_token='3034204206-J59JZiKYrnzjMw69tUVyOd9zB0UQHk9ouTCKOTe'
access_token_secret='k2rzGa8tdLmFOIHHEns7CuP9QKp9c9QO7KURYgdYAcibG '
auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth)
tweets=api.search('Aritificial Intelligence')
data=pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['Tweets'])
display(data.head(10))
print(tweets[0].created_at)



