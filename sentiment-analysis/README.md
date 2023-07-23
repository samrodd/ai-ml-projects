The files in this directory should be downloaded and run in Anaconda Jupyter Notebook.

The files will allow you to user Tweepy to authenticate with the Twitter API and pull in tweets with specific keywords.

The original purpose of this analysis was to use sentiment analysis of tweets and see if positive or negative sentiment over
the course of a hours or days corresponded with positive/negative price movement of cryptocurrencies like Bitcoin and Ethereum.

The program uses nltk.sentiment.vader, a module in the Natural Language Toolkit (NLTK) library that provides access to 
the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. VADER is a sentiment analysis tool 
specifically designed for social media text, including emojis.

After reading in the tweets from Twitter based on the keyword provided, VADER scores the Tweet's positivity, negativity and neutrality
and buckets the tweet into one of those 3 groupings based on which has the highest score. 

This allows us to see what quantity of tweets are positive, negative and neutral of the ones we've queried from the API.

NOTE: This code will not run in its entirety unless you have Pro edition of the Twitter API. 
