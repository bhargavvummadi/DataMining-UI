# to scrape the twitter and get tweets
import snscrape.modules.twitter as sntwitter
import pandas as pd

'''
This function returns the json response with tweet date,user and actual text
'''
def twitterApi(q,c):
    query  = "python"
    query1 = "Elonmusk"
    tweets = []
    limits = 100



    for t in sntwitter.TwitterSearchScraper(query1).get_items():
        if(len(tweets)==limits):
            break
        else:
            tweets.append([t.date,t.user.username,t.content])

    df = pd.DataFrame(tweets,columns=['Date','User','Tweet'])

    return df.to_dict()
