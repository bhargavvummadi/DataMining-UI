import os
import snscrape.modules.twitter as sntwitter
import pandas as pd
import csv, re
import requests

import matplotlib
import pickle
import operator
import urllib.request
from collections import Counter
from wordcloud import WordCloud


matplotlib.use('agg')
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, Flask, jsonify, redirect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


#fixing the route for static and template folders
second = Blueprint("second", __name__, static_folder="static", template_folder="template")
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
classifier = pickle.load(open('svm_model.sav', 'rb'))
# classifier = pickle.load(urllib.request.urlopen("https://drive.google.com/file/d/1w3d2Trmjqe5Cxs4VkFwsVlBQac9UJuhL/view?usp=share_link"))

positive_tweet_count = 0
negative_tweet_count = 0
searchkey = ''
df = pd.DataFrame([], columns=['Date', 'User', 'Tweet'])
result = []
'''
This function returns the rendering sentiment analyzer html file
'''
@second.route("/sentiment_analyzer")
def sentiment_analyzer():
    return render_template("sentiment_analyzer.html")

'''
 Sentiment Analysis class to analyze the tweets and performs all the required tasks in different 
 functions it mainly has DownloadData and percentage functions
'''
class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []
    '''
       tweet dataframe - df:param
        sentiment result:param
        % of positive and negative tweets:returns
        
    '''
    def DownloadData(self, df, result):




        # input for term to be searched and how many tweets to search
        # searchTerm = input("Enter Keyword/Tag to search about: ")
        # NoOfTerms = int(input("Enter how many tweets to search: "))


        # searching for tweets
        global positive_tweet_count
        positive_tweet_count = 0
        global negative_tweet_count
        negative_tweet_count = 0
        n_c = 0
        toal_tweets = len(df['Tweet'])
        k = 0
        for i in df['Tweet']:
            if result[k] == -1:
                negative_tweet_count+=1
            elif result[k] == 1:
                positive_tweet_count+=1
            else:
                n_c+=1
            k+=1


        # Open/create a file to append data to
        csvFile = open('result.csv', 'a')

        # Use csv writer
        csvWriter = csv.writer(csvFile)

        # creating some variables to store info

        positive = self.percentage(positive_tweet_count,toal_tweets)
        negative = self.percentage(negative_tweet_count,toal_tweets)

        return positive,negative

        #return polarity, htmlpolarity, positive, wpositive, spositive, negative, wnegative, snegative, neutral, keyword, tweets

    # function to calculate percentage
    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')



'''
this function receives the entered keyword and count performs all the required operations like
generating csv, wordcloud and sentiment percentages running the model
'''
@second.route('/sentiment_logic', methods=['POST', 'GET'])
def sentiment_logic():
    keyword = request.form.get('keyword')
    tweets = request.form.get('tweets')
    print('Entered keyword', keyword)
    print('Entered count of tweets ', tweets)
    tweets_list = []
    global  searchkey
    searchkey = keyword
    for t in sntwitter.TwitterSearchScraper(keyword).get_items():
        if (len(tweets_list) == int(tweets)):
            break
        else:
            tweets_list.append([t.date, t.user.username, t.content])
    global df
    df = pd.DataFrame(tweets_list, columns=['Date', 'User', 'Tweet'])
    print(df['Tweet'])
    df.to_csv('static/output.csv')
    vect_text = vectorizer.transform(df['Tweet'])
    global result
    result = classifier.predict(vect_text.toarray())
    k = 0
    for i in df['Tweet']:
        print('Text is ', i, "-", "sentiment score ", result[k])
        k += 1
    allwords = ''.join([t for t in df['Tweet']])
    wordcloud = WordCloud(width=3000, height=1500, random_state=1, background_color='black', colormap='Set2',
                          collocations=False).generate(allwords)

    worcl_bool = False
    plt.imshow(wordcloud,interpolation='bilinear')
    # plt.figure(figsize=[8,10])
    plt.axis('off')
    plt.savefig('static/images/wordcloud.png',bbox_inches='tight')
    sa = SentimentAnalysis()
    pos,neg  = sa.DownloadData(df,result)

    # return df.to_dict()
    htmlpolarity = ''
    if pos>neg:
        htmlpolarity = 'Positive'
    elif pos<neg:
        htmlpolarity = 'Negative'
    else:
        htmlpolarity = 'Neutral'
    worcl_bool = True

    # sa = SentimentAnalysis()
    # polarity,htmlpolarity,positive,wpositive, spositive,negative,wnegative, snegative,neutral,keyword1,tweet1=sa.DownloadData(keyword,tweets)
    return render_template('sentiment_analyzer.html', positive=pos,negative=neg,keyword=keyword,tweets=tweets,htmlpolarity = htmlpolarity,word_cloud = worcl_bool)


def handling_emojis(text):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', text)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', text)
    # Love -- <3, :*
    text = re.sub(r'(<3|:\*)', ' EMO_POS ', text)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', text)
    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', text)
    # Cry -- :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', text)

    return text

# cleaning the text

# removing tagged username '@'
def cleaningText(text):
    text = text.strip('\'"?!,.():;') # removing punctuation
    text = re.sub(r'(.)\1+', r'\1\1', text) # convert more than 2 letter repetitions to 2 letter #fooood -> food
    text = re.sub(r'(-|\')','',text) # removing additional -& '
    text = re.sub(r'@[A-Za-z0-9]+','',text) #removing @usernames
    text = re.sub(r'#','',text) #removing '#' symbols
    text = re.sub(r'RT[\s]+','',text) #removes RT(Re-Tweet) string
    text = re.sub(r'https?:\/\/\S+','',text) #removing the hyperlink
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', text) #removing urls
    # Replace 2+ dots with space
    text = re.sub(r'\.{2,}', ' ', text)
    # Strip space, " and ' from tweet
    text = text.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    text = handling_emojis(text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    text = text.lower() #make the text to lowercase
    return text

contractionWords = {
"aren’t":"are not","can’t":"can not","couldn’t":"could not ","didn’t":"did not","doesn’t":"does not","don’t":"do not","hadn’t":"had not","hasn’t":"has not ","haven’t":"have not",
"I’m":"I am","I’ve":"I have","isn’t":"is not","let’s":"let us","mightn’t":"might not","mustn’t":"must not","shan’t":"shall not","shouldn’t":"should not","that’s":" that is","he’ll":" he will",
"I’ll":"I will","she’ll":"she will","she’s":"she is","there’s":"there is","they’ll":" they will","they’re":"they are","they’ve":"they have","we’re":"we are","we’ve":"we have","weren’t":"were not",
"what’ll":"what will","what’re":"what are","what’ve":"what have","where’s":"where is","who’d":"who would","who’ll":"who will","who’re":"who are","who’s":"who is","who’ve":"who have","won’t":"will not",
"wouldn’t":"would not","you’d":"you would","you’re":"you are","you’ve":"you have","it’s":"it is","wasn't":"was not"
}


# negation handling
def negationHandling(text):
    words = text.split()
    temp = [contractionWords[word] if word in contractionWords else word for word in words]
    temp = " ".join(temp)
    return temp

#tokenizing the words
word_set = []
def wordTokenize(text):
    tokens = word_tokenize(text)
    return tokens

#remove the stop words
stop_words = stopwords.words('english')
def removeStopWords(tokens):
    temp = [word for word in tokens if word not in stop_words]
    return temp


# removing non alpha characters
def removeUnnecessaryChars(tokens):
    temp = [word for word in tokens if word.isalpha()]
    return temp

'''
generating plot data by calculating positive tweet and negative tweet data
 and rendering using the plotly on frontend
'''    
@second.route('/visualize')
def visualize():
    try:
        posv=positive_tweet_count
        negv=negative_tweet_count
        hasht = str(searchkey).capitalize()
        pos_list = []
        neg_list = []
        k = 0
        reqpositivetweetdate = []
        reqpositivetweetuser = []

        for i in df['Tweet']:
            if result[k] == -1:
                neg_list.append(i)
            elif result[k] == 1:


                pos_list.append(i)
            k += 1

        k = 0
        for d,u in zip(df['Date'],df['User']):

            if result[k] == 1:
                reqpositivetweetdate.append(str(d))
                reqpositivetweetuser.append(u)

            k += 1
        reqnegativetweetdate = []
        reqnegativetweetuser = []
        k = 0
        for d,u in zip(df['Date'],df['User']):

            if result[k] == -1:
                reqnegativetweetdate.append(str(d))
                reqnegativetweetuser.append(u)

            k += 1

        td_l = []
        for i in neg_list:
            t = cleaningText(i)
            t = negationHandling(t)
            t = wordTokenize(t)
            t = removeStopWords(t)
            t = removeUnnecessaryChars(t)
            td_l.append(t)
        final_neg_list = [item for subl in td_l for item in subl]
        neg_counts = Counter(final_neg_list)
        neg_counts = {k: v for k, v in sorted(neg_counts.items(), key=lambda item: item[1])}
    #----------------------------------
        td_l2 = []
        for i in pos_list:
            t = cleaningText(i)
            t = negationHandling(t)
            t = wordTokenize(t)
            t = removeStopWords(t)
            t = removeUnnecessaryChars(t)
            td_l2.append(t)
        final_pos_list = [item for subl in td_l2 for item in subl]
        pos_counts = Counter(final_pos_list)
        pos_worcl = False
        allwords = ''.join([t for t in pos_list])
        wordcloud = WordCloud(width=3000, height=3000, random_state=1, background_color='black', colormap='Set2',
                              collocations=False).generate(allwords)


        plt.imshow(wordcloud, interpolation='bilinear')
        # plt.figure(figsize=[8,10])
        plt.axis('off')
        plt.savefig('static/images/positivewordcloud.png', bbox_inches='tight')

        neg_worcl = False
        allwords = ''.join([t for t in neg_list])
        wordcloud = WordCloud(width=3000, height=3000, random_state=1, background_color='black', colormap='Set2',
                              collocations=False).generate(allwords)

        plt.imshow(wordcloud, interpolation='bilinear')
        # plt.figure(figsize=[8,10])
        plt.axis('off')
        plt.savefig('static/images/negativewordcloud.png', bbox_inches='tight')

        pos_counts = {k: v for k, v in sorted(pos_counts.items(), key=lambda item: item[1])}

        dash_pos = max(pos_counts.items(),key=operator.itemgetter(1))[0]
        neg_pos = max(neg_counts.items(), key=operator.itemgetter(1))[0]
        pos_counts_keys = list({key:val for key,val in pos_counts.items() if val!=1}.keys())
        neg_counts_keys = list({key:val for key,val in neg_counts.items() if val!=1}.keys())
        neg_counts_list = list({key:val for key,val in neg_counts.items() if val!=1}.values())
        pos_counts_list = list({key:val for key,val in pos_counts.items() if val!=1}.values())
        rj = df.to_dict()
        pos_worcl = True
        neg_worcl = True
        TRIVIA_URL = 'https://api.api-ninjas.com/v1/quotes?category=good'
        resp = requests.get(TRIVIA_URL, headers={'X-Api-Key': 'rpIdD2xmrc5nje11T6uhQg==vXMDFvthUdzrrjn1'}).json()
        # Get first trivia result since the API returns a list of results.
        trivia = resp[0]
        TRIVIA_URL_S = 'https://api.api-ninjas.com/v1/quotes?category=success'
        resp = requests.get(TRIVIA_URL_S, headers={'X-Api-Key': 'rpIdD2xmrc5nje11T6uhQg==vXMDFvthUdzrrjn1'}).json()
        # Get first trivia result since the API returns a list of results.
        trivias = resp[0]
        sucresp = trivias

        return render_template('PieChart.html',positive_count=posv,negative_count=negv,hashtag=hasht,fulltabledata=rj,positive_dash=dash_pos.capitalize(),negative_dash=neg_pos.capitalize(),
                               positivetweetlist = pos_counts_keys,positivecountslist=pos_counts_list,positivetweettabletweet=pos_list,positivetweettableuser=reqpositivetweetuser,positivetweettabledate = reqpositivetweetdate,
                               positivewordcloud=pos_worcl,negativekeys = neg_counts_keys,negvalues=neg_counts_list,negativetweettabletweet=neg_list,negativetweettableuser=reqnegativetweetuser,negativetweettabledate = reqnegativetweetdate,negativewordcloud=neg_worcl,trivia=trivia,sucresp =sucresp)
    except():
        print('error')


