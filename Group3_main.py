from flask import Flask,render_template

import api
from sentiments import second
import os

app=Flask(__name__)
app.secret_key=os.urandom(24)
app.register_blueprint(second)


#rendering the landing page
@app.route('/')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')

#Accessing tweeter scraping module through api endpoint not used in finalized working
@app.route('/tweetapi')
def tweetApi():
    return api.twitterApi(1,2)

#takes data from the frontend i.e, keyword and number of tweets and render home.html
@app.route('/home',methods=['POST'])
def Home():
    print('function calling is working...')
    return render_template('home.html')

    # else:
    #     return redirect('/')




#runs the flask application with debugging the output on port 5000 which is default
if __name__=="__main__":
    app.run(debug=True)