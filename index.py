# app.py
import flask
from flask import Flask, render_template, request
import pickle
from flask_cors import CORS
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
import re
import string
from flask_restplus import Resource, Api, fields
from werkzeug.utils import cached_property
from werkzeug.contrib.fixers import ProxyFix


app = Flask(__name__)
CORS(app)

model = pickle.load(open('Minor_project_ml_model.pickle', 'rb'))

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app,
          version='0.1',
          title='Our sample API',
          description='This is our sample API'
)

# home route
@app.route("/")
def home():
    return 'Home page'

# serving form web page
@app.route( '/get_sentiment', methods=['POST'] )
def analyse_text():
    data = request.data.decode('UTF-8')
    print("\nRequest : ", data)
    q = remove_noise(word_tokenize(data))

    prediction = model.classify(dict([token, True] for token in q))

    print(prediction)
    return prediction


if __name__ == "__main__":
    app.run( debug=False )  # for deployment turn it off(False)
