# import necessary libraries
import random
import string  # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

nltk.download('popular', quiet=True)  # for downloading packages

# uncomment the following only the first time
# nltk.download('punkt')  # first-time use only
# nltk.download('wordnet')  # first-time use only

# Reading in the corpus
with open('corpus.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "what's up", "hey",)
GREETING_RESPONSES = ["Hello. Good day! Tell me how can I help you?"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    global idx_list, tfidf_list, index
    index = 0
    robo_response = ''
    sent_tokens.append(user_response)
    tfidf_vec = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english')
    tfidf = tfidf_vec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    sort_vals = vals.argsort()[0]
    idx = [sort_vals[-2], sort_vals[-3], sort_vals[-4], sort_vals[-5], sort_vals[-6], sort_vals[-7],
           sort_vals[-8], sort_vals[-9], sort_vals[-10], sort_vals[-11], sort_vals[-12], sort_vals[-13]]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = [flat[-2], flat[-3], flat[-4], flat[-5], flat[-6], flat[-7], flat[-8],
                 flat[-9], flat[-10], flat[-11], flat[-12], flat[-13]]

    idx_list = idx
    tfidf_list = req_tfidf

    if req_tfidf[0] == 0:
        robo_response = "I am sorry! I don't understand you."
        return robo_response
    else:
        for i in range(3):
            res_data = sent_tokens[idx[i]].replace('\n', '<br>')
            robo_response = robo_response + res_data
            # robo_response += '<br><br>' + 'confidence: ' + str(round(100 * req_tfidf[i], 2)) + '%'
            robo_response += '<br><br>=============================================<br><br>'
        return robo_response


def load_response(index):
    robo_response = ''
    for i in range(index, index + 3):
        res_data = sent_tokens[idx_list[i]].replace('\n', '<br>')
        robo_response = robo_response + res_data
        # robo_response += '<br><br>' + 'confidence: ' + str(round(100 * tfidf_list[i], 2)) + '%'
        robo_response += '<br><br>=============================================<br><br>'

    return robo_response


app = Flask(__name__)


@app.route("/chatbot")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    # user_text = user_text.lower()

    if greeting(user_text) is not None:
        msg_bot = greeting(user_text)

    elif user_text == 'thanks' or user_text == 'thank you':
        msg_bot = 'You are welcome..'

    else:
        msg_bot = response(user_text)
        sent_tokens.remove(user_text)

    return msg_bot


@app.route("/getloadmore")
def get_load_more():
    msg_bot = ''
    global index
    index = index + 3
    if index < 12:
        msg_bot = load_response(index)
    return msg_bot


# app.run(debug=False)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
