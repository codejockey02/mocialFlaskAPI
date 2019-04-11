#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 03:03:04 2019

@author: priyesh_saraswat
"""
import pickle
from flask import Flask
from flask import request
from flask import render_template
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def create_word_features(words):
    useful_words = [
        word for word in words if word not in stopwords.words('english')]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict


app = Flask(__name__, template_folder='templates')

model = pickle.load(open("model.pkl", "rb"))


@app.route('/', methods=['GET'])
def index():
    return render_template('client.html')


@app.route('/getrating', methods=['POST'])
def predict():
    # getting the review from the user through the node API
    review = request.form['review']
    words = word_tokenize(review)
    words = create_word_features(words)
    return (model.classify(words))


if __name__ == "__main__":
    app.run("127.0.0.1", "8080", debug=True)
