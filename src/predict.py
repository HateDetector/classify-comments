import argparse
import csv
import os
import pandas as pd
import pickle
import warnings
from src.preprocess_utils import PreProcessUtils

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import tensorflow as tf
    import numpy as np
    from keras import models
    from keras.preprocessing.sequence import pad_sequences
    import sklearn
    import sklearn.svm
    from src.train_cnn import *
    from src.train_logreg import *
    from src.train_svm import *
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


pp = PreProcessUtils()


def preprocess(tweet):
    more_stopwords = ['rt', 'dm']
    pp_tweet = tweet
    pp_tweet = pp.remove_noise(pp_tweet, mentions=True, replacement=' ')
    pp_tweet = pp.normalise(pp_tweet, numbers=True, stopwords=True, other_stopwords=more_stopwords,
                            stem_words=True, replacement=' ')
    return pp_tweet


def load_basic_model(model_file, vectorizer_file):
    """Load the either the Logistic Regression or SVM model and vectorizer"""
    load_model = pickle.load(open(model_file, 'rb'))
    load_vectorizer = pickle.load(open(vectorizer_file, 'rb'))
    print(" ### Loading model complete ### ")
    return load_model, load_vectorizer


# basic or advanced passed as parameter
def load_cnn():
    """Load either the normal CNN or advanced CNN and tokenizer"""
    load_model = models.load_model("./models/cnn.keras")
    load_tokenizer = pickle.load(open("./models/cnn_tokenizer.pickle", 'rb'))
    print(" ### Loading model complete ### ")
    return load_model, load_tokenizer


def predict(df):

    cnn_model, cnn_tokenizer = load_cnn()
    svm_model, svm_vectorizer = load_basic_model("./models/svm.pickle",
                                                 "./models/svm_vectorizer.pickle")
    logreg_model, logreg_vectorizer = load_basic_model("./models/logreg.pickle",
                                                       "./models/logreg_vectorizer.pickle")
    logreg_scores = []
    svm_classes = []
    cnn_scores = []
    final_class = []

    for tweet in df['text']:

        # preprocess tweet
        try:
            pp_tweet = [preprocess(str(tweet))]
        except:
            print("Error")
            logreg_scores = [-1]
            svm_classes = [-1]
            cnn_scores = [-1]
            final_class = [-1]
            continue

        # format and predict with cnn
        predict_tok = cnn_tokenizer.texts_to_sequences(pp_tweet)
        predict_tok = pad_sequences(predict_tok, maxlen=140, padding='post')
        predict_tok = np.array(predict_tok, dtype=np.float32)
        result_prob = cnn_model.predict_proba(predict_tok)
        cnn_result = result_prob[0][0]
        cnn_scores.append(cnn_result)

        # format and predict with svm
        predict_vect = svm_vectorizer.transform(pp_tweet)
        result = svm_model.predict(predict_vect)
        svm_result = result[0]
        svm_classes.append(svm_result)

        # format and predict with logreg
        predict_vect = logreg_vectorizer.transform(pp_tweet)
        result_prob = logreg_model.predict_proba(predict_vect)
        logreg_result = result_prob[0][1]
        logreg_scores.append(logreg_result)

        logreg_is_hate = logreg_result > 0.50
        cnn_is_hate = cnn_result > 0.50
        svm_is_hate = svm_result == 1

        if logreg_is_hate and cnn_is_hate:
            final_class.append(1)
        elif logreg_is_hate and svm_is_hate:
            final_class.append(1)
        elif svm_is_hate and cnn_is_hate:
            final_class.append(1)
        else:
            final_class.append(0)


    # append scores to dataframe
    df['hate_score_logreg'] = logreg_scores
    df['hate_class_svm'] = svm_classes
    df['hate_score_cnn'] = cnn_scores
    df['hate_score_consensus'] = final_class

    return df


# def hate_max(row):
#     return hate_score
