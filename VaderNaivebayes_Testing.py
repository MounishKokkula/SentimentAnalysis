import time
import pickle
import os
import nltk as nl
import random

# Unpickle the classifier
# Open the pickle file
f = open('Vader_classifier.p', 'rb')

# Load the pickle file
classifier = pickle.load(f)

# Close the file once we have the classifier unpickled
f.close()


def create_word_features(words):
    my_dict = dict([(word, True) for word in words])
    return my_dict



msg2 = '''Stupidity'''

msg1 = '''Hello mounish, he was rude'''

msg3 = '''pos'''


words = nl.word_tokenize(msg1)
features = create_word_features(words)

words = nl.word_tokenize(msg2)
features1 = create_word_features(words)

words = nl.word_tokenize(msg3)
features2 = create_word_features(words)

print("Message 1 is :", classifier.classify(features))

print("Message 2 is :", classifier.classify(features1))

print("Message 3 is :", classifier.classify(features2))

