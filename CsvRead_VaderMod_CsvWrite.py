import pandas as pd
from glob import glob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pickle
import os
import nltk as nl
import random

# columns from input excel
sentiment = 'airline_sentiment'
sentimentConfd = 'airline_sentiment_confidence'
Vader = 'Vader'
# output excel & sheet name
outFilename = 'Output.xlsx'
sheet_name = 'Report'


# Assign spreadsheet filename to `file`

def getAllFiles(Regexformat):
    # returns iterator
    return glob(Regexformat)

def readCsvExcel(fileName):
    if fileName.split(".")[1].lower() == "csv":
        return pd.read_csv(fileName)
    elif  fileName.split(".")[1].lower() == "xlsx":
        return pd.ExcelFile(fileName)
    else:
        print("'%s' format not supported !" %fileName)

def VaderPolarity():
    posCount= 0
    neuCount= 0
    negCount= 0
    for index, row in df.iterrows():
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(row['text'])
        if vs['compound'] >= 0.5:
            posCount+=1
            df.set_value(index, 'Vader', 'positive')
        if vs['compound'] > -0.5 and vs['compound'] < 0.5:
            neuCount+=1
            df.set_value(index, 'Vader', 'neutral')
        if vs['compound'] <= -0.5:
            negCount+=1
            df.set_value(index, 'Vader', 'negative')
        # totalCount = posCount + negCount + neuCount
        # positive=format(100 * posCount / totalCount)
        # print("Positive tweets percentage:" + positive + "%")
        # negative = format(100 * negCount / totalCount)
        # print("Negative tweets percentage:"+ negative +"%")
        # neutral = format(100 * neuCount / totalCount)
        # print("Neutral tweets percentage:"+ str(neutral) +"% ")


def excelWriter():
    writer = pd.ExcelWriter(outFilename, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name=sheet_name)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    writer.save()

file = 'Tweets.xlsx'
df = pd.ExcelFile(file, sheetname='Tweets')

df = df.parse()
df['Index'] = df.index

# get vader polarity
VaderPolarity()
print(df)

# check if Vader and predefined sentiment does not match
# Not a concern if both match (i.e both Pos/Neg/Neu)
for index, row in df.iterrows():
    if row['Vader'] != row[sentiment]:
        # if predefined sentimentConfd greater than 70% then consider it as correct
        if row[sentimentConfd] > 0.7:
            df.set_value(index, 'ComputedSentiment', row[sentiment])
        else:
            df.set_value(index, 'ComputedSentiment', row['Vader'])
    else:
        df.set_value(index, 'ComputedSentiment', row['Vader'])

# print(df)

# write it to excel
excelWriter()
'''****************************************************'''
'''****************************************************'''
'''***********Naive bayes Machine learning ************'''
'''****************************************************'''
'''****************************************************'''

# Location of the classifier raw data on disk
csvFileLoc= "C:\\Users\\Mounish\\Documents\\books\\Information Retrieval\\Spam filter\\Enron Spam"

# declaring the HAM and SPAM data
pos_list = []
neu_list = []
neg_list = []

def create_word_features(words):
    my_dict = dict( [ (word, True) for word in words] )
    return my_dict

for index, row in df.iterrows():
    words = nl.word_tokenize(row['text'])
    # Marking the contents as pos/neu/neg
    if row['ComputedSentiment'] == 'positive':
        pos_list.append((create_word_features(words), "positive"))
    elif row['ComputedSentiment'] == 'neutral':
        neu_list.append((create_word_features(words), "neutral"))
    elif row['ComputedSentiment'] == 'negative':
        neg_list.append((create_word_features(words), "negative"))

combined_list = pos_list + neu_list + neg_list

random.shuffle(combined_list)

training_part = int(len(combined_list) * .7)

print("len(combined_list) " + str(len(combined_list)))
training_set = combined_list[:training_part]

# test_set = combined_list[training_part:]

print("len(training_set) "+str(len(training_set)))
# print(len(test_set))

classifier = nl.NaiveBayesClassifier.train(training_set)
print(classifier.show_most_informative_features())

# pickle the classifier
# Opening the pickle file
f = open('Vader_classifier.p', 'wb')
# dumping the pickle classifier
pickle.dump(classifier, f)
# closing the file
f.close()

