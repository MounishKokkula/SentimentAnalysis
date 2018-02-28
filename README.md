## Sentiment-enhanced Multidimensional Analysis of Online Social Networks - A Continuous learning approach (Python,Machine learning Naive bayes,MS SQL Server)

Note:Integration with MSSQL Servercoming pending. 

### Vader (Valence Aware Dictionary and sEntiment Reasoner) - a Sentiment Analysis library, is a very useful library for sentiment analysis. But it fails in analyzing short texts (such as tweets )

### Goal
Improve sentiment analysis on the short text by adding a machine learning approach to Vader. 

### Approach 
Use manually labeled data and compare it with the output from Vader analysis data.
In the case of contradiction between the outputs use the data with higher confidence. 
Construct a new data column based on the condition above. Treating the obtained data as training data - Feed the data to the Naive Bayes Classifier (for initial test purposes) and pickle the classifier. 
Use the pickled data for sentiment Analysis for new texts.

The output is manually analyzed and corrected in case of an error - the corrected output is given as an input to the model making it a continuous learning model

### Result 
The Hybrid model thus developed, would combine the reasoning of Vader with that of Naive Bayes from Machine learning with the power of continuous learning.

### CsvRead_VaderMod_CsvWrite.py
Reads data from the csv file and converts it into a data frame with pandas.
It is then used to work with Vader and determine the sentiment.
This is then compared with the manual sentiment grading in the CSV file.
In the case of contradiction between the outputs use the data with higher confidence. 
Construct a new data column based on the condition above. Treating the obtained data as training data - Feed the data to the Naive Bayes Classifier (for initial test purposes) and pickle the classifier. 
Use the pickled data for sentiment Analysis for new texts.

### tweets.csv
The CSV file containing the tweets/text to train the classifier.

### VaderNaivebayes_Testing.py
A simple test file to check the correctness of the results.


