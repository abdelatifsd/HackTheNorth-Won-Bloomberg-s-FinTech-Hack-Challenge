from bs4 import BeautifulSoup
import urllib.request
import re   
from textblob import TextBlob    
import numpy as np
import pandas as pd 
import csv


with open('abcnews-date-text.csv', 'r') as f:
    reader = csv.reader(f)
    (reviews) = list(reader)

with open("parags.tsv",'w') as tsvfile:
    fileWriter = csv.writer(tsvfile, delimiter = '\t')
    fileWriter.writerow(["Review"] + ["Liked"])
    for title in reviews:
         analysis = TextBlob(str(title))
         x = analysis.sentiment.polarity # we will classify the sentence as positive(1) or negative(0). The classification depends on the polarity
         if x >= 0:
            x = 1
         elif x < 0:
            x = 0
         fileWriter.writerow(title + [x])
        

dataset = pd.read_csv("parags.tsv", delimiter = '\t',quoting = 3)

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in  range(0,len(dataset)):
    review = re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [word for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
 
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 0)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

x = cm[0,0] + cm[1,1]
y = cm[0,1] + cm[1,0]
print("accuracy: ", (1 - y/x) * 100)


    
    
    


    
    
    
    
    
    
    
    
    
