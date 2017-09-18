from bs4 import BeautifulSoup
import urllib.request
import re   
from textblob import TextBlob    
import numpy as np
import pandas as pd 
import csv


def links_Analysis(name,website):
    
    File = urllib.request.urlopen(website).read()
    corpus = []
    soup = BeautifulSoup(File)

    for links in soup.find_all('a'):
        corpus.append(links.get('href'))
    
    corpus = list(filter(None,corpus))
    
    
    indices = []
    links = []

    for i in range(0,len(corpus)):
        review = re.sub('[^a-zA-Z]',' ',corpus[i])
        review = str(review)
        review = review.split()
        for word in review:
             if word == name:
                indices.append(i)
    for i in indices:
        links.append(corpus[i])
        
      
    links = list(set(links))
    
    avg_sents = []

    for i in range(0,len(links)):      
    
        total_sents = []
        if len(links) == 0:
            exit()
        website = urllib.request.urlopen(links[i])
        soup = BeautifulSoup(website)
        for parag in soup.find_all('p'):
            parag = str(parag)
            parag = re.sub('[^a-zA-z]', ' ', parag)
            analysis = TextBlob(parag)
            print(analysis.sentiment.polarity)
            if analysis.sentiment.polarity > 0.5:
                print(parag)
            total_sents.append(analysis.sentiment.polarity)
        
        sum = 0
        for sent in total_sents:
            sum = sum + sent
        
        if len(total_sents) != 0:
            avg_sents.append(sum/len(total_sents))            
      
             
    return indices, links,avg_sents


print("Hey! This program takes a keyword, goes through all the articles in marketwatch.com and extracts all the links related to that keyword. After collecting all the links, it performs sentiment analysis and returns the average sentiment per link(article)")     
 
keyword = input("Enter a keyword:(ex:netflix)")

indices,links, avg_sents = links_Analysis(str(keyword),"http://www.marketwatch.com/")    


if len(links) == 0:
   print("no articles")

num = 1
for i in links:
    print("Link%s: %s"% (num,i))
    num = num + 1
   

num = 1   
for i in avg_sents:
    print(("Sentiment Polarity for link %s is %s")% (num, i) )
    num = num + 1  
        





    
    
    


    
    
    
    
    
    
    
    
    
