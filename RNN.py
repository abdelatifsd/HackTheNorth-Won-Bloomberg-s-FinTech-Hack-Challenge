import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import urllib.request
import requests
import re


"""
Companies = ["L-T","RY","TD","BNS","SLF","CRM","FB","MSFT"]

link = "http://www.finance.yahoo.com/quote/RY/history?period1=1410840000&period2=1505534400&interval=1d&filter=history&frequency=1d"

link = link.split("/")

link[4] = Companies[0]

link = "/".join(link)

openSite = urllib.request.urlopen(link).read()
soup = BeautifulSoup(openSite,"lxml")

#<a class="Fl(end) Mt(3px) Cur(p)">
for links in soup.find_all('a'):
    print(links.get_all("href"))
"""



def nextDayForecast(trainingCsvFile):
    
    dataset = pd.read_csv(trainingCsvFile)
    
    training_data = dataset.iloc[0 : len(dataset) - 20,1:2].values
    
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_data)
    
    X_train = training_data[0:len(training_data) - 1]
    y_train = training_data[1:len(training_data)]
    
    X_train = np.reshape(X_train, (len(training_data) - 1,1,1))
    
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    
    regressor = Sequential()
    
    regressor.add(LSTM(units = 4,activation = "sigmoid", input_shape = (None,1)))
    
    regressor.add(Dense(units = 1))
    
    regressor.compile(optimizer = "adam", loss ="mean_squared_error")
    
    regressor.fit(X_train,y_train,batch_size = 32, epochs = 100)
    
    
    real_stock_price = dataset.iloc[len(dataset) - 20:len(dataset), 1:2].values
    
    inputs = real_stock_price 
    
    inputs = sc.fit_transform(inputs)
    
    inputs = np.reshape(inputs,(len(real_stock_price),1,1))
    
    predicted_stock_price = regressor.predict(inputs)
    
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    return real_stock_price, predicted_stock_price



co = ["EFX.csv","SHOP.csv","GOOGL.csv","IBM.csv","TD.csv"]

#for i in range(len(co)):
actual, forecasted = nextDayForecast(co[0])

def plotResults(x,y):
    
    plt.plot(actual,color="red",label="Red is the actual price")
    plt.plot(forecasted, color= "blue",label="Blue is the forecasted price")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show
    
    """
    if i == 0:
        plt.title("Equifax Stock")
    elif i == 1:
        plt.title("Shopify Stock")
    elif i == 2:
        plt.title("Google Stock")
    elif i == 3:
        plt.title("IBM Stock")
    elif i == 4:
        plt.title("TD Stock")
    plt.show()
    """
    
plotResults(actual,forecasted)
















