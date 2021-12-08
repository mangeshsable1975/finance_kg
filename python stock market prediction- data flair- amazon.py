#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import quandle package
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# In[ ]:


#get amazon stock data 

amazon = quandl.get("wiki/AMZN")
print(amazon.head())


# In[ ]:


#get data for adjusted close column
amazon =amazon[[;Adj.close']]
print(amazon.head())


# In[ ]:


#set forecast length to 30 days
forecast_len = 30
amazon['Predicted']= amazon[['Adj.Close']].shift(-forecast_len)
print(amazon.tail())


# In[ ]:


#drop predicted column & create a numpy array from it call 'x'

x = np.array(amazon.drop(['predicted'],1))
x= x[:-forecast_len]
print(x)


# In[ ]:


#create dependent dataset y & remove last 30 rows
y= np.array(amazon['predicted'])
y=y[:-forecast_len]
print(y)


# In[ ]:


#split the traon & test dataset
x_train , x_test, y_train , y_test = train_test_split(x,y,test_size=0.2)


# In[ ]:


#create an SVR model
svr_rbf = SVR(kernel='rbf', C= le3,gamma= 0.1)
svr_rbf.fit(x_train,y_train)


# In[ ]:


#create SVR model now & train it

svr_rbf= SVR(kernel='rbf',c=le3, gamma = 0.1)
svr_rbf.fit(x_train,y_train)


# In[ ]:


#get score of this model & print percentage
svr_rbf_confidence = svr_rbf.score(x_test, y_test)
print(f"SVR Confidence:{round(svr_rbf_confidence*100,2)}%")


# In[ ]:


#now create linearregression  model & train it
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


#get the core for this model
lr_confidence= lr.score(x_test , y_test)
print(f"Linear Regression Confidence:(round(lr_confidence*100,2)}%"")

