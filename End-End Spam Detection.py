#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# In[4]:


# Data Collection from CSV File
df = pd.read_csv('spam.csv')
print(df)


# In[5]:


#Data Preparing
data = df.where((pd.notnull(df)),'')
data.head(10)


# In[6]:


#To know the information about the data
data.info()


# In[7]:


#Shape of the data
data.shape


# In[9]:


#Data Preprocessing
#Convert the data which in text mode into lowercase and special characters and numbers if needed
def preprocess_text(text):
    text = text.lower()
    return text
data['Category'] =data['Category'].apply(preprocess_text)


# In[10]:


#split the data into training and testing sets
x = data['Category']
y = data['Message']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[11]:


print(x)


# In[12]:


print(y)


# In[13]:


print(x.shape)


# In[14]:


print(y.shape)


# In[15]:


print(X_train)


# In[16]:


print(y_train)


# In[17]:


print(X_test)


# In[18]:


print(y_test)


# In[19]:


print(X_train.shape)


# In[20]:


print(y_train.shape)


# In[21]:


print(X_test.shape)


# In[22]:


print(y_test.shape)


# In[25]:


#Feature Engineering
from sklearn.feature_extraction.text import CountVectorizer
#Covert text data into numerical features using convertvector
v = CountVectorizer()
X_train_vectorized = v.fit_transform(X_train)


# In[26]:


print(X_train_vectorized)


# In[27]:


X_test_vectorized = v.fit_transform(X_test)
print(X_test_vectorized)


# In[28]:


y_train_vectorized = v.fit_transform(y_train)
print(y_train_vectorized)


# In[29]:


y_test_vectorized = v.fit_transform(y_test)
print(y_test_vectorized)


# In[36]:


#Model Training
#We train the data by using Machnine learning Model on the victorized data
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


# In[40]:


m = LogisticRegression()
m.fit(X_train_vectorized, y_train)


# In[43]:


p = m.predict(X_train_vectorized)
accuracy = accuracy_score(y_train, p)


# In[44]:


print("Accuracy on Training Data :", accuracy)


# In[45]:


P_on_test_data = m.predict(X_test_vectorized)
a_on_test_data = accuracy_score(y_test,P_on_test_data)


# In[46]:


print(a_on_test_data)


# In[47]:


#Model Evealuation
#Make predicitons on the test data and evaluting the model's performance
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)


# In[48]:


print(accuracy)


# In[51]:


report = classification_report(y_test, y_pred)
print(f'Accuracy : {accuracy}')
print(report)


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the vectorizer and fit it to your training data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
new_message = ["Congratulations, you won a free gift!", "Hey, can we meet for lunch?"]
new_msg_vectorized = vectorizer.transform(new_message)
# Make predictions using your trained model
predictions = model.predict(new_msg_vectorized)


# In[57]:


print(predictions)


# In[58]:


print(new_msg_vectorized)


# In[62]:


if(predictions[0] != 1):
    print(f"Message : {new_message[0]} => 'Spam mail'")
else:
    print(f"Message : {new_message[0]} => 'Ham mail'")


# In[67]:


if(predictions[0] == 0):
    print(f"Message : {new_message[1]} => 'ham mail'")
    print(f"Message : {new_message[0]} => 'spam mail'")
else:
    print(f"Message : {new_message[0]} => 'spam mail'")
    print(f"Message : {new_message[1]} => 'ham mail'")


# In[ ]:




