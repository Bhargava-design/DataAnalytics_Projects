#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras


# In[3]:


from keras.datasets import mnist


# In[4]:


from keras.models import Sequential


# In[5]:


from keras.layers import Dense,Dropout,Flatten


# In[6]:


from keras.layers import Conv2D,MaxPooling2D


# In[7]:


from keras import backend as b


# In[8]:


#spliting data into test and train it
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[9]:


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
input_shape=(28,28,1)


# In[10]:


#converting class vectors into binary
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)


# In[11]:


x_train=x_train.astype("float32")
x_test=x_test.astype("float32")


# In[12]:


x_train/=255
x_test/=255


# In[13]:


batch_size=128
num_classes=10
epochs=10


# In[14]:


model=Sequential()


# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D


# In[16]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))


# In[17]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[18]:


model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))


# In[19]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[20]:


model.add(Flatten())


# In[21]:


model.add(Dense(128,activation="relu"))


# In[22]:


model.add(Dropout(0.3))


# In[23]:


model.add(Dense(64,activation='relu'))


# In[24]:


model.add(Dropout(0.5))


# In[25]:


model.add(Dense(num_classes,activation='softmax'))


# In[26]:


from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])


# In[27]:


hist=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))


# In[28]:


score=model.evaluate(x_test,y_test,verbose=0)


# In[29]:


print("loss",score[0])


# In[30]:


print("accuracy",score[1])


# In[33]:


model.save('mnist.h5')


# In[34]:


model.save('my_model.keras')


# In[ ]:




