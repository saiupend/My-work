#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np


# In[15]:


titanic_ds = pd.read_csv("titanic.csv")
titanic_hidden_ds = pd.read_csv("titanic_hidden.csv")


# In[16]:


titanic_ds


# In[17]:


titanic_hidden_ds


# In[18]:


titanic_ds.head()


# In[19]:


titanic_hidden_ds.head()


# In[20]:


titanic_ds.isnull().sum()


# In[21]:


x = titanic_ds[['Age','Parch','Pclass']]


# In[22]:


y = titanic_ds[['Survived']]


# In[23]:


from sklearn.preprocessing import LabelEncoder


# In[26]:


lb = LabelEncoder()
y = lb.fit_transform(y)


# In[27]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.30,random_state=40)


# In[28]:


from tensorflow.keras.models import Sequential


# In[18]:


pip install tensorflow


# In[29]:


import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))


# In[30]:


import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))


# In[31]:


from tensorflow.keras.models import Sequential


# In[32]:


from tensorflow.keras.layers import Dense


# In[33]:


from tensorflow.keras import layers


# In[34]:


from tensorflow.keras import activations


# In[35]:


from tensorflow.keras.losses import binary_crossentropy


# In[36]:


model = Sequential()


# In[37]:


model.add(Dense(30, input_dim = 3, activation = activations.relu))


# In[38]:


model.add(Dense(30, activation = activations.relu))


# In[39]:


model.add(Dense(1, activation=activations.sigmoid))


# In[40]:


model.summary()


# In[41]:


model.compile(loss='binary_crossentropy', metrics=['accuracy'])


# In[42]:


model.fit(x,y,epochs=5,verbose=1)


# In[43]:


prediction = model.predict(x)


# In[44]:


print(prediction[:2])


# In[45]:


from sklearn.metrics import accuracy_score


# In[46]:



accuracy = accuracy_score(y, prediction.round())


# In[47]:


print(accuracy)


# In[48]:


X1 = titanic_hidden_ds[['Age','Parch','Pclass']]


# In[49]:


Y1 = titanic_hidden_ds[['Survived']]


# In[50]:


model.compile(loss='binary_crossentropy', metrics=['accuracy'])


# In[51]:


model.fit(X1,Y1,epochs=5,verbose=1)


# In[52]:


prediction = model.predict(x)


# In[53]:


print(prediction[:2])


# In[54]:


from sklearn.metrics import accuracy_score


# In[55]:


accuracy = accuracy_score(y, prediction.round())


# In[56]:


print(accuracy)


# In[ ]:




