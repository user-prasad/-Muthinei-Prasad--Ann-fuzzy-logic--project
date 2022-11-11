#!/usr/bin/env python
# coding: utf-8

# # Music Recommendation using K-Means Clusering

# In[1]:


#1.import libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()


# In[2]:


data = pd.read_csv("music.csv")
data.head()


# In[3]:


len(data)


# In[4]:


#2.data preprocessing


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


df = data.drop(columns=[ 'id','release_date','year','name', 'artists'])
df.corr()


# In[8]:


##Data transformation
from sklearn.preprocessing import MinMaxScaler
datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
normalization = data.select_dtypes(include=datatypes)
for col in normalization.columns:
    MinMaxScaler(col)


# In[9]:


#importing K- means from sklearn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=30)
features = kmeans.fit_predict(normalization)
data['features'] = features
MinMaxScaler(data['features'])


# In[10]:


class Music_Recommendation():
    def __init__(self, dataset):
        self.dataset = dataset
    def recommend(self, songs, amount=1):
        distance = []
        song = self.dataset[(self.dataset.name.str.lower() == songs.lower())].head(1).values[0]
        rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]
        for songs in tqdm(rec.values):
            d = 0
            for col in np.arange(len(rec.columns)):
                if not col in [1,6,12,14,18]:
                    d = d + np.absolute(float(song[col]) - float(songs[col]))
            distance.append(d)
        rec['distance'] = distance
        rec = rec.sort_values('distance')
        columns = ['artists', 'name']
        return rec[columns][:amount]

recommendations = Music_Recommendation(data)


# In[11]:


recommendations.recommend("Lovers Rock", 5)


# In[12]:


recommendations.recommend("Lovers Rock", 10)


# In[13]:


recommendations.recommend("Lovers Rock", 15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




