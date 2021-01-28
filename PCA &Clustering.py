#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[6]:


#Loading the dataset
df=pd.read_csv("wine.csv")
df.head()


# ### Visualize the data

# In[8]:


#Normalize the data
scaler=StandardScaler()
std=scaler.fit_transform(df)


# In[12]:


#fit standard data to PCA
pca = PCA()
pca.fit(std)


# In[14]:


pca.explained_variance_ratio_


# In[30]:


#we choose the pca components
pca=PCA(n_components=3)


# In[31]:


pca.fit(std)


# pca.transform(std)

# In[33]:


scores_pca=pca.transform(std)


# In[34]:


kmeans_pca=KMeans(n_clusters=3,init='k-means++',random_state=42)


# #we fit our data with the kmeans pca model
# kmeans_pca.fit(scores_pca)

# In[39]:


#Kmeans clustering with PCA Results
#create a new data frame with the original features and add the pca scaores & assigned clusters
df_pca_kmeans=pd.concat([df.reset_index(drop='True'),pd.DataFrame(scores_pca)],axis=1)
df_pca_kmeans.columns.values[-3:]=['Component1','Component2','Component3']

#The last column added will contain the pca kmeans clustering labels
df_pca_kmeans['Segment K-means PCA']=kmeans_pca.labels_


# In[40]:


df_pca_kmeans.head()


# In[41]:


df_pca_kmeans.info()


# In[44]:


#we should add the names of the segments to the labels.
df_pca_kmeans['Segment']=df_pca_kmeans['Segment K-means PCA'].map({0:'first',
                                                                    1:'second',
                                                                    2:'third'})


# In[53]:


#Plot data by PCA components
x_axis=df_pca_kmeans['Component2']
y_axis=df_pca_kmeans['Component1']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis,y_axis,hue=df_pca_kmeans['Segment'],palette=['green','red','yellow'])
plt.title('Clusters by PCA Components')
plt.show()

