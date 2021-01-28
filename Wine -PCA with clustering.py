#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[3]:


#Loading the dataset
wine=pd.read_csv("wine.csv")
wine.head()


# In[4]:


wine.describe()


# In[5]:


# Considering only numerical data 
wine.drop(['Type'],axis=1,inplace=True)
wine.describe()


# In[7]:


# Normalizing the numerical data 
wine_normal=scale(wine)

##Converting from float to Dataframe format 
wine_normal=pd.DataFrame(wine_normal) 


# In[8]:


pca=PCA(n_components=13)
pca_values=pca.fit_transform(wine_normal)


# In[9]:


var=pca.explained_variance_ratio_
var


# In[10]:


pca.components_[0]
pca.components_


# In[11]:


var1=np.cumsum(np.round(var,decimals=4)*100)
var1


# In[12]:


plt.plot(var1,color='red')


# ##### Clustering

# In[13]:


new_def=pd.DataFrame(pca_values[:,0:3])
new_def.head()


# In[14]:


new_def.describe()


# In[15]:


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch ##for creating dendrogram


# In[16]:


type(new_def)


# In[17]:


z=linkage(new_def,method="complete",metric="euclidean")


# In[38]:


plt.figure(figsize=(15, 5));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');
plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[36]:


# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=5,linkage='complete',affinity = "euclidean").fit(new_def) 
h_complete.labels_


# In[21]:


cluster_labels=pd.Series(h_complete.labels_)
wine['clust']=cluster_labels

# creating a  new column and assigning it to new column 
wine = pd.concat([cluster_labels,wine],axis=1)


# In[23]:


# getting aggregate mean of each cluster
wine.groupby(wine.clust).mean()


# In[24]:


# creating a csv file 
wine.to_csv("winehie.csv",index=False) #,encoding="utf-8")


# In[25]:


import os
os.getcwd()


# ##### K Means Clustering 

# In[26]:


new_df = pd.DataFrame(pca_values[:,0:3])


# In[27]:


from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


# ###### screw plot or elbow curve

# In[28]:


k = list(range(2,15))
k


# In[29]:


TWSS = [] # variable for storing total within sum of squares for each kmeans 

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_def)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_def.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_def.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[30]:


# Scree plot 
plt.plot(k,TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS");
plt.xticks(k)


# In[31]:


# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=3) 
model.fit(new_def)


# In[32]:


model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 


# In[33]:


wine['clust']=md # creating a  new column and assigning it to new column 
new_def.head()


# In[34]:


wine.groupby(wine.clust).mean()


# ### 
