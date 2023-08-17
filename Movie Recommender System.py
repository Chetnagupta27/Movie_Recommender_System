#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.head()


# In[5]:


credits.head()


# In[6]:


movies=movies.merge(credits,on='title')


# In[7]:


movies.head(1)


# In[8]:


#genres
#id
#keywords
#title
#overview
#cast
#crew

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.head()


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[15]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[16]:


movies['genres']=movies['genres'].apply(convert)


# In[17]:


movies.head()


# In[18]:


movies['keywords']=movies['keywords'].apply(convert)


# In[19]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[20]:


movies['cast']=movies['cast'].apply(convert3)


# In[21]:


movies.head()


# In[22]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L
    


# In[23]:


movies['crew'].apply(fetch_director)


# In[24]:


movies.head()


# In[25]:


movies["overview"][0]


# In[26]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[27]:


movies.head()


# 'Sam Worthington'-->'SamWorthington'

# In[28]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[29]:


movies.head()


# In[30]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[31]:


movies.head()


# In[32]:


new_df=movies[['movie_id','title','tags']]


# In[33]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[34]:


new_df.head()


# In[35]:


new_df['tags'][0]


# In[36]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[37]:


new_df.head()


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[39]:


new_df['tags'][1]


# In[40]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[41]:


vectors[0]


# In[42]:


cv.get_feature_names()


# In[43]:


pip install nltk


# In[44]:


import nltk


# In[45]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[46]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return" ".join(y)


# In[47]:


ps.stem("loving")


# In[48]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver ')


# In[49]:


new_df['tags']=new_df['tags'].apply(stem)


# In[50]:


from sklearn.metrics.pairwise import cosine_similarity


# In[51]:


similarity=cosine_similarity(vectors)


# In[52]:


similarity


# In[53]:


sorted(similarity[0],reverse=True)


# In[57]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[58]:


recommend('Batman Begins')


# In[56]:


new_df.iloc[1216].title


# In[61]:


import pickle


# In[62]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[63]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[ ]:




