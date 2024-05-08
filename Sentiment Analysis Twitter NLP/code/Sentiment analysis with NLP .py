#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('Twitter Sentiments.csv')


# In[4]:


df.head()


# In[5]:


df['tweet']


# In[6]:


df['label'].value_counts()


# In[7]:


df.info()


# In[8]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "",input_txt)
    return input_txt


# In[9]:


df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'],"@[\w]*")
df.head()


# In[10]:


df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
df.head()


# In[11]:


df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
df.head()


# In[12]:


tokenized_tweet = df['clean_tweet'].apply(lambda x : x.split())
tokenized_tweet.head()


# In[14]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda sentence:[stemmer.stem(word) for word in sentence])
tokenized_tweet.head()


# In[15]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
df['clean_tweet'] = tokenized_tweet
df.head()


# In[16]:


def hashtag_extract(tweets):
    hashtags = []
    # loop words in the tweet
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags    


# In[17]:


ht_positive = hashtag_extract(df['clean_tweet'][df['label']==0])

# extract hashtags from racist/sexist tweets
ht_negative = hashtag_extract(df['clean_tweet'][df['label']==1])


# In[18]:


ht_positive[:5]


# In[19]:


ht_positive = sum(ht_positive, [])
ht_negative = sum(ht_negative, [])


# In[20]:


ht_positive[:5]


# In[21]:


freq = nltk.FreqDist(ht_positive)
d = pd.DataFrame({'Hashtag': list(freq.keys()),
                 'Count': list(freq.values())})
d.head()


# In[22]:


d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d, x='Hashtag', y='Count')
plt.show()


# In[23]:


freq = nltk.FreqDist(ht_negative)
d = pd.DataFrame({'Hashtag': list(freq.keys()),
                 'Count': list(freq.values())})
d.head()


# In[24]:


# select top 10 hashtags
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d, x='Hashtag', y='Count')
plt.show()


# In[25]:


# feature extraction
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])


# In[26]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state=42, test_size=0.25)


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


# In[28]:


# training
model = LogisticRegression()
model.fit(x_train, y_train)


# In[29]:


# testing
pred = model.predict(x_test)
f1_score(y_test, pred)


# In[30]:


accuracy_score(y_test,pred)


# In[31]:


# use probability to get output
pred_prob = model.predict_proba(x_test)
pred = pred_prob[:, 1] >= 0.3
pred = pred.astype(np.int)

f1_score(y_test, pred)


# In[32]:


accuracy_score(y_test,pred)


# In[33]:


pred_prob[0][1] >= 0.3


# In[ ]:




