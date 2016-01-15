
# coding: utf-8

# In[12]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from xgboost.sklearn import XGBClassifier
#回去在mac上整，我就不在windows里面弄了。

#np.random.seed(0)#bench mark后面的split用，我在这里也不用弄了


# In[13]:

df_train = pd.read_csv('../airbnb_kaggle/train_users_2.csv')


# In[34]:

df_train2= df_train.drop([0], axis=0)
df_train2


# In[24]:

labels = df_train['country_destination'].values
labels.shape


# In[74]:

user_languages=pd.unique(df_train.language.ravel())
user_languages
#没有nan


# In[38]:

piv_train = df_train.shape[0]
piv_train #nrow，benchmark里面为了后面split用


# 我想给每个每行的user,根据langauge，在user表里面加9列destination_language_distance。
# 现在问题是
# 1.我不知道aribnb给的levenshtein distance是用什么比出来的，我算的和他给的不一样。我网上也找不到什么资料说levenshtein distance在不同的语言之间该怎么算。要继续用我算的distance来做么？
# 2.无关痛痒的一个问题，为什么df_country.destination_language，不对啊？df_country.country_destination都是对的。

# In[68]:

df_country=pd.read_csv('../airbnb_kaggle/countries.csv')
df_country


# In[66]:

#df_country['destination_km2']
df_country.destination_km2


# In[71]:

df_country.destination_language
#没天理啊，这为什么不行
'''
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-71-d37cfd9d5e28> in <module>()
----> 1 df_country.destination_language
      2 #country_destination

C:\Users\gogo\Anaconda2\lib\site-packages\pandas\core\generic.pyc in __getattr__(self, name)
   2358                 return self[name]
   2359             raise AttributeError("'%s' object has no attribute '%s'" %
-> 2360                                  (type(self).__name__, name))
   2361 
   2362     def __setattr__(self, name, value):

AttributeError: 'DataFrame' object has no attribute 'destination_language'


'''


# In[72]:

df_country.country_destination


# In[77]:

for l in user_languages:
    if l=='en':
        print("user language is english")
    else:
        print ("user language is "+l)


# In[ ]:

def get_user_dest_language_distance:
    if 


# In[3]:

import Levenshtein
#要下一堆东西
# http://zhidao.baidu.com/link?url=_tv2TRJ4B1YR08MfNROWmCQR9Us3_9TFuj62QdkE9NlomSz1H4nq0SezY9r_ONwwrQ4iTNLJHUM5gygHP9ECtIdQ-yU2O1jDDVALHsB7W0i
Levenshtein_distance('abc','ac')


# In[4]:


#字符串相似度算法
#!/usr/bin/env python
__author__ = 'Administrator'


def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]

def levenshtein_distance(first, second):
    """Find the Levenshtein distance between two strings."""
    if len(first) > len(second):
        first, second = second, first
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [range(second_length) for x in range(first_length)]
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]
            if first[i-1] != second[j-1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[first_length-1][second_length-1]


# In[6]:

levenshtein_distance("abc","ac")#1, correct answer


# In[8]:

levenshtein_distance("ABCDEFGHIJKLMNOPQRSTUVWXYZ","AÄBCDEFGHIJKLMNOÖPQRSTUÜVWXYZß")


# In[9]:

levenshtein("ABCDEFGHIJKLMNOPQRSTUVWXYZ","AÄBCDEFGHIJKLMNOÖPQRSTUÜVWXYZß")
#airbnb表上给的是72,72这个值肯定不是ratio. 就不知道他用什么算的distance了。


# In[ ]:
