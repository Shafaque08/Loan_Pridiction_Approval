#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[25]:


df=pd.read_csv("loan.csv")


# In[3]:


import os


# In[4]:


os.getcwd()


# In[12]:


os.chdir('C:\\Users\\KIIT\Downloads')


# In[22]:


df=pd.read_csv(r"C:\Users\KIIT\Downloads\loan.csv")


# In[23]:


df.head()


# In[26]:


df.info


# In[27]:


df.isnull().sum()


# In[28]:


df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[29]:


df.isnull().sum()


# In[30]:


df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)


# In[31]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmount_log = df.LoanAmount_log.fillna(df.LoanAmount_log.mean())

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df.isnull().sum()


# In[49]:


x=df.iloc[:,np.r_[1:5,9:11,13:15]].values
y=df.iloc[:,12].values

x


# In[50]:


y


# In[34]:


print("per of missing gender is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0])*100))


# In[35]:


print("number of people who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df,palette='Set1')


# In[36]:


print("number of people who take loan as group by marital status:")
print(df['Married'].value_counts())
sns.countplot(x='Married',data=df,palette='Set1')


# In[37]:


print("number of people who take loan as group by Dependents:")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents',data=df,palette='Set1')


# In[38]:


print("number of people who take loan as group by self employed:")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data=df,palette='Set1')


# In[39]:


print("number of people who take loan as group by loan amount:")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data=df,palette='Set1')


# In[40]:


print("number of people who take loan as group by credit history:")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data=df,palette='Set1')


# In[51]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import LabelEncoder
Labelencoder_x = LabelEncoder()


# In[52]:


for i in range(0,5):
    X_train[:,i]=Labelencoder_x.fit_transform(X_train[:,i])
    X_train[:,7]=Labelencoder_x.fit_transform(X_train[:,7])
    
X_train


# In[53]:


Labelencoder_y = LabelEncoder()
y_train = Labelencoder_y.fit_transform(y_train)

y_train


# In[55]:


for i in range(0,5):
    X_test[:,i] = Labelencoder_x.fit_transform(X_test[:,i])
    X_test[:,7] = Labelencoder_x.fit_transform(X_test[:,7])
    
X_test    


# In[57]:


Labelencoder_y = LabelEncoder()

y_test = Labelencoder_y.fit_transform(y_test)

y_test


# In[69]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train= ss.fit_transform(X_train)
x_test = ss.fit_transform(X_test)


# In[92]:


from sklearn.ensemble import RandomForestClassifier

rf_clf =  RandomForestClassifier()
rf_clf.fit(X_train, y_train)


# In[97]:


from sklearn import metrics
y_pred = rf_clf.predict(x_test)

print("Accuracy of Random Forest classifier is:", metrics.accuracy_score(y_pred, y_test))

y_pred


# In[113]:


from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)


# In[129]:


y_pred = nb_classifier.predict(X_test)
print("Accuracy of Gaussian Naive Bayes is %.", metrics.accuracy_score(y_pred, y_test))


# In[118]:


y_pred


# In[121]:


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)


# In[123]:


y_pred = dt_clf.predict(X_test)
print("Accuracy of of Dt is", metrics.accuracy_score(y_pred, y_test))


# In[124]:


y_pred


# In[128]:


from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train, y_train)


# In[130]:


y_pred = kn_clf.predict(X_test)
print("Accuracy of KN is", metrics.accuracy_score(y_pred, y_test))


# In[131]:


y_pred


# In[ ]:




