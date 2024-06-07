#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


# In[2]:


# Load the dataset from the csv file using pandas
# best way is to mount the drive on colab and 
# copy the path for the csv file
data = pd.read_csv("creditcard.csv")


# In[3]:


data


# In[4]:


# Print the shape of the data
# data = data.sample(frac = 0.1, random_state = 48)
print(data.shape)
print(data.describe())


# In[3]:


# Determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))


# In[4]:


print("Amount details of the fraudulent transaction")
fraud.Amount.describe()


# In[10]:


# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[5]:


# dividing the X and the Y from the dataset
X = data.drop(['Class'], axis = 1).values
Y = data["Class"].values
print(X.shape)
print(Y.shape)
# getting just the values for the sake of processing 
# (its a numpy array with no columns)


# In[6]:


# Using Scikit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_Train,X_Test,Y_Train, Y_Test = train_test_split(
        X, Y, test_size = 0.2, random_state = 42)


# In[7]:


# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()


# In[ ]:





# In[ ]:


y_predict=rfc.predict(x_Test)


# In[8]:


rfc.fit(X_Train,Y_Train)


# In[9]:


# predictions
yPred = rfc.predict(X_Test)


# In[10]:


#Building all kinds of evaluating parameters


# In[11]:


# Evaluating the classifier
# printing every score of the classifier
# scoring in anything
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix


# In[13]:


n_outliers = len(fraud)
n_errors = (yPred != Y_Test).sum()
print("The model used is Random Forest classifier")

acc = accuracy_score(Y_Test, yPred)
print("The accuracy is {}".format(acc))

prec = precision_score(Y_Test, yPred)
print("The precision is {}".format(prec))

rec = recall_score(Y_Test, yPred)
print("The recall is {}".format(rec))

f1 = f1_score(Y_Test, yPred)
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(Y_Test, yPred)
print("The Matthews correlation coefficient is{}".format(MCC))


# In[14]:


# printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_Test, yPred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:




