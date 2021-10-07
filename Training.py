#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # Importing Libraries
import numpy as np 
import pandas as pd 
import pickle
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
import warnings
warnings.filterwarnings("ignore")


# In[2]:



# ## Loading data
features, target = load_wine(return_X_y=True)
data=pd.DataFrame(features)
data['class']=target
data


# In[3]:


# # checking null values

np.sum(data.isnull().any(axis=1))


# In[4]:



data.describe()


# In[5]:


# # class count in data

y=data['class']
y.value_counts()


# In[6]:


# # Frequency Distribution of class

get_ipython().run_line_magic('matplotlib', 'inline')
carrier_count = data["class"].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of class')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()


# In[7]:


# In[8]:


data["class"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# In[8]:



# # Making data Random


data=data.loc[np.random.permutation(data.index)]


# In[9]:



# # By getting features and Class


Class_label=data['class']
Features_train=data.drop(columns=['class'])


# In[10]:


# # Spliting class into 5 parts

C_part1=Class_label[0:30]
C_part2=Class_label[30:60]
C_part3=Class_label[60:90]
C_part4=Class_label[90:120]
C_part5=Class_label[120:150]


# In[11]:


# # Spliting Features into 5 parts

F_part1=Features_train[0:30]
F_part2=Features_train[30:60]
F_part3=Features_train[60:90]
F_part4=Features_train[90:120]
F_part5=Features_train[120:150]


# In[12]:



# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------

# # Training on part1, part2,part3 and part4 as kfold 1


parts=[F_part1,F_part2,F_part3,F_part4]
Features_train=pd.concat(parts)
part=[C_part1,C_part2,C_part3,C_part4]
Class_label=pd.concat(part)


# In[13]:


# # Train KNN algorithm on training set


knn=KNeighborsClassifier(n_neighbors=5, leaf_size=10)
knn= knn.fit(Features_train , Class_label)
knn


# In[14]:


# # Saving KNN_trained_model


KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model1.txt"
with open(KNN_trained_model, 'wb') as file:
    pickle.dump(knn, file)


# In[15]:


# # Train Decision Trees algorithm on training set



DTC=DecisionTreeClassifier(criterion='gini', random_state=10)
DTC= DTC.fit(Features_train , Class_label)
DTC


# In[16]:


# # Saving Decision_Trees_trained_model in txt file


Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model1.txt"
with open(Decision_Trees_trained_model, 'wb') as file:
    pickle.dump(DTC, file)


# In[17]:


# # Train Random Forest Algorithm on training set



from sklearn.ensemble import RandomForestClassifier
Ran_For= RandomForestClassifier(n_estimators=100,max_depth=10, random_state=100,max_leaf_nodes=100)
Ran_For= Ran_For.fit(Features_train , Class_label)
Ran_For


# In[18]:



# # Saving Random Forest _trained_model in txt file


Ran_For_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model1.txt"
with open(Ran_For_trained_model, 'wb') as file:
    pickle.dump(Ran_For, file)


# In[19]:


# # Train LogisticRegression Algorithm on training set


from sklearn.linear_model import LogisticRegression
LR= LogisticRegression(max_iter=100, random_state=10)
LR= LR.fit(Features_train , Class_label)
LR


# In[20]:










# # Saving Logistic_regression_ _trained_model in txt file



Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model1.txt"
with open(Logistic_regression_trained_model, 'wb') as file:
    pickle.dump(LR, file)


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------

# # Training on part2,part3 and part4, part5 as kfold 2



parts=[F_part2,F_part3,F_part4, F_part5]
Features_train=pd.concat(parts)
part=[C_part2,C_part3,C_part4, C_part5]
Class_label=pd.concat(part)


# # Train KNN algorithm on training set



knn=KNeighborsClassifier(n_neighbors=10, leaf_size=20)
knn= knn.fit(Features_train , Class_label)
knn


# # Saving KNN_trained_model


KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model2.txt"
with open(KNN_trained_model, 'wb') as file:
    pickle.dump(knn, file)


# # Train Decision Trees algorithm on training set



DTC=DecisionTreeClassifier(criterion='entropy', random_state=20)
DTC= DTC.fit(Features_train , Class_label)
DTC


# # Saving Decision_Trees_trained_model in txt file


Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model2.txt"
with open(Decision_Trees_trained_model, 'wb') as file:
    pickle.dump(DTC, file)


# # Train Random Forest Algorithm on training set



from sklearn.ensemble import RandomForestClassifier
Ran_For= RandomForestClassifier(n_estimators=200,max_depth=20, random_state=200,max_leaf_nodes=200)
Ran_For= Ran_For.fit(Features_train , Class_label)
Ran_For


# # Saving Random Forest _trained_model in txt file



Ran_For_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model2.txt"
with open(Ran_For_trained_model, 'wb') as file:
    pickle.dump(Ran_For, file)


# # Train LogisticRegression Algorithm on training set



from sklearn.linear_model import LogisticRegression
LR= LogisticRegression(max_iter=200, random_state=20)
LR= LR.fit(Features_train , Class_label)
LR


# # Saving Logistic_regression_ _trained_model in txt file


Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model2.txt"
with open(Logistic_regression_trained_model, 'wb') as file:
    pickle.dump(LR, file)


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------

# # Training on part1,part3 and part4, part5 as kfold 3


parts=[F_part1,F_part3,F_part4, F_part5]
Features_train=pd.concat(parts)
part=[C_part1,C_part3,C_part4, C_part5]
Class_label=pd.concat(part)


# # Train KNN algorithm on training set



knn=KNeighborsClassifier(n_neighbors=15, leaf_size=30)
knn= knn.fit(Features_train , Class_label)
knn


# # Saving KNN_trained_model


KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model3.txt"
with open(KNN_trained_model, 'wb') as file:
    pickle.dump(knn, file)


# # Train Decision Trees algorithm on training set



DTC=DecisionTreeClassifier(criterion='gini', random_state=30)
DTC= DTC.fit(Features_train , Class_label)
DTC


# # Saving Decision_Trees_trained_model in txt file


Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model3.txt"
with open(Decision_Trees_trained_model, 'wb') as file:
    pickle.dump(DTC, file)


# # Train Random Forest Algorithm on training set



from sklearn.ensemble import RandomForestClassifier
Ran_For= RandomForestClassifier(n_estimators=300,max_depth=30, random_state=300,max_leaf_nodes=300)
Ran_For= Ran_For.fit(Features_train , Class_label)
Ran_For


# # Saving Random Forest _trained_model in txt file


Ran_For_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model3.txt"
with open(Ran_For_trained_model, 'wb') as file:
    pickle.dump(Ran_For, file)


# # Train LogisticRegression Algorithm on training set


from sklearn.linear_model import LogisticRegression
LR= LogisticRegression(max_iter=300, random_state=30)
LR= LR.fit(Features_train , Class_label)
LR


# # Saving Logistic_regression_ _trained_model in txt file



Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model3.txt"
with open(Logistic_regression_trained_model, 'wb') as file:
    pickle.dump(LR, file)


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------

# # Training on part1,part2 and part4, part5 as kfold 4



parts=[F_part1,F_part2,F_part4, F_part5]
Features_train=pd.concat(parts)
part=[C_part1,C_part2,C_part4, C_part5]
Class_label=pd.concat(part)


# # Train KNN algorithm on training set

# In[45]:


knn=KNeighborsClassifier(n_neighbors=20, leaf_size=40)
knn= knn.fit(Features_train , Class_label)
knn


# # Saving KNN_trained_model



KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model4.txt"
with open(KNN_trained_model, 'wb') as file:
    pickle.dump(knn, file)


# # Train Decision Trees algorithm on training set


DTC=DecisionTreeClassifier(criterion='entropy', random_state=40)
DTC= DTC.fit(Features_train , Class_label)
DTC


# # Saving Decision_Trees_trained_model in txt file

Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model4.txt"
with open(Decision_Trees_trained_model, 'wb') as file:
    pickle.dump(DTC, file)


# # Train Random Forest Algorithm on training set

from sklearn.ensemble import RandomForestClassifier
Ran_For= RandomForestClassifier(n_estimators=400,max_depth=40, random_state=400,max_leaf_nodes=400)
Ran_For= Ran_For.fit(Features_train , Class_label)
Ran_For


# # Saving Random Forest _trained_model in txt file

Ran_For_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model4.txt"
with open(Ran_For_trained_model, 'wb') as file:
    pickle.dump(Ran_For, file)


# # Train LogisticRegression Algorithm on training set

from sklearn.linear_model import LogisticRegression
LR= LogisticRegression(max_iter=400, random_state=40)
LR= LR.fit(Features_train , Class_label)
LR


# # Saving Logistic_regression_ _trained_model in txt file


Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model4.txt"
with open(Logistic_regression_trained_model, 'wb') as file:
    pickle.dump(LR, file)


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------

# # Training on part1,part2 and part3, part5 as kfold 5

parts=[F_part1,F_part2,F_part3, F_part5]
Features_train=pd.concat(parts)
part=[C_part1,C_part2,C_part3, C_part5]
Class_label=pd.concat(part)


# # Train KNN algorithm on training set

knn=KNeighborsClassifier(n_neighbors=25, leaf_size=50)
knn= knn.fit(Features_train , Class_label)
knn


# # Saving KNN_trained_model


KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model5.txt"
with open(KNN_trained_model, 'wb') as file:
    pickle.dump(knn, file)


# # Train Decision Trees algorithm on training set


DTC=DecisionTreeClassifier(criterion='gini',  random_state=50)
DTC= DTC.fit(Features_train , Class_label)
DTC


# # Saving Decision_Trees_trained_model in txt file

Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model5.txt"
with open(Decision_Trees_trained_model, 'wb') as file:
    pickle.dump(DTC, file)


# # Train Random Forest Algorithm on training set


from sklearn.ensemble import RandomForestClassifier
Ran_For= RandomForestClassifier(n_estimators=500,max_depth=50, random_state=500,max_leaf_nodes=500)
Ran_For= Ran_For.fit(Features_train , Class_label)
Ran_For


# # Saving Random Forest _trained_model in txt file


Ran_For_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model5.txt"
with open(Ran_For_trained_model, 'wb') as file:
    pickle.dump(Ran_For, file)


# # Train LogisticRegression Algorithm on training set


from sklearn.linear_model import LogisticRegression
LR= LogisticRegression(max_iter=500, random_state=50)
LR= LR.fit(Features_train , Class_label)
LR


# # Saving Logistic_regression_ _trained_model in txt file


Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model5.txt"
with open(Logistic_regression_trained_model, 'wb') as file:
    pickle.dump(LR, file)

