#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


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

# In[2]:


features, target = load_wine(return_X_y=True)
data=pd.DataFrame(features)
data['class']=target
data


# In[3]:


# # Making data Random

# In[3]:


data=data.loc[np.random.permutation(data.index)]


# # By getting features and Class

# In[4]:


Class_label=data['class']
Features_test=data.drop(columns=['class'])


# # Spliting class into 5 parts

# In[5]:


C_part1=Class_label[0:30]
C_part2=Class_label[30:60]
C_part3=Class_label[60:90]
C_part4=Class_label[90:120]
C_part5=Class_label[120:150]


# # Spliting Features into 5 parts

# In[6]:


F_part1=Features_test[0:30]
F_part2=Features_test[30:60]
F_part3=Features_test[60:90]
F_part4=Features_test[90:120]
F_part5=Features_test[120:150]


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------


# In[4]:


# # Testing on part1 according to fold1

# In[7]:


Features_test=F_part1
Class_label=C_part1


# # Loading KNN_trained_model1

# In[8]:


KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model1.txt"
with open(KNN_trained_model, 'rb') as file:
    KNN_trained_model = pickle.load(file)


# # Accuracy of KNN

# In[9]:


y_pred = KNN_trained_model.predict(Features_test)
print('1.Accuracy score of KNN= {:.2f}'.format(KNN_trained_model.score(Features_test, Class_label)))
knn1=KNN_trained_model.score(Features_test, Class_label)


# In[5]:


# # Precision, Recall, F1 of KNN

# In[10]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[6]:


# # Confusion Matrix of KNN

# In[11]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[7]:


# # Loading Decision_Trees_trained_model1

# In[12]:


Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model1.txt"
with open(Decision_Trees_trained_model, 'rb') as file:
    Decision_Trees_trained_model = pickle.load(file)


# In[8]:


# # Accuracy of  Decision Trees Algorithm

# In[13]:


y_pred = Decision_Trees_trained_model.predict(Features_test)

print('1.Accuracy score of Decision Trees= {:.2f}'.format(Decision_Trees_trained_model.score(Features_test, Class_label)))

dtc1=Decision_Trees_trained_model.score(Features_test, Class_label)


# In[9]:


# # Precision, Recall, F1 of Decision trees

# In[14]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[10]:


# # Confusion Matrix of decision trees

# In[15]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM) 


# In[11]:


# # Loading Random_forest_trained_model1

# In[16]:


Random_forest_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model1.txt"
with open(Random_forest_trained_model, 'rb') as file:
    Random_forest_trained_model = pickle.load(file)


# # Accuracy of  Random forest

# In[17]:


y_pred = Random_forest_trained_model.predict(Features_test)

print('1.Accuracy score of random forest= {:.2f}'.format(Random_forest_trained_model.score(Features_test, Class_label)))
RF1=Random_forest_trained_model.score(Features_test, Class_label)


# In[12]:


# # Precision, Recall, F1 of random forest

# In[18]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[13]:


# # Confusion Matrix of random forest

# In[19]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[14]:


# # Loading Logistic_regression_trained_model1

# In[20]:


Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model1.txt"
with open(Logistic_regression_trained_model, 'rb') as file:
    Logistic_regression_trained_model = pickle.load(file)


# # Accuracy of  Logistic regression

# In[21]:


y_pred = Logistic_regression_trained_model.predict(Features_test)

print('1.Accuracy score of logistic regression= {:.2f}'.format(Logistic_regression_trained_model.score(Features_test, Class_label)))
LR1=Logistic_regression_trained_model.score(Features_test, Class_label)


# In[15]:


# # Precision, Recall, F1 of logistic regression

# In[22]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[16]:


# # Confusion Matrix of logistic regression

# In[23]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------


# In[17]:


# # Testing on part2 according to fold2

# In[24]:


Features_test=F_part2
Class_label=C_part2


# In[18]:


# # Loading KNN_trained_model2

# In[25]:


KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model2.txt"
with open(KNN_trained_model, 'rb') as file:
    KNN_trained_model = pickle.load(file)


# # Accuracy of KNN

# In[26]:


y_pred = KNN_trained_model.predict(Features_test)
print('1.Accuracy score of KNN= {:.2f}'.format(KNN_trained_model.score(Features_test, Class_label)))
knn2=KNN_trained_model.score(Features_test, Class_label)


# In[19]:


# # Precision, Recall, F1 of KNN

# In[27]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[20]:


# # Confusion Matrix of KNN

# In[28]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)  


# In[21]:


# # Loading Decision_Trees_trained_model2

# In[29]:


Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model2.txt"
with open(Decision_Trees_trained_model, 'rb') as file:
    Decision_Trees_trained_model = pickle.load(file)


# # Accuracy of  Decision Trees Algorithm

# In[30]:


y_pred = Decision_Trees_trained_model.predict(Features_test)

print('1.Accuracy score of Decision Trees= {:.2f}'.format(Decision_Trees_trained_model.score(Features_test, Class_label)))

dtc2=Decision_Trees_trained_model.score(Features_test, Class_label)


# In[22]:


# # Precision, Recall, F1 of Decision trees

# In[31]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[23]:


# # Confusion Matrix of decision trees

# In[32]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM) 


# In[24]:


# # Loading Random_forest_trained_model2

# In[33]:


Random_forest_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model2.txt"
with open(Random_forest_trained_model, 'rb') as file:
    Random_forest_trained_model = pickle.load(file)


# # Accuracy of  Random forest

# In[34]:


y_pred = Random_forest_trained_model.predict(Features_test)

print('1.Accuracy score of random forest= {:.2f}'.format(Random_forest_trained_model.score(Features_test, Class_label)))
RF2=Random_forest_trained_model.score(Features_test, Class_label)


# In[25]:


# # Precision, Recall, F1 of random forest

# In[35]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[26]:


# # Confusion Matrix of random forest

# In[36]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[27]:


# # Loading Logistic_regression_trained_model2

# In[37]:


Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model2.txt"
with open(Logistic_regression_trained_model, 'rb') as file:
    Logistic_regression_trained_model = pickle.load(file)


# # Accuracy of  Logistic regression

# In[38]:


y_pred = Logistic_regression_trained_model.predict(Features_test)

print('1.Accuracy score of logistic regression= {:.2f}'.format(Logistic_regression_trained_model.score(Features_test, Class_label)))

LR2=Logistic_regression_trained_model.score(Features_test, Class_label)


# In[28]:


# # Precision, Recall, F1 of logistic regression

# In[39]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[29]:


# # Confusion Matrix of logistic regression

# In[40]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------


# In[30]:


# # Testing on part3 according to fold3

# In[41]:


Features_test=F_part3
Class_label=C_part3


# # Loading KNN_trained_model3

# In[42]:


KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model3.txt"
with open(KNN_trained_model, 'rb') as file:
    KNN_trained_model = pickle.load(file)


# # Accuracy of KNN

# In[43]:


y_pred = KNN_trained_model.predict(Features_test)
print('1.Accuracy score of KNN= {:.2f}'.format(KNN_trained_model.score(Features_test, Class_label)))
knn3=KNN_trained_model.score(Features_test, Class_label)


# In[31]:


# # Precision, Recall, F1 of KNN

# In[44]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[32]:


# # Confusion Matrix of KNN

# In[45]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM) 


# In[33]:


# # Loading Decision_Trees_trained_model3

# In[46]:


Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model3.txt"
with open(Decision_Trees_trained_model, 'rb') as file:
    Decision_Trees_trained_model = pickle.load(file)


# # Accuracy of  Decision Trees Algorithm

# In[47]:


y_pred = Decision_Trees_trained_model.predict(Features_test)

print('1.Accuracy score of Decision Trees= {:.2f}'.format(Decision_Trees_trained_model.score(Features_test, Class_label)))

dtc3=Decision_Trees_trained_model.score(Features_test, Class_label)


# In[34]:


# # Precision, Recall, F1 of Decision trees

# In[48]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[35]:


# # Confusion Matrix of decision trees

# In[49]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[36]:


# # Loading Random_forest_trained_model3

# In[50]:


Random_forest_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model3.txt"
with open(Random_forest_trained_model, 'rb') as file:
    Random_forest_trained_model = pickle.load(file)


# # Accuracy of  Random forest

# In[51]:


y_pred = Random_forest_trained_model.predict(Features_test)

print('1.Accuracy score of random forest= {:.2f}'.format(Random_forest_trained_model.score(Features_test, Class_label)))

RF3=Random_forest_trained_model.score(Features_test, Class_label)


# In[37]:


# # Precision, Recall, F1 of random forest

# In[52]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[38]:


# # Confusion Matrix of random forest

# In[53]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[39]:


# # Loading Logistic_regression_trained_model3

# In[54]:


Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model3.txt"
with open(Logistic_regression_trained_model, 'rb') as file:
    Logistic_regression_trained_model = pickle.load(file)


# # Accuracy of  Logistic regression

# In[55]:


y_pred = Logistic_regression_trained_model.predict(Features_test)

print('1.Accuracy score of logistic regression= {:.2f}'.format(Logistic_regression_trained_model.score(Features_test, Class_label)))

LR3=Logistic_regression_trained_model.score(Features_test, Class_label)


# In[40]:


# # Precision, Recall, F1 of logistic regression

# In[56]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[41]:


# # Confusion Matrix of logistic regression

# In[57]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------


# In[42]:


# # Testing on part4 according to fold4

# In[58]:


Features_test=F_part4
Class_label=C_part4


# # Loading KNN_trained_model4

# In[59]:


KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model4.txt"
with open(KNN_trained_model, 'rb') as file:
    KNN_trained_model = pickle.load(file)


# # Accuracy of KNN

# In[60]:


y_pred = KNN_trained_model.predict(Features_test)
print('1.Accuracy score of KNN= {:.2f}'.format(KNN_trained_model.score(Features_test, Class_label)))
knn4=KNN_trained_model.score(Features_test, Class_label)


# In[43]:


# # Precision, Recall, F1 of KNN

# In[61]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[44]:


# # Confusion Matrix of KNN

# In[62]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[45]:


# # Loading Decision_Trees_trained_model4

# In[63]:


Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model4.txt"
with open(Decision_Trees_trained_model, 'rb') as file:
    Decision_Trees_trained_model = pickle.load(file)


# In[46]:


# # Accuracy of  Decision Trees Algorithm

# In[64]:


y_pred = Decision_Trees_trained_model.predict(Features_test)

print('1.Accuracy score of Decision Trees= {:.2f}'.format(Decision_Trees_trained_model.score(Features_test, Class_label)))
dtc4=Decision_Trees_trained_model.score(Features_test, Class_label)


# In[47]:


# # Precision, Recall, F1 of Decision trees

# In[65]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[48]:


# # Confusion Matrix of decision trees

# In[66]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[49]:


# # Loading Random_forest_trained_model4

# In[67]:


Random_forest_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model4.txt"
with open(Random_forest_trained_model, 'rb') as file:
    Random_forest_trained_model = pickle.load(file)


# # Accuracy of  Random forest

# In[68]:


y_pred = Random_forest_trained_model.predict(Features_test)

print('1.Accuracy score of random forest= {:.2f}'.format(Random_forest_trained_model.score(Features_test, Class_label)))

RF4=Random_forest_trained_model.score(Features_test, Class_label)


# In[50]:


# # Precision, Recall, F1 of random forest

# In[69]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[51]:


# # Confusion Matrix of random forest

# In[70]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)   


# In[52]:


# # Loading Logistic_regression_trained_model4

# In[71]:


Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model4.txt"
with open(Logistic_regression_trained_model, 'rb') as file:
    Logistic_regression_trained_model = pickle.load(file)


# # Accuracy of  Logistic regression

# In[72]:


y_pred = Logistic_regression_trained_model.predict(Features_test)

print('1.Accuracy score of logistic regression= {:.2f}'.format(Logistic_regression_trained_model.score(Features_test, Class_label)))

LR4=Logistic_regression_trained_model.score(Features_test, Class_label)


# In[53]:


# # Precision, Recall, F1 of logistic regression

# In[73]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[54]:


# # Confusion Matrix of logistic regression

# In[74]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------


# In[55]:


# # Testing on part5 according to fold5

# In[75]:


Features_test=F_part5
Class_label=C_part5


# # Loading KNN_trained_model5

# In[76]:


KNN_trained_model = "Trained_models_with_5folds_&_tuning_parameters/KNN_trained_model5.txt"
with open(KNN_trained_model, 'rb') as file:
    KNN_trained_model = pickle.load(file)


# # Accuracy of KNN

# In[77]:


y_pred = KNN_trained_model.predict(Features_test)
print('1.Accuracy score of KNN= {:.2f}'.format(KNN_trained_model.score(Features_test, Class_label)))
knn5=KNN_trained_model.score(Features_test, Class_label)


# In[56]:


# # Precision, Recall, F1 of KNN

# In[78]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[57]:


# # Confusion Matrix of KNN

# In[79]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[58]:


# # Loading Decision_Trees_trained_model5

# In[80]:


Decision_Trees_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Decision_Trees_trained_model5.txt"
with open(Decision_Trees_trained_model, 'rb') as file:
    Decision_Trees_trained_model = pickle.load(file)


# # Accuracy of  Decision Trees Algorithm

# In[81]:


y_pred = Decision_Trees_trained_model.predict(Features_test)

print('1.Accuracy score of Decision Trees= {:.2f}'.format(Decision_Trees_trained_model.score(Features_test, Class_label)))

dtc5=Decision_Trees_trained_model.score(Features_test, Class_label)


# In[59]:


# # Precision, Recall, F1 of Decision trees

# In[82]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[60]:


# # Confusion Matrix of decision trees

# In[83]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[61]:


# # Loading Random_forest_trained_model5

# In[84]:


Random_forest_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Random_forest_trained_model5.txt"
with open(Random_forest_trained_model, 'rb') as file:
    Random_forest_trained_model = pickle.load(file)


# # Accuracy of  Random forest

# In[85]:


y_pred = Random_forest_trained_model.predict(Features_test)

print('1.Accuracy score of random forest= {:.2f}'.format(Random_forest_trained_model.score(Features_test, Class_label)))
RF5=Random_forest_trained_model.score(Features_test, Class_label)


# In[62]:


# # Precision, Recall, F1 of random forest

# In[86]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[63]:


# # Confusion Matrix of random forest

# In[87]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# In[64]:


# # Loading Logistic_regression_trained_model5

# In[88]:


Logistic_regression_trained_model = "Trained_models_with_5folds_&_tuning_parameters/Logistic_regression_trained_model5.txt"
with open(Logistic_regression_trained_model, 'rb') as file:
    Logistic_regression_trained_model = pickle.load(file)


# # Accuracy of  Logistic regression

# In[89]:


y_pred = Logistic_regression_trained_model.predict(Features_test)

print('1.Accuracy score of logistic regression= {:.2f}'.format(Logistic_regression_trained_model.score(Features_test, Class_label)))
LR5=Logistic_regression_trained_model.score(Features_test, Class_label)


# In[65]:


# # Precision, Recall, F1 of logistic regression

# In[90]:


from sklearn.metrics import classification_report, confusion_matrix


print('\n')
print("Precision, Recall, F1")
print('\n')
CR=classification_report(Class_label, y_pred)
print(CR)
print('\n')


# In[66]:


# # Confusion Matrix of logistic regression

# In[91]:


print('\n')
print("Confusion Matrix")
print('\n')
CM=confusion_matrix(Class_label, y_pred)
print(CM)    


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------


# In[67]:


# # Average performance of KNN algorithm

# In[112]:


average_result1=(knn1+knn2+knn3+knn4+knn5)/5
average_result1=round(average_result1,2)

print('The average accuracy of KNN algorithm is = ', average_result1 ,'%')


# In[68]:


# # KNN algorithm results with parameter tuning each fold from 1 to 5

# In[122]:


import plotly.graph_objects as go


print('\n\n\nKNN algorithm results with parameter tuning each fold from 1 to 5')
x = ["fold1", "fold2", "fold3", "fold4", "fold5"]
y = [knn1,knn2,knn3,knn4,knn5]
fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="sum", y=y, x=x, name="sum", marker_color='green'))
fig.show()


# In[69]:


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------

# # Average performance of Decision Trees algorithm

# In[123]:


average_result2=(dtc1+dtc2+dtc3+dtc4+dtc5)/5
average_result2=round(average_result2,2)
print('The average accuracy of Decision Trees algorithm is = ', average_result2 ,'%')



# In[70]:


# # Decision Trees algorithm results with parameter tuning each fold from 1 to 5

# In[124]:


import plotly.graph_objects as go


print('\n\n\nDecision Trees algorithm results with parameter tuning each fold from 1 to 5')
x = ["fold1", "fold2", "fold3", "fold4", "fold5"]
y = [dtc1,dtc2,dtc3,dtc4,dtc5]

fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="sum", y=y, x=x, name="sum", marker_color='blue'))
fig.show()


# In[71]:


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------

# # Average performance of Random forest algorithm

# In[125]:


average_result3=(RF1+RF2+RF3+RF4+RF5)/5
average_result3=round(average_result3,2)
print('The average accuracy of Random forest algorithm is = ', average_result3 ,'%')


# In[72]:


# # Random forest algorithm results with parameter tuning each fold from 1 to 5

# In[126]:


import plotly.graph_objects as go


print('\n\n\nRandom forest algorithm results with parameter tuning each fold from 1 to 5')
x = ["fold1", "fold2", "fold3", "fold4", "fold5"]
y = [RF1,RF2,RF3,RF4,RF5]

fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="sum", y=y, x=x, name="sum", marker_color='red'))
fig.show()


# In[73]:


# # ------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------

# # Average performance of Logistic regression algorithm

# In[127]:


average_result4=(LR1+LR2+LR3+LR4+LR5)/5
average_result4=round(average_result4,2)
print('The average accuracy of Logistic regression algorithm is = ', average_result4 ,'%')


# In[74]:


# # Logistic regression algorithm results with parameter tuning each fold from 1 to 5

# In[128]:


import plotly.graph_objects as go


print('\n\n\nLogistic regression algorithm results with parameter tuning each fold from 1 to 5')
x = ["fold1", "fold2", "fold3", "fold4", "fold5"]
y = [LR1,LR2,LR3,LR4,LR5]

fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="sum", y=y, x=x, name="sum", marker_color='yellow'))
fig.show()


# In[77]:


# # Comparison of all algorithms with their average accuracies

# In[134]:


import plotly.graph_objects as go
import numpy as np

x = ["KNN","Decision Trees","Random Forest","Logistic Regression"]
y = [average_result1, average_result2, average_result3, average_result4]

fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="sum", y=y, x=x, name="sum", marker_color='pink'))
fig.show()


# In[ ]:




