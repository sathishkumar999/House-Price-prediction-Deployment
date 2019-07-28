
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv(r"F:\Data_Sets\house_price.csv")


# In[3]:


data=pd.DataFrame(data)


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.drop(["Unnamed: 0"],axis=1,inplace=True)


# In[9]:


data.columns


# In[10]:


data.info()


# In[11]:


data.describe()


# In[12]:


data=data.rename(columns={"Sq.ft":"Square_Feet"})


# In[13]:


data['Furnishing']=data['Furnishing'].map({1:"Yes",0:"No"})


# In[14]:


data['Furnishing'].astype('str').head()


# In[15]:


#data['Price'] = data['Price'].apply(np.floor)


# In[16]:


data['Price']=data['Price'].astype(int)


# In[17]:


data.dtypes


# In[18]:


#x=data.iloc[:, 0:6].values
#y= data.Price


# In[19]:


x = data.drop(['Price'], axis=1)
y = data['Price']


# In[20]:


x.head()


# In[21]:


y.head()


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=0)


# In[23]:


print(x_train.shape)
print(x_test.shape)


# In[24]:


print(y_train.shape)
print(y_test.shape)


# In[25]:


x_train.head()


# In[26]:


y_train.head()


# In[27]:


from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")


# In[28]:


categorical_features_indices = ['Location','Furnishing']
#categorical_features_indices = [0,2]


# In[29]:


model=CatBoostRegressor(cat_features=categorical_features_indices)


# In[30]:


from sklearn.model_selection import GridSearchCV


# In[31]:


#parameters = {'depth'         : list(range(1, 20, 2)),
#              'learning_rate' : [0.01, 0.05, 0.1],
#             'iterations'    : list(range(10, 200, 20))
#                }


# In[32]:


#grid_model = GridSearchCV(estimator=model, param_grid = parameters, cv = 5, n_jobs=-1)


# In[33]:


parameters = {'depth'         : [2,6,12],
              'learning_rate' : [0.01, 0.05, 0.1],
              'iterations'    : [20,100,200]
                 }


# In[34]:


grid_model = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)


# In[35]:


grid_model.fit(x_train, y_train) 


# In[36]:


#The best hyper parameters set
print("Best Hyper Parameters:", grid_model.best_params_)


# In[37]:


#The Best Score
print("The best score :",grid_model.best_score_)


# In[38]:


#print(grid_model.cv_results_)


# In[39]:


CTB=CatBoostRegressor(iterations=100,
                      depth=10,
                      learning_rate=0.1, 
                      loss_function='RMSE',
                      cat_features=categorical_features_indices)


# In[40]:


CTB_model=CTB.fit(x_train, y_train,eval_set=(x_test,y_test),plot=True)


# In[41]:


prediction = CTB_model.predict(x_test)


# In[42]:


prediction = prediction.astype(int)
print(prediction)


# In[43]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction))) 


# In[44]:


from sklearn.metrics import r2_score
test_score=r2_score(y_test,prediction)
print("Testing Score On Given Data: {:.4f}".format(test_score))


# In[45]:


df1 = pd.DataFrame({'Actual_price': y_test, 'Predicted_price': prediction})  
df1.head()


# In[46]:


new_data=x.iloc[0:100,]
new_data.head()


# In[47]:


#new_data=pd.DataFrame(new_data,columns=['Location', 'BHK', 'Furnishing', 'Square_Feet', 'Old(years)', 'Floor'])


# In[48]:


result=CTB_model.predict(new_data).astype(int)
result=pd.DataFrame(result,columns=['price'])
result.head()


# In[49]:


df=pd.concat([new_data,result],axis=1)
df.head()


# In[50]:


#saving the predictive data in csv format
df.to_csv(r"C:\Users\sathish kumar\Downloads\predictive data\house_price_prediction.csv",index=False)


# In[51]:


#reading the csv file
df=pd.read_csv(r"C:\Users\sathish kumar\Downloads\predictive data\house_price_prediction.csv")
df.head()


# In[52]:


import pickle


# In[53]:


#saving the trained model
with open('house_price_trained_model','wb') as file:
    pickle.dump(CTB_model,file)


# In[54]:


with open('house_price_trained_model','rb') as file:
    trained_model = pickle.load(file)


# In[55]:


prediction=trained_model.predict(x_test)
prediction[0:10].astype(int)


# In[56]:


# Saving model to disk
pickle.dump(CTB_model, open('house_model.pkl','wb'))

# Loading model to compare the results
house_trained_model = pickle.load(open('house_model.pkl','rb'))
print(house_trained_model.predict([['Bommanahalli',3,'Yes',3000,1,3]]))
#Actual Price:28000


# In[73]:


# Saving model to disk
pickle.dump(CTB_model, open(r"C:\Users\sathish kumar\Downloads\ML_Deployment\model",'wb'))

# Loading model to compare the results
trained_model = pickle.load(open(r"C:\Users\sathish kumar\Downloads\ML_Deployment\model",'rb'))
print(trained_model.predict([['Bommanahalli',3,'Yes',3000,1,3]]))
#Actual Price:28000

