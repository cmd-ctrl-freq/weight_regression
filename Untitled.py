#!/usr/bin/env python
# coding: utf-8

# In[1]:


# I might want to create virtualenvs for future notebooks. This one is using the system interpreter.
import sys
print(sys.executable)

# If you want to do that in the future:


# ![image.png](attachment:image.png)

# In[2]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn scipy matplotlib')







# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[4]:


df = pd.read_csv('data.csv')
df = df.drop(columns=['Fasting'])
df


# In[5]:


# Step 3: Convert "DateTime" column to datetime objects
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Step 4: Prepare the input features (X) and the target variable (y)
X = df['DateTime'].astype(int).values.reshape(-1, 1)
y = df['Weight (lbs)'].values.reshape(-1, 1)


# In[6]:


# Step 5: Create a LinearRegression model
model = LinearRegression()


# In[7]:


# Step 6: Fit the model to the data
model.fit(X, y)


# In[8]:


# Step 8: Visualize the linear regression line and the data points
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('DateTime')
plt.ylabel('Weight (lbs)')
plt.title('Linear Regression')
plt.show()


# In[ ]:




