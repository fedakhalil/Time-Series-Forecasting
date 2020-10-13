"""
Data Description

The task is to forecast the total amount of products sold in every shop for next month.


File descriptions

sales.csv - the dataset. Daily historical data from January 2013 to October 2015.
shops.csv- supplemental information about the shops.

Data fields

shop_id - unique identifier of a shop
item_id - unique identifier of a product
item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
item_price - current price of an item
date - date in format dd/mm/yyyy
date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
shop_name - name of shop

"""

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


sales_data = pd.read_csv("sales.csv").copy()
shops = pd.read_csv("shops.csv").copy()


# In[4]:


sales_data.head()


# In[5]:


shops.head()


# In[6]:


sales_data.info()


# In[7]:


sales_data.isnull().sum()


# In[ ]:





# # Data Preprocessing

# In[8]:


sales_data.info()


# In[9]:


# convert date column to datetime variable 

sales_data["date"] = pd.to_datetime(sales_data["date"], format = "%d.%m.%Y")


# In[10]:


sales_data


# In[11]:


sales_data.describe().T


# In[12]:


# outliers

plt.figure(figsize = (8,6))
sns.boxplot(sales_data["item_cnt_day"], orient="v");


# In[13]:


# outliers

plt.figure(figsize = (8,6))
plt.boxplot(sales_data["item_price"]);


# In[14]:


# clean outliers from data

df = sales_data[(sales_data["item_cnt_day"] < 1000) & 
                 (sales_data["item_cnt_day"] >= 0) & 
                 (sales_data["item_price"] < 250000) &
                (sales_data["item_price"] > 0)]


# In[15]:


df


# In[16]:


df.describe().T


# In[17]:


# dataframe from date, shop_id and count per day

count_df = df[["date" , "shop_id", "item_cnt_day"]]
count_df


# In[18]:


# rename columns for Prophet model

count_df.rename(columns={"date" : "ds", "item_cnt_day" : "y"}, inplace=True)
count_df


# In[19]:


# dataframe group by for per month

count_df = count_df.set_index("ds").groupby([pd.Grouper(freq = "M"),
                                              "shop_id"]).sum().reset_index()
count_df


# In[20]:


# filter dataframe for shop_id with less than 3 records to avoid errors as prophet only works for 2+ records by group

count_df = count_df.groupby(["shop_id"]).filter(lambda x: len(x) > 2)
count_df


# In[ ]:





# In[21]:


# import Prophet model

from fbprophet import Prophet


# In[22]:


final = pd.DataFrame(columns=["shop_id","ds","yhat"])

grouped = count_df.groupby("shop_id")
for g in grouped.groups:
    group = grouped.get_group(g)
    m = Prophet()
    m.fit(group)
    future = m.make_future_dataframe(1, freq = "M") # forecast next single month
    forecast = m.predict(future)
    # add a column with shop id
    forecast["shop_id"] = g
    # concat result with dataframe named final
    final = pd.concat([final, forecast], ignore_index=True)


# In[23]:


final


# In[24]:


predicted_df = final[["ds", "shop_id", "yhat", "yhat_lower", "yhat_upper"]]
predicted_df


# In[ ]:





# ## Visualization

# In[26]:


import matplotlib.dates as mdates
import datetime as dt
import numpy as np


# In[28]:


# min and max labels for x axes in graphic

left = dt.date(2013, 1, 31)
right = dt.date(2015, 11, 30)


# In[29]:


fig, ax = plt.subplots(figsize = (12,10))

ax.plot(count_df[count_df["shop_id"] == 4].drop("shop_id", axis = 1).set_index("ds"), label = "Actual")
ax.plot(predicted_df[predicted_df["shop_id"] == 4].drop(["shop_id",
                                                      "yhat_lower", 
                                                      "yhat_upper"], axis = 1).set_index("ds"), label = "Prediction")
leg_lines = ax.get_lines()
plt.setp(leg_lines, linewidth=4)

myFmt = mdates.DateFormatter("%Y-%b")
ax.xaxis.set_major_formatter(myFmt)
fig.autofmt_xdate(rotation = 45, ha = "center")

legend_properties = {"weight":"bold", "size" : 15}
ax.legend(prop = legend_properties)

plt.xticks(np.arange(left, right, 30), fontsize = 12, fontweight = "bold", color = "r")
ax.set_yticks(np.arange(400,4000,200))
plt.yticks(fontsize = 12, fontweight = "bold", color = "r")

plt.title("Forecasting sales of 4th shop_id as example", fontsize = 16, fontweight = "bold", color = "b")
plt.show()


# In[ ]:





# In[30]:


# sort dataframe by date for split forecasted values

predicted_df = predicted_df.sort_values("ds", ignore_index=True)
predicted_df


# In[31]:


# forecasted values for November 2015

y_pred = predicted_df[predicted_df["ds"] > "2015-10-31"].reset_index(drop = True)
result = y_pred.merge(shops, on = "shop_id")
result


