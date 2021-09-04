#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


pd.options.display.max_rows = 500
pd.options.display.max_columns = 40


# In[3]:


df = pd.read_csv('master_output_1.csv')
dx = pd.read_csv('master_output_2.csv')
dc = pd.read_csv('master_output_3.csv')
dt = pd.read_csv('master_output_4.csv')
dk = pd.read_csv('master_output_5.csv')


# In[4]:


df.shape, dx.shape, dc.shape, dt.shape, dk.shape, 


# In[5]:


df = df.append([dx, dc, dt, dk])
df.shape


# In[6]:


df.head()


# In[7]:


Fuel = ['Petro', 'Petroleum', 'IOCL', 'BPCL', 'Hindustan', 'station', 'service', 'oil', 'fuel']
Vehicle = ['car', 'auto', 'sai']
Bills = ['Vodafone', 'voda', 'airt', 'Idea', 'Docomo', 'Airtel', 'broadband', 'tatasky', 'dth', 'bharati', 'CDMA']
CC = ['BRN PYMT']
UPI = ['upi']
Ration = ['bigbasket', 'big basket', 'grophers', 'mart', 'market', 'supermarket', 'bigbazaar', 'super', 'dorabjee']
Ecom = ['Flipkart', 'amazon', 'ebay', 'olx', 'quikr', 'shaadi', 'jeevansathi', 'tinder', 'pepperfry', 'paypal']
Transfer = ['Transfer', 'party']
Food = ['bikaner', 'zomato', 'swiggy', 'restaurant', 'hotel', 'food', 'burger', 'pizza', 'cuisine', 'coffee', 'cafe', 'mcdonalds', 'subway', 'kfc', 'starbucks', 'taco', 'mc donalds', 'bundal', 'bundl', 'barbeque', 'chilis', 'krabi', 'jovos', 'grille', 'grill', 'chung', 'classic', 'afk', 'betos', 'sizzle', 'sizzler', 'sizzlers', 'effingut', 'mini punjab', 'teno']
Hotel = ['jadh', 'loung', 'lounge', 'resort', 'inn', 'hotel', 'westin', 'conrad', 'hilton', 'ibis', 'ginger', 'hyatt', 'twisted']
Commodity = ['pepperfry', 'urban']
Wines = ['wines', 'drinks', 'beer', 'liquors', 'wine', 'liquor']
Medical = ['med', 'medicals', 'hospital', 'clinic', 'chemist', 'pharmacy', 'pharmaceuticals', 'chemists', 'druggists', 'drugs', 'wellness', 'drug']
Cash = ['cash', 'ATM', 'brew', 'brewery']
Interest = ['Int', 'pd', 'interest']
Chq = ['clg', 'cheque']
Tax = ['GST', 'Charge', 'charges', 'consolidated', 'filing', 'filings']
Bank = ['bank', 'hdfc', 'axis', 'kotak', 'sbi', 'icici', 'cosmos', 'icicb']
Insurance = ['insurance', 'maxbupa', 'religare']
Rent = ['rent', 'rental', 'rentals']
Tickets = ['makemytrip', 'IRCTC', 'easemytrip', 'tripadvisor', 'goibibo']
Apparels = ['wear', 'luxury', 'wears', 'scents', 'appar', 'life style']
Movies = ['netflix', 'hotstar', 'cinemas', 'pvr', 'inox', 'imax', 'book my show', 'zee5', 'amazon prime', 'bigtree']
Property = ['realty']
Tech = ['mobile', 'technology', 'smartphone', 'mobiles', 'smartphones', 'solutions']
Investment = ['stock', 'stocks', 'shares', 'woodstock']
Refund = ['referral', 'refund', 'credit', 'reversal', 'rev', 'rewarde']
Edu = ['college', 'edureka', 'school', 'schools', 'udemy', 'udacity']
Family = ['MB:supriya', 'expenses']

categs = (Fuel, CC, Ecom, UPI, Bills, Ration, Transfer, Food, Wines, Vehicle, Medical, Cash, Interest, Chq, Hotel, Tax, Insurance, Rent, Tickets, Apparels, Property, Tech, Investment, Refund, Movies, Edu, Family)
cat_labels = ["Fuel", "CC", "Ecom", "UPI", "Bills", "Rations", "Transfer", "Food", "Wines", "Vehicles", "Medical", 'Cash', 'Interest', 'Cheque', 'Hotel', 'Tax', 'Insurance', 'Rentals', 'Tickets', 'Apparels', 'Property', 'Tech', 'Investment', 'Refund', 'Movies', 'Education', 'Family']

df["Label"] = np.select([df["PARTICULARS"].str.contains("|".join(i), case=False) for i in categs], cat_labels, "Others")


# In[8]:


# df["Label"] = np.select([df["PARTICULARS"].str.contains("|".join(i), case=False) for i in \
#                          (Fuel, CC, Ecom, UPI, Bills, Ration, Transfer, Food, Wines, Vehicle, Medical, Cash, Interest, Chq, Hotel, Tax, Insurance, Rent, Tickets, Apparels, Property, Restaurant, Tech, Investment, Refund, Movies)],
#                         ["Fuel", "CC", "Ecom", "UPI", "Bills", "Rations", "Transfer", "Food", "Wines", "Vehicles", "Medical", 'Cash', 'Interest', 'Cheque', 'Hotel', 'Tax', 'Insurance', 'Rentals', 'Tickets', 'Apparels', 'Property', 'Restaurant', 'Tech', 'Investment', 'Refund', 'Movies'], "Others")


# In[9]:


df[df.Label == 'Others']


# In[10]:


len(df[df.Label == 'Others'])


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.subplots(figsize=(14,4))
sns.countplot(df.Label, palette='plasma')


# In[14]:


df.reset_index(drop=True, inplace=True)


# In[15]:


df.shape


# In[16]:


# df.to_csv('bsa_training_data.csv', index=False)


# In[ ]:




