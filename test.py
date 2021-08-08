import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10000)


df = pd.read_csv("finaldata.csv", encoding = "unicode_escape")
#df = data.dropna()
#print(df.head(5))


#How many unique product are there?

unique_product = df["StockCode"].nunique()
print('Number of unique products are :',unique_product)



#How many of each product are there?

each_product = df["Item_name"].value_counts().head()
#print('Number of each products have :',each_product)

#Sort the 5 most ordered products from most to least.
most_sold_products = df["Item_name"].value_counts().sort_values(ascending=False).head()
#print(most_sold_products)





plt.figure(figsize=(15,5))
sns.barplot(x = df.Item_name.value_counts().head(20).index, y = df.Item_name.value_counts().head(20).values, palette = 'gnuplot')
plt.xlabel('Item_name', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
plt.title('Top 20 Items purchased by customers', color = 'green', size = 20)
#plt.show()



# Create a variable named 'TotalPrice' that represents the total earnings per invoice.

df["TotalPrice"] = df["Quantity"] * df["Unit_price(TK)"]

#print(df.head())





# the last date of purchase
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
#print(df['InvoiceDate'].max())
# make sure that none of the Recency values become zero
import datetime as dt
today_date = dt.datetime(2021,7,27)




rfm = df.groupby('CustomerID').agg({'InvoiceDate': lambda invoiceDate: (today_date - invoiceDate.max()).days,
                                     'InvoiceNo': lambda InvoiceNo: InvoiceNo.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


print(rfm.head())

rfm.columns = ['recency', 'frequency', 'monetary']
rfm = rfm[rfm["monetary"] > 0]
#print(rfm.head(10))








#We need to score these values between 1 and 5. After scoring, we will segment it.



rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])


rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])


rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

#print(rfm.head(5))


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#print(rfm.head(5))





champions = rfm[rfm['segment'] == 'champions']
cant_loose = rfm[rfm['segment'] == 'cant_loose']

champ =  champions[['recency','frequency','monetary']].agg(['mean', 'count'])
#print(champ)
cant_loose_var =  cant_loose[['recency','frequency','monetary']].agg(['mean', 'count'])
#print(cant_loose_var)



# loyal_df = pd.DataFrame()
# loyal_df["loyal_customer_id"] = rfm[rfm["segment"] == "loyal_customers"].index
# loyal_df.head()
#
# loyal_df.to_excel("loyal_customers.xlsx", sheet_name='Loyal Customers Index')











#-------------------------------------------------APRIORI-------------------------------------------------------------

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


#Finding total number of unique customers:
total_cust = len(df["CustomerID"].unique())
print(total_cust)

#Creating dataset which contains number of distinct items in each customer’s market basket:
ListItem = df.groupby(['CustomerID'])['Item_name'].apply(list).values.tolist()
print(ListItem)


#Use ItemIndicator method to find desired dataset:
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)





#Final dataset which gives distint items in each customer’s market basket:
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)
nItemPurchase = df.groupby('CustomerID').size()
freqTable = pd.Series.sort_index(pd.Series.value_counts(nItemPurchase))
nItemPurchase = df.groupby('CustomerID').size()
freqTable = pd.Series.sort_index(pd.Series.value_counts(nItemPurchase))
final = pd.DataFrame(freqTable.index.values.astype(int), columns = ['Unique_Item_set'])
final['Number of customer per unique item'] = freqTable.values.astype(int)
print("Dataset for distinct items in each customer's market basket: \n",final)




frequent_itemsets = apriori(ItemIndicator, min_support = 0.05, max_len = 32, use_colnames = True)
print("Number of itemsets are: ",frequent_itemsets.count()[1])
print("Largest k value among the itemsets is: ",len(frequent_itemsets['itemsets'].apply(list).values.tolist()[-1]))


# Association rules for confidence matrix at atleast 1%
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print("Number of association rules are: ", assoc_rules.count()[1])



plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()



# Association rules for confidence matrix at atleast 60%
assoc_rules1 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
print("Association rules for 60% confidence: \n",assoc_rules1)










































# basket = df.groupby(['InvoiceNo','Item_name'])['Quantity'].sum().unstack()
# #print(basket.shape)
#
#
# basket = basket.applymap(lambda x: 1 if x>0 else 0)
# #print(basket.head(1))
#
# var = basket.iloc[0].value_counts()
# #print(var)
#
# itemsets = apriori(basket, min_support=0.05, use_colnames=True)
# print(itemsets.shape)
#
# freq_items = itemsets.sort_values('support',ascending=False).head()
# print(freq_items)
#
#
# rules = association_rules(itemsets, metric="lift", min_threshold=1)
# print(rules.shape)
#
#
# sns.scatterplot(x='support', y='confidence', hue='lift', data=rules)
# #plt.show()
#
# sns.scatterplot(x='support', y='confidence', hue='leverage', data=rules)
# #plt.show()
#
#
# result = rules.sort_values('lift', ascending=False).head(10)
# print(result)