import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10000)




df = pd.read_csv("ecomdata.csv", encoding = "unicode_escape")
#df = data.dropna()
#print(df.head(5))


# Are there any missing values?
df.isnull().any()
df.isnull().sum()


# Filling in missing data
df.dropna(inplace=True)




# Check again
df.isnull().sum()



#How many unique product are there?

unique_product = df["StockCode"].nunique()
print('Number of unique products are :',unique_product)



#How many of each product are there?

each_product = df["Description"].value_counts().head()
print('Number of each products have :',each_product)





#Sort the 5 most ordered products from most to least.
most_sold_products = df["Description"].value_counts().sort_values(ascending=False).head()
print(most_sold_products)

plt.figure(figsize=(15,5))
sns.barplot(x = df.Description.value_counts().head(20).index, y = df.Description.value_counts().head(20).values, palette = 'gnuplot')
plt.xlabel('Description', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
plt.title('Top 20 Items purchased by customers', color = 'green', size = 20)
#plt.show()






#The 'C' in the invoices shows the canceled transactions. Since we will not use the canceled transactions, we should remove them.

df = df[~df["InvoiceNo"].str.contains("C", na = False)]




# Create a variable named 'TotalPrice' that represents the total earnings per invoice.

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

#print(df.head())


# the last date of purchase
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
#print(df['InvoiceDate'].max())
# make sure that none of the Recency values become zero
import datetime as dt
today_date = dt.datetime(2011, 12, 11)




rfm = df.groupby('CustomerID').agg({'InvoiceDate': lambda invoiceDate: (today_date - invoiceDate.max()).days,
                                     'InvoiceNo': lambda InvoiceNo: InvoiceNo.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


print(rfm.head())

rfm.columns = ['recency', 'frequency', 'monetary']
rfm = rfm[rfm["monetary"] > 0]
print(rfm.head(5))





#We need to score these values between 1 and 5. After scoring, we will segment it.



rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])


rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])


rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

print(rfm.head(5))







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







X = rfm[['recency_score', 'frequency_score', 'monetary_score']]

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
plt.figure(figsize=(15,10))
distortions=[]
sil_scores=[]
for i in range(2,10):
    kmeans= KMeans(n_clusters=i, n_init=10, init= 'k-means++', algorithm='full', max_iter=300)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)
    label= kmeans.labels_
    sil_scores.append(silhouette_score(X, label))
plt.plot(np.arange(2,10,1), distortions, alpha=0.5)
plt.plot(np.arange(2,10,1), distortions,'o' ,alpha=0.5)
#plt.show()



kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300)
#print(kmeans.fit(X))


#print(kmeans.labels_)

#rfm['kmeans_cluster'] = kmeans.labels_
# print(rfm[rfm['kmeans_cluster'] == 4].mean())
#
# print(rfm.mean())
#
# print(kmeans.inertia_)













#-----------------------------------------Apriori--------------------------------------------------------------------

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# df = df.copy()
# df.query('Quantity < -80000')
# df.query('Quantity > 80000')
#
# basket = df.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack()
# print(basket.shape)
#
# basket = basket.applymap(basket.iloc[0].value_counts())
# print(basket.head(1))