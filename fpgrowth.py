import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpgrowth_py import fpgrowth
import time


data = pd.read_csv('ecomdata.csv', encoding = 'unicode_escape')
data['GroupPrice']=data['Quantity']*data['UnitPrice']
data=data.dropna()
# print('The dimensions of the dataset are : ', data.shape)
# print('---------')
#print(data.head(5))





"""
Removal of products that correspond to gifts offered by the company to customers. We keep only the products that the customer has actually put in his shopping cart.
We group all the products a customer has purchased together. Each line corresponds to a transaction composed of the invoice number, the customer ID and all the products purchased.
"""

liste= data['StockCode'].unique()
stock_to_del=[]
for el in liste:
    if el[0] not in ['1','2','3','4','5','6','7','8','9','10']: # products corresponding to gifts.
        stock_to_del.append(el)

data= data[data['StockCode'].map(lambda x: x not in stock_to_del)] # delete these products

basket = data.groupby(['InvoiceNo','CustomerID']).agg({'StockCode': lambda s: list(set(s))}) # grouping product from the same invoice.

# print('Dimension of the new grouped dataset : ', basket.shape)
# print('----------')
# print(basket.head(5))







#------------------------------Applying Fp Growth Asscociation------------------------------------------------------------------
a=time.time()
freqItemSet, rules = fpgrowth(basket['StockCode'].values, minSupRatio=0.005, minConf=0.3)
b=time.time()
print('time to execute in seconds : ',b-a, ' secs.')
print('Number of rules generated : ', len(rules))

association=pd.DataFrame(rules,columns =['basket','next_product','proba'])
association=association.sort_values(by='proba',ascending=False)
# print('Dimensions of the association table are : ', association.shape)
# print(association.head(5))






def compute_next_best_product(basket_el):
    """
    parameter : basket_el = list of consumer basket elements
    return : next_pdt, proba = next product to recommend, buying probability. Or (0,0) if no product is found.


    Description : from the basket of a user, returns the product to recommend if it was not found
    in the list of associations of the table associated with the FP Growth model.
    To do this, we search in the table of associations for the product to recommend from each
    individual product in the consumer's basket.

    """

    for k in basket_el:  # for each element in the consumer basket
        k = {k}
        if len(association[association[
                               'basket'] == k].values) != 0:  # if we find a corresponding association in the fp growth table
            next_pdt = list(association[association['basket'] == k]['next_product'].values[0])[
                0]  # we take the consequent product
            if next_pdt not in basket_el:  # We verify that the customer has not previously purchased the product
                proba = association[association['basket'] == k]['proba'].values[0]  # Find the associated probability.
                return (next_pdt, proba)

            return (0, 0)  # return (0,0) if no product was found.






def find_next_product(basket):
    """
    Parameter : basket = consumer basket dataframe
    Return : list_next_pdt, list_proba = list of next elements to recommend and the buying probabilities associated.

    description : Main function that uses the one above. For each client in the dataset we look for a corresponding
    association in the Fp Growth model table. If no association is found, we call the compute_next_best_product
    function which searches for individual product associations.
    If no individual ssociations are found, the function returns (0,0).

    """
    n = basket.shape[0]
    list_next_pdt = []
    list_proba = []
    for i in range(n):  # for each customer
        el = set(basket['StockCode'][i])  # customer's basket
        if len(association[association[
                               'basket'] == el].values) != 0:  # if we find a association in the fp growth table corresponding to all the customer's basket.
            next_pdt = list(association[association['basket'] == el]['next_product'].values[0])[
                0]  # We take the consequent product
            proba = association[association['basket'] == el]['proba'].values[0]  # Probability as sociated in the table
            list_next_pdt.append(next_pdt)
            list_proba.append(proba)


        elif len(association[association['basket']==el].values) == 0: # If no antecedent to all the basket was found in the table
            next_pdt,proba= compute_next_best_product(basket['StockCode'][i]) # previous function
            list_next_pdt.append(next_pdt)
            list_proba.append(proba)

    return(list_next_pdt, list_proba)











a=time.time()
list_next_pdt, list_proba = find_next_product(basket)
b=time.time()
print(b-a)
basket['Recommended Product']= list_next_pdt # Set of recommended products
basket['Probability']= list_proba # Set of rprobabilities associated
print(basket.head(5))