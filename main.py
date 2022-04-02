from unittest import skip

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('ItemList.csv', low_memory=False, sep='\t', header=None, names=['products'])
lg = len(data)
# print(data)
ls = list(data["products"].apply(lambda x: x.split(',')))
for x in ls[:]:
    del x[0]

te = TransactionEncoder()
te_data = te.fit(ls).transform(ls)
dataf = pd.DataFrame(te_data, columns=te.columns_)
df = apriori(dataf, min_support=0.001, use_colnames=True)
df.sort_values(by="support", ascending=False, inplace=True)
df['product'] = df['itemsets'].apply(lambda x: len(x))
df['sup'] = df['support'].apply(lambda x: x * lg)
df['rsup'] = df['support'].apply(lambda x: x * 100)
# df['confidence'] = df['itemsets'].apply(lambda x: {
#     if len(x) == 2: print(x)
# })
ls = list(df.values)
print(ls)

products = list(df['itemsets'].head(20).apply(lambda x: list(x)[0]).astype("unicode"))
suppost = list(df['support'].head(20))
products10 = list(df['itemsets'].head(10).apply(lambda x: list(x)[0]).astype("unicode"))
suppost10 = list(df['support'].head(10))

# print(products)
# print(suppost)
# df = df[(df.length == 2)]
print(df)

plt.figure(figsize=(15, 10))
plt.bar(products10, suppost10, color='maroon', width=0.4)
plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.show()

plt.figure(figsize=(15, 10))
sns.barplot(products, suppost, palette='gnuplot')
plt.xlabel('Items', size=15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size=15)
plt.title('Top 20 Items purchased by customers', color='green', size=20)
plt.show()


#
# # Intializing the list
# transacts = []
# # populating a list of transactions
# for i in range(0, data.size):
#     transacts.append([str(data.values[i, j]) for j in range(0, 20)])


# df = pd.read_csv('ItemList.csv', names=['products'], header=None)

# data = list(df["products"].apply(lambda x: x.split(',')))
#
# te = TransactionEncoder()
# te_data = te.fit(data).transform(data)
# df = pd.DataFrame(te_data, columns=te.columns_)
#
# df1 = apriori(df, min_support=0.01, use_colnames=True)
# df1.sort_values(by="support", ascending=False, inplace=True)
#
# df1['length'] = df1['itemsets'].apply(lambda x: len(x))



# df1 = df1[(df1.length == 2) & (df1.support >= 0.05)]

# plt.figure(figsize=(15, 5))
# sns.barplot(x=df1.itemsets.value_counts().head(20).index, y=df.itemsets.value_counts().head(20).values, palette='gnuplot')
# plt.xlabel('Items', size=15)
# plt.xticks(rotation=45)
# plt.ylabel('Count of Items', size=15)
# plt.title('Top 20 Items purchased by customers', color='green', size=20)

# plt.show()


