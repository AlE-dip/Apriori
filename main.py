from builtins import print

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('ItemList.csv', low_memory=False, sep='\t', header=None, names=['products'])
lg = len(data)

ls = list(data["products"].apply(lambda x: x.split(',')))
for x in ls[:]:
    del x[0]

te = TransactionEncoder()
te_data = te.fit(ls).transform(ls)
dataf = pd.DataFrame(te_data, columns=te.columns_)
df = apriori(dataf, min_support=0.001, use_colnames=True)
df_ar = association_rules(df, metric="confidence", min_threshold = 0.1)
df.sort_values(by="support", ascending=False, inplace=True)
df['product'] = df['itemsets'].apply(lambda x: len(x))
# df.sort_values(by="product", ascending=False, inplace=True)
print(df.to_string())
df = df.head(10)
df['sup'] = df['support'].apply(lambda x: x * lg)
# df['rsup'] = df['support'].apply(lambda x: x * 100)

# def conf(itemsets):
#     if len(itemsets) > 1:
#         ls = list(itemsets)
#         df1 = df.loc[df['itemsets'].apply(lambda x: len(x) == 1 and list(x)[0]).astype("unicode") == ls[0]]
#         df2 = df.loc[df['itemsets'].apply(lambda x: x == itemsets)]
#         fl1 = df1.sup.values
#         fl2 = df2.sup.values
#         fl3 = float(fl2 * 100 / fl1)
#         return fl3

# df['confidence'] = df['itemsets'].apply(lambda x: conf(x))
# print(df[['itemsets', 'sup']])

products10 = list(df['itemsets'].apply(lambda x: list(x)[0]).astype("unicode"))
support10 = list(df['sup'])
df10 = pd.DataFrame(list(zip(products10, support10)), columns=['products', 'support'])
df10 = df10.reset_index()

# df1 = df.loc[df['itemsets'].apply(lambda x: len(x) == 2)]
# df1 = df1.loc[df1['confidence'].apply(lambda x: x > 5)]
# df1.sort_values(by="confidence", ascending=False, inplace=True)
# products = list(df1['itemsets'].head(10).apply(lambda x: nameProduct(x)).astype("unicode"))
# suppost = list(df1['confidence'].head(10).values)

def nameProduct(itemsets):
    df_ar1 = df_ar.loc[df_ar['antecedents'].apply(lambda x: x == itemsets)]
    consequents = df_ar1.consequents.values[0]
    lsCon = list(consequents)
    ls = list(itemsets)
    s = ''
    for x in ls[:]:
        s += x + ', '
    s = s[0:len(s) - 2]
    s += ' → '
    for x in lsCon[:]:
        s += x + ', '
    s = s[0:len(s) - 2]
    return s

df_ar.sort_values(by="confidence", ascending=False, inplace=True)
products = list(df_ar['antecedents'].head(10).apply(lambda x: nameProduct(x)).astype("unicode"))
confidence = list(df_ar['confidence'].head(10).values)
support = list(df_ar['support'].head(10).apply(lambda x: x * lg))
df1 = pd.DataFrame(list(zip(products, support, confidence)), columns=['products', 'support', 'confidence'])
df1 = df1.reset_index()
# print(df1)

# print(products)
# print(suppost)
# df = df[(df.length == 2)]
# print(df)

# df.to_csv(r'data\df.txt', header=True, index=True, sep='\t', mode='a')

# plt.figure(figsize=(15, 10))
# plt.bar(products10, suppost10, color='maroon', width=0.4)
# plt.xlabel("Product", size=14)
# plt.ylabel("Count of product", size=14)
# plt.title("Top 10 product purchased by customers'")
# plt.show()

# plt.figure(figsize=(15, 10))
# sns.barplot(data=df1, palette='gnuplot')
# # plt.xlabel('Items', size=15)
# plt.xticks(rotation=10)
# # plt.ylabel('%', size=15)
# plt.ylim(0, 1)
# plt.title('Confidence', color='green', size=10)
# plt.show()

plt.figure(figsize=(15, 10))
plots = sns.barplot(x="products", y="support", data=df10)
for bar in plots.patches:
    plots.annotate(int(bar.get_height()),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.xticks(rotation=10)
plt.xlabel("Product", size=14)
plt.ylabel("Count of product", size=14)
plt.title('Top 10 product purchased by customers', color='green', size=20)
plt.show()

# products10.append('other product')
# def otherProduct():
#     s=0
#     for x in support10[:]:
#         s += x
#     return lg - s
# support10.append(otherProduct())
# explode = []
# t = 0.1
# for i in range(11):
#     explode.append(t)
#     if t == 0.1: t = 0
#     else: t = 0.1
# plt.pie(support10, explode=explode, labels=products10,
#         autopct='%1.1f%%', shadow=True,
#         startangle=90,
#         wedgeprops={"edgecolor": "black",
#                     'linewidth': 2,
#                     'antialiased': True})
# plt.axis('equal')
# plt.show()

# Defining the plot size
plt.figure(figsize=(15, 10))
plots = sns.barplot(x="products", y="confidence", data=df1, palette='gnuplot')
i = 0
for bar in plots.patches:
    plots.annotate(int(support[i]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
    i += 1
plt.xticks(rotation=10)
plt.xlabel("Items", size=14)
plt.ylabel("%", size=14)
plt.title('Confidence', color='green', size=20)
plt.show()

# Show top 10
# def showTopOfDf(itemsets):
#     df_ar1 = df_ar.loc[df_ar['antecedents'].apply(lambda x: x == itemsets)].copy()
#     df_ar1['confi'] = df_ar1['confidence'].apply(lambda x: x * 100)
#     df_ar1.sort_values(by="confidence", ascending=False, inplace=True)
#     consequents = list(df_ar1['consequents'].head(10).apply(lambda x: nameProduct(x)).astype("unicode"))
#     confidence = list(df_ar1['confi'].head(10).values)
#     if consequents and confidence:
#         plt.figure(figsize=(15, 10))
#         sns.barplot(consequents, confidence, palette='gnuplot')
#         plt.xlabel('Product', size=15)
#         plt.xticks(rotation=10)
#         plt.ylabel('Percent(%)', size=15)
#         plt.ylim(0, 100)
#         plt.title('Top 10 product bundled with ' + nameProduct(itemsets), color='green', size=20)
#         plt.show()
#
# df['itemsets'].apply(lambda x: showTopOfDf(x))

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
# print(df)
# print(df1)
print(df_ar.to_string())
# df_ar.to_csv(r'data\df.txt', header=True, index=True, sep='\t', mode='a')

# print(list(df.head(10).values))
#
# fig, ax =plt.subplots(1,1)
# data = list(df.head(10).values)
# column_labels = list(df.columns)
# ax.axis('tight')
# ax.axis('off')
# ax.table(cellText=data,colLabels=column_labels,loc="center")
#
# plt.show()


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


