#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:06:25 2022

@author: Drew Tatum
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Relevance from http://ir.dcs.gla.ac.uk/resources/test_collections/medl/
columns=['Query', 'Doc ID'] 
df = pd.read_table('./TestFiles/ML/MED.REL', sep=' ', header=None, usecols=[0,2])  # Holds the query ID and it's relevant doc IDs
df.columns = columns

# Results from system_test_query.py
results = pd.read_csv('./TestFiles/test_results.csv')
topic_weights_lst = results['Topic Weight'].unique()

# Going through the results and counting how many times the returned document was relevant according the MedLine test collection 
n_queries = 30
n_retrieved = 8
final_results = {}
for query_num in range(1, n_queries+1):
    ## Looking in dataframe for the specific query
    query_df = results[results['Query #'] == query_num]
    for weight in topic_weights_lst:  # Going through each weighting scheme
        relevant_count = 0
        weight = round(weight,1)
        weight_df = query_df[query_df['Topic Weight'] == weight]
        returned_documents = weight_df['Doc ID'].values  # Getting the documents the system returned 
        # Check if the document is relevant 
        relevant_query = df[df['Query'] == query_num]
        relevant_docs = relevant_query['Doc ID'].values
        
        for doc in returned_documents:  # Looping through returned documents by system
            if doc in relevant_docs:  # Checking if document was part of the relevant collection 
                relevant_count += 1
        
        # Updating results (Results are the term weight with the total number of relevant returned documents)
        if weight not in final_results.keys():
            final_results[weight] = relevant_count
        else:
            final_results[weight] += relevant_count

        
x = []  # Term Weights
y = []  # Precision
for key in final_results.keys():
    x.append(key)
    y.append(final_results[key]/(n_queries*n_retrieved))  # Dividing by number of queries and number of documents returned to get precision


# Plot of the results 
sns.barplot(x= x, y=y)
plt.xlabel('Topic Weight')
plt.ylabel('Precision')
plt.title('Query Retrieval Test Accuracy for Multiple Topic Weights')
plt.savefig('./Images/weights_image.png')
plt.show()