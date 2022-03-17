#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:06:25 2022

@author: Drew Tatum
"""

### MedLine Collection http://ir.dcs.gla.ac.uk/resources/test_collections/medl/
### Chosen because there is an article title and description 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json 

#### Reading in the files
# Documents
raw_data = {}  # Will hold the raw form of each document
doc_id = 0
first_iteration = True
with open('./TestFiles/ML/MED.ALL') as infile:
    new_doc = []
    x = infile.readlines()
    for line in x:
        txt = line.strip()
        first_two = txt[:2]
        if first_two == '.I':  # Marks a new document
            if first_iteration == False:
                raw_data[doc_id] = ' '.join(new_doc)
                new_doc = []
                doc_id += 1
            else:
                first_iteration = False
                doc_id += 1
        elif len(txt) == 2:
            pass
        else:
            new_doc.append(txt)
            
    raw_data[doc_id] = ' '.join(new_doc)

# Creating a Topic Dictionary (Article title)
# Creating a Info Dictionary (Article description)
topic_dict = {}  # Will hold the article titles for each document
info_dict = {}  # Will hold the article description for each document 
for key in raw_data.keys():
    description = []
    val = raw_data[key]
    sentences = val.split('.')
    for index, sentence in enumerate(sentences):
        if index == 0:  # First sentence is the article topic
            topic_dict[key] = sentence
        else:  # All other sentences are the article description 
            description += [sentence]
    info_dict[key] = ' '.join(description)
    

# Queries
raw_queries = {}  # Will hold each query 
query_id = 0
first_iteration = True
with open('./TestFiles/ML/MED.QRY') as infile:
    new_query = []
    x = infile.readlines()
    for line in x:
        txt = line.strip()
        if len(txt) == 4 or len(txt) == 5:
            if first_iteration == False:
                raw_queries[query_id] = ' '.join(new_query)
                new_query = []
                query_id += 1
            else:
                first_iteration = False
                query_id += 1
        elif len(txt) == 2:
            pass
        else:
            new_query.append(txt)
            
    raw_queries[query_id] = ' '.join(new_query)

### Tokenization according to project model (Using the same process that was peformed for the course catalog project)
### Only changed the lines with tokens_info variable (2 lines) to get each dictionary accordingly to new name 

##### Creating inverted index for Course Information 
info_inverted_index = []

##### Tokenization, Stop Word Removal, and Stemming for Article Description
tokens_info = [word_tokenize(info_dict[key]) for key in info_dict.keys()]  # Tokens for Article Description
# Lowering each token, removing punctuation, removing stop words, and stemming
port_stemmer = PorterStemmer()
for doc_id, tokens_lst in enumerate(tokens_info):
    cleaned_tokens = [token.lower() for token in tokens_lst if token.isalpha() and token not in stopwords.words('english') and len(token) > 1]
    stemmed_tokens = [port_stemmer.stem(token) for token in cleaned_tokens]
    for token in stemmed_tokens:
        info_inverted_index.append([token, doc_id])

info_inverted_index = sorted(info_inverted_index)  # Sorting the inverted index

##### Creating index/dictionary and postings for course description 
info_index = {}
info_postings = {}  # (docID, term frequency)
doc_id_lst = []  # Keep track of doc_id's for postings list
first_iteration = True 
count = 0
prev_docID = None
prev_term = None
for term, docID in info_inverted_index:
    if term not in info_index.keys():  # New term
        info_index[term] = {'Doc Freq': 1, 'Total Freq': 1}
        # Update postings list from prior
        if first_iteration == False:
            doc_id_lst.append([prev_docID, count])
            count = 0
            info_postings[prev_term] = doc_id_lst
            doc_id_lst = []
    else:  # Already added term
        if docID == prev_docID:  # Term is within same document
            info_index[term]['Total Freq'] += 1
        else:  # Term is in a new document
            info_index[term]['Doc Freq'] += 1
            info_index[term]['Total Freq'] += 1
            doc_id_lst.append((prev_docID, count))
            count = 0 
    count += 1
    prev_docID = docID  # Assigning docID as previous docID
    prev_term = term  # Assigning term as previous term
    first_iteration = False
# Adding last postings term 
doc_id_lst.append((prev_docID, count))
info_postings[prev_term] = doc_id_lst     
    
##### Creating inverted index for Article Topic
topic_inverted_index = []

##### Tokenization, Stop Word Removal, and Stemming for Course Topic
tokens_info = [word_tokenize(topic_dict[key]) for key in topic_dict.keys()]  # Tokens for course info
# Lowering each token, removing punctuation, removing stop words, and stemming
port_stemmer = PorterStemmer()
for doc_id, tokens_lst in enumerate(tokens_info):
    cleaned_tokens = [token.lower() for token in tokens_lst if token.isalpha() and token not in stopwords.words('english') and len(token) > 1]
    stemmed_tokens = [port_stemmer.stem(token) for token in cleaned_tokens]
    for token in stemmed_tokens:
        topic_inverted_index.append([token, doc_id])

topic_inverted_index = sorted(topic_inverted_index)  # Sorting the inverted index

##### Creating index/dictionary and postings for course topic
topic_index = {}
topic_postings = {}  # (docID, term frequency)
doc_id_lst = []  # Keep track of doc_id's for postings list
first_iteration = True 
count = 0
for term, docID in topic_inverted_index:
    if term not in topic_index.keys():  # New term
        topic_index[term] = {'Doc Freq': 1, 'Total Freq': 1}
        # Update postings list from prior
        if first_iteration == False:
            doc_id_lst.append([prev_docID, count])
            count = 0
            topic_postings[prev_term] = doc_id_lst
            doc_id_lst = []
    else:  # Already added term
        if docID == prev_docID:  # Term is within same document
            topic_index[term]['Total Freq'] += 1
        else:  # Term is in a new document
            topic_index[term]['Doc Freq'] += 1
            topic_index[term]['Total Freq'] += 1
            doc_id_lst.append((prev_docID, count))
            count = 0 
    count += 1
    prev_docID = docID  # Assigning docID as previous docID
    prev_term = term  # Assigning term as previous term
    first_iteration = False
    
# Adding last postings term 
doc_id_lst.append((prev_docID, count))
topic_postings[prev_term] = doc_id_lst 

##### RECAP
# Have the following 4 dictionaries 
## Two Indexes: info_index, topic_index
## Two Postings: info_postings, topic_postings


##### Saving info index and info postings as json files
with open('./TestFiles/test_info_index.json', 'w') as outfile1:
    json.dump(info_index, outfile1)

with open('./TestFiles/test_info_postings.json', 'w') as outfile2:
    json.dump(info_postings, outfile2)

# Saving topic index, topic postings, 
with open('./TestFiles/test_topic_index.json', 'w') as outfile3:
    json.dump(topic_index, outfile3)

with open('./TestFiles/test_topic_postings.json', 'w') as outfile4:
    json.dump(topic_postings, outfile4)
    