#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 04:49:57 2022

@author: Drew Tatum
"""

import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Loading courses json file from web crawler
with open('./IndexFiles/courses.json', 'r') as infile:
    courses = json.load(infile)

##### Creating inverted index for Course Information 
info_inverted_index = []

##### Tokenization, Stop Word Removal, and Stemming for Course Information 
tokens_info = [word_tokenize(courses[key]['Info']) for key in courses.keys()]  # Tokens for course info
# Lowering each token, removing punctuation, removing stop words, and stemming
port_stemmer = PorterStemmer()
for doc_id, tokens_lst in enumerate(tokens_info):
    cleaned_tokens = [token.lower() for token in tokens_lst if token.isalpha() and token not in stopwords.words('english') and len(token) > 1]
    stemmed_tokens = [port_stemmer.stem(token) for token in cleaned_tokens]
    for token in stemmed_tokens:
        info_inverted_index.append([token, doc_id])

info_inverted_index = sorted(info_inverted_index)  # Sorting the inverted index

##### Creating index/dictionary and postings for course info
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
    

##### Saving info index and info postings as json files
with open('./IndexFiles/info_index.json', 'w') as outfile1:
    json.dump(info_index, outfile1)

with open('./IndexFiles/info_postings.json', 'w') as outfile2:
    json.dump(info_postings, outfile2)
    
    
##### Creating inverted index for Course Topic
topic_inverted_index = []

##### Tokenization, Stop Word Removal, and Stemming for Course Topic
tokens_info = [word_tokenize(courses[key]['Topic']) for key in courses.keys()]  # Tokens for course info
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

# Saving topic index, topic postings, 
with open('./IndexFiles/topic_index.json', 'w') as outfile3:
    json.dump(topic_index, outfile3)

with open('./IndexFiles/topic_postings.json', 'w') as outfile4:
    json.dump(topic_postings, outfile4)
