#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:52:11 2022

@author: Drew Tatum
"""

import json 
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import webbrowser  
import pandas as pd 

# Loading info index, info postings, and docID_mapping from indexing
with open('./IndexFiles/info_index.json', 'r') as infile1:
    info_index = json.load(infile1)
    
with open('./IndexFiles/info_postings.json', 'r') as infile2:
    info_postings = json.load(infile2)

with open('./IndexFiles/docID_mapping.json', 'r') as infile3:
    docID_mapping = json.load(infile3)

# Loading topic index, and topic postings
with open('./IndexFiles/topic_index.json', 'r') as infile4:
    topic_index = json.load(infile4)
    
with open('./IndexFiles/topic_postings.json', 'r') as infile5:
    topic_postings = json.load(infile5)

class QueryRetrieval:
    def __init__(self, info_index, info_postings, topic_index, topic_postings, mapping, k, d, topic_weight):
        # Info is the course description while topic is the course name 
        self.info_index = info_index  
        self.info_postings = info_postings
        self.topic_index = topic_index
        self.topic_postings = topic_postings
        self.mapping = mapping 
        self.N = len(mapping)
        self.k = k  # Number of retrieved docs based on cosine similarity 
        self.d = d  # Top d documents returned to user 
        self.topic_weight = topic_weight  # Weight of the topic name keywords similarity (Course info weight is 1-topic_weight)
    
    def RetrieveDocs(self, query):
        """Retrieves top 3 Doc URL's based on User Query"""
        self.query = str(query)
        # Obtaining IDF Weights
        info_idf_weights = self.IDF(info_index) 
        topic_idf_weights = self.IDF(self.topic_index) 
        # Creating vector space model of postings matrix
        info_posting_matrix = self.VectorSpace(self.info_postings)
        topic_posting_matrix = self.VectorSpace(self.topic_postings) 
        # Normalizing vector space model
        normalized_info_matrix = self.TFIDF(info_idf_weights, info_posting_matrix)
        normalized_topic_matrix = self.TFIDF(topic_idf_weights, topic_posting_matrix)
        # Tokenizing query and linguistic processing 
        tokenized_query = self.QueryTokens()
        # Vector representation of the query
        info_vector_query = self.QueryVector(tokenized_query, self.info_postings)
        topic_vector_query = self.QueryVector(tokenized_query, self.topic_postings)
        # Calculating Similarity Values 
        info_sim_list = self.CosineSimilarity(normalized_info_matrix, info_vector_query)
        topic_sim_list = self.CosineSimilarity(normalized_topic_matrix, topic_vector_query)
        # Computing Weighted Similarity Value 
        query_docs = self.DocRetrieval(info_sim_list, topic_sim_list)
        # Opens top d document URLs to user
        for doc_id, sim in query_docs:
            course_URL = self.mapping[str(doc_id)]['URL']
            webbrowser.open(course_URL)
    
    def QueryTokens(self):
        """Tokenizes the Query Based on the Rules for the Document Index and Posting Tokenization"""
        modified_query = []
        tokens_query = word_tokenize(self.query)  # Tokens of the query
        modified_query = []
        port_stemmer = PorterStemmer()
        for token in tokens_query:
            if token.isalpha() and token not in stopwords.words('english'):  # Removing stop words and punctuation 
                modified_query.append(port_stemmer.stem(token))  # Stemming the tokens
        
        return modified_query
    
    def QueryVector(self, token_query, postings):
        """Returns a Vector Representation of the Query Using the Postings List"""
        vector = np.zeros(len(postings))
        for term in token_query:  # Iterating through each query term
            for index, key in enumerate(postings.keys()):  # Iterating through the term dictionary, aka the keys
                if term == key:  # Finding when the query term matches the dictionary
                    vector[index] += 1  # Adding a one to the vector location
        return vector
        
    def IDF(self, index):
        """Returns IDF Weights for the Index"""
        idf_lst = []
        for term in index.keys():
            idf_weight = np.log2(self.N/index[term]['Doc Freq']) # using log2
            idf_lst.append(idf_weight)
        return idf_lst            

    def TFIDF(self, idf_vals, non_normalized_matrix):
        """Calculates the normalized postings list using TF*IDF Weights"""
        non_normalized_matrix = np.transpose(non_normalized_matrix)
        normalized_matrix = []
        for row in range(len(idf_vals)):
            new_val = np.round(idf_vals[row] * non_normalized_matrix[row], 3) # IDF * TF for each term as an array
            normalized_val = np.round(np.sqrt(sum(idf_vals[row]**2 * non_normalized_matrix[row]**2)),3)  # Normalizing the value 
            normalized_matrix.append(new_val/normalized_val)
        normalized_matrix = np.transpose(normalized_matrix)
    
        return normalized_matrix
     
    def VectorSpace(self, postings):
        """Vector Space Representation of the Postings List"""
        # Sparse Empty Matrix
        df = pd.DataFrame(columns = postings.keys(), index=list(range(self.N)))
        df.fillna(value=0, inplace=True)  # Creating sparse matrix of zeros
        
        for term in postings.keys():  # Iterating through each term in postings list (key)
            for occurence in postings[term]:  # Obtaining docID and freq for each term
                doc_num = occurence[0]
                term_freq = occurence[1]
                df.at[doc_num, term] += term_freq  # Updating term frequency 

        return np.array(df)  # Returning numpy array of df

    def CosineSimilarity(self, posting_matrix, vector_query):
        """Calculates the Cosine Similarity between each Document Vector and the Query Vector. Returns Top K Similar Documents"""
        cosine_sim_list = []
        for docID, doc in enumerate(posting_matrix):
            dot_prod = np.dot(doc, vector_query)
            x_norm = 0
            y_norm = 0
            
            for val in doc:
                x_norm += val**2
            
            for val in vector_query:
                y_norm += val**2
            
            if x_norm * y_norm != 0:  # Catch dot prod zero errors where no terms appear
                cosine_similarity_val = round(dot_prod/(np.sqrt(x_norm * y_norm)), 3)        
                cosine_sim_list.append([cosine_similarity_val, docID])
    
        sorted_list = sorted(cosine_sim_list, reverse=True)
        return sorted_list[0:self.k]  

    def DocRetrieval(self, info_similarity, topic_similarity):
        """Returns the Top D Courses to the Items Based on the Weighted Similarity Measure between Topic and Info of the course"""
        info_weight = 1 - self.topic_weight
        topic_weight = self.topic_weight
        
        combined_sim_dic = {}
        
        for similarity in info_similarity: 
            doc_id = similarity[1]
            sim_val = similarity[0] * info_weight
            combined_sim_dic[doc_id] = sim_val
            
        for similarity in topic_similarity:  
            doc_id = similarity[1]
            sim_val = similarity[0] * topic_weight
            if doc_id not in combined_sim_dic.keys():
                combined_sim_dic[doc_id] = sim_val
            else:
                combined_sim_dic[doc_id] += sim_val
        
        return list(sorted(combined_sim_dic.items(), key=lambda item: item[1], reverse=True))[0:self.d]
    
k = 10  # Number of Similar Documents Returned to Weighting System
d = 3  # Number of Documents Displayed to User
topic_weight = .2  # Weight for the course name (Course description weight is 1-topic_weight)

#x = QueryRetrieval(info_index, info_postings, topic_index, topic_postings, docID_mapping, k, d, topic_weight)      
#x.RetrieveDocs('Mining')