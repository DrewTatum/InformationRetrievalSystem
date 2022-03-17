#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:07:28 2022

@author: Drew Tatum
"""

import sqlite3
import json


##### Loading info index, info postings, and docID_mapping from indexing
with open('./IndexFiles/info_index.json', 'r') as infile1:
    info_index = json.load(infile1)
    
with open('./IndexFiles/info_postings.json', 'r') as infile2:
    info_postings = json.load(infile2)

with open('./IndexFiles/docID_mapping.json', 'r') as infile3:
    docID_mapping = json.load(infile3)

##### Creating Course Database
conn = sqlite3.connect('./DB/course.db')
cur = conn.cursor()

## Creating tables
info_index_table = """CREATE TABLE IF NOT EXISTS info_index(
Term TEXT PRIMARY KEY,
Doc_Freq INT,
Total_Freq INT);
"""

info_postings_table = """CREATE TABLE IF NOT EXISTS info_postings(
Term TEXT,
Doc_Num INT,
Term_Freq INT,
PRIMARY KEY (Term, Doc_Num),
FOREIGN KEY (Term) REFERENCES dictionary (Term)
);
"""

# Executing the tables into the database
cur.execute(info_index_table)
cur.execute(info_postings_table)


def populate_index(index_dic):
    values = []
    for key in index_dic.keys():
        term = key
        doc_freq = index_dic[key]['Doc Freq']
        tot_freq = index_dic[key]['Total Freq']
        values.append(tuple([term, doc_freq, tot_freq]))
    return values

def populate_postings(postings_dic):
    values = []
    for key in postings_dic.keys():
        for value in postings_dic[key]:
            term = key
            doc_num = value[0]
            term_freq = value[1]
            values.append(tuple([term, doc_num, term_freq]))
    return values

# Check to make sure tables are up to date
cur.execute("SELECT COUNT(*) FROM info_index;")
dictionary_table_length = cur.fetchone()

if dictionary_table_length[0] != len(info_index.keys()):  # Sizes don't match
    cur.execute("DROP TABLE IF EXISTS info_index;")
    cur.execute(info_index_table)
    dictionary_values = populate_index(info_index)
    conn.executemany("INSERT INTO info_index VALUES(?,?,?);", dictionary_values) # Populating the table

# Check
cur.execute("SELECT COUNT(*) FROM info_postings;")
postings_table_length = cur.fetchone()

postings_length = 0
for key in info_postings.keys():
    postings_length += len(info_postings[key])

if postings_table_length[0] != postings_length:
    cur.execute("DROP TABLE IF EXISTS info_postings;")
    cur.execute(info_postings_table)
    postings_values = populate_postings(info_postings)
    conn.executemany("INSERT INTO info_postings VALUES(?,?,?);", postings_values) # Populating the table

cur.execute("SELECT * FROM info_postings LIMIT 5;")
post_tbl = cur.fetchall()
cur.execute("SELECT * FROM info_index LIMIT 5;")
index_tbl = cur.fetchall()

print(index_tbl)
print(post_tbl)


conn.commit()
conn.close()

