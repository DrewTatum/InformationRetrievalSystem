#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 04:49:57 2022

@author: Drew Tatum
"""

import requests
from bs4 import BeautifulSoup
import re
import json

##### Finding what courses to add to index based on Data Science Computational Methods program
data_science_url = 'https://www.cdm.depaul.edu/academics/Pages/Current/Requirements-MS-In-Data-Science-Computational-Methods.aspx'
url = requests.get(data_science_url)
soup = BeautifulSoup(url.content, 'html.parser')
class_ids = soup.find_all(class_='CDMExtendedCourseInfo')
additional_classes = soup.find_all(class_='manualSimplePopup')
all_ids = class_ids + additional_classes

course_ids = []
course_pattern = re.compile('>(.*)<')

for c_id in all_ids:
    name = re.findall(course_pattern, str(c_id))[0]
    if len(name) < 8 and name not in course_ids:  # len is to not include courses that aren't available anymore since 'no longer offered' appears with course id
        course_ids.append(name)
    
##### Web scraping each individual course
courses = {}
courseID_docID = {}  # CourseID mapping
for index, course in enumerate(course_ids):
    course_num = course[-3:]
    course_sub = course[:3].strip()
    course_url = 'https://www.cdm.depaul.edu/academics/pages/courseinfo.aspx?Subject=' + course_sub + '&CatalogNbr=' + course_num 
    # Update Mapping
    courseID_docID[int(index)] = {'Course ID': course, 'URL': course_url}
    # Each URL now has relevant course number and subject
    url = requests.get(course_url)
    
    # Parsing URL
    soup = BeautifulSoup(url.content, 'html.parser')
    name_results = soup.find_all(class_='PageTitle')
    info_results = soup.find_all('p')
    course_raw_info = info_results[1]
    # Course information and course name patterns
    info_pattern = re.compile('<p>\s+(.*)</p>')
    name_pattern = re.compile('"PageTitle">\s+(\w+)\s+(\d+):\s+(.+)<\/h1>')
    
    course_info = re.findall(info_pattern, str(course_raw_info))[0]
    course_name = re.findall(name_pattern, str(name_results))[0]
   
    course_id = course_name[0] + course_name[1]  # Course ID is the department + 3 number representation for the class
    course_topic = course_name[2]
        
    courses[course_id] = {'Topic': course_topic, 'Info': course_info}
    
    
# Saving json format of dictionary for further linguistic processing
with open('./IndexFiles/courses.json', 'w') as outfile:
    json.dump(courses, outfile)

with open('./IndexFiles/docID_mapping.json', 'w') as outfile2: # Dictionary with docID matching to course ID and url
    json.dump(courseID_docID, outfile2)
    
