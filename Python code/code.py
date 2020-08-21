# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 18:51:42 2020

@author: Lenovo
"""

import os
import numpy as np
import pandas as pd

import re
from tqdm import tqdm

import collections

from sklearn.cluster import KMeans

from nltk.stem import WordNetLemmatizer  # For Lemmetization of words
from nltk.corpus import stopwords  # Load list of stopwords
from nltk import word_tokenize # Convert paragraph in tokens

import pickle
import sys

from gensim.models import word2vec # For represent words in vectors
import gensim

# Read given data-set using pandas
os.chdir("E:\\IIT Kanpur\\Summer_Internship\\Precily\\Precily Assessment")
text_data = pd.read_csv("Text_Similarity_Dataset.csv")
print("Shape of text_data : ", text_data.shape)
text_data.head(3)

text_data.isnull().sum() # Check if text data have any null values

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# Combining all the above stundents 

preprocessed_text1 = []

# tqdm is for printing the status bar

import nltk
nltk.download('stopwords')

for sentance in tqdm(text_data['text1'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text1.append(sent.lower().strip())
    

# Merging preprocessed_text1 in text_data

text_data['text1'] = preprocessed_text1
text_data.head(3)

# Combining all the above stundents 
from tqdm import tqdm
preprocessed_text2 = []

# tqdm is for printing the status bar
for sentance in tqdm(text_data['text2'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
   
    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text2.append(sent.lower().strip())
    
# Merging preprocessed_text2 in text_data

text_data['text2'] = preprocessed_text2

text_data.head(3)

def word_tokenizer(text):
            #tokenizes and stems the text
            tokens = word_tokenize(text)
            lemmatizer = WordNetLemmatizer() 
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
            return tokens
        
# Load pre_trained Google News Vectors after download file

wordmodelfile = "GoogleNews-vectors-negative300.bin.gz"
wordmodel = gensim.models.KeyedVectors.load_word2vec_format(wordmodelfile, binary=True)

# This code check if word in text1 & text2 present in our google news vectors vocabalry.
# if not it removes that word and if present it compares similarity score between
# text1 and text2 words


similarity = [] # List for store similarity score

nltk.download('punkt')
nltk.download('wordnet')

for ind in text_data.index:
    
        s1 = text_data['text1'][ind]
        s2 = text_data['text2'][ind]
        
        if s1==s2:
                 similarity.append(1.0) # 1 means highly similar
                
        else:   

            s1words = word_tokenizer(s1)
            s2words = word_tokenizer(s2)
            
           
            
            vocab = wordmodel.vocab #the vocabulary considered in the word embeddings
            
            if len(s1words and s2words)==0:
                    similarity.append(0.0)

            else:
                
                for word in s1words.copy(): #remove sentence words not found in the vocab
                    if (word not in vocab):
                           
                            
                            s1words.remove(word)
                        
                    
                for word in s2words.copy(): #idem

                    if (word not in vocab):
                           
                            s2words.remove(word)
                            
                            
                similarity.append((wordmodel.n_similarity(s1words, s2words))) # as it is given 1 means highly dissimilar & 0 means highly similar
                
# Get Unique_ID and similarity

final_score = pd.DataFrame({'Unique_ID':text_data.Unique_ID,
                     'Similarity_score':similarity})
final_score.head(3)

# SAVE DF as CSV file 

final_score.to_csv('final_score.csv',index=False)
