import os
import copy
import pandas as pd
from gensim.models import Word2Vec
from random import shuffle
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import logging
import time
from tqdm import tqdm
import re
import string
import gc; gc.enable()

from nltk.corpus import stopwords                
from nltk.stem.snowball import RussianStemmer
stemmer = RussianStemmer(ignore_stopwords=False)

def clean_text(txt):
    return u" ".join([stemmer.stem(re.sub(r'\b\d+\b', '', wrd)) for wrd in str(txt).lower().strip().split(string.punctuation)
                         if wrd not in stopwords.words('russian')])
  

def process_chunk(df, is_title):
    if is_title:
        df['title'].fillna('unknowntitle', inplace=True)
    else:
        df['description'].fillna('unknowndescription', inplace=True)
        
    return df

def text_generator(is_title):
    files = [
        'train.csv', 'test.csv', 'train_active.csv', 'test_active.csv'
    ]
    
    for f in files:
        print('Processing file:', f)

        if is_title:
            target_col = 'title'
        else:
            target_col = 'description'

        usecols = [target_col]
        for chunk in pd.read_csv(f, usecols=usecols, chunksize=2000000):
            chunk = process_chunk(chunk, is_title)
            
            for s in tqdm(chunk[target_col].values):
                yield text_to_word_sequence(clean_text(s)) 
                    
            del chunk; gc.collect()

def text_generator2(file, is_title, start):
    
    if is_title:
        target_col = 'title'
    else:
        target_col = 'description'

    usecols = [target_col]
    df = process_chunk(pd.read_csv(file, usecols=usecols, nrows= 2000000, skiprows=range(1, start)), is_title)
    return [text_to_word_sequence(clean_text(s)) for s in tqdm(df[target_col].values)]

logging.basicConfig(level=logging.INFO)
'''
def load_text(start):
    print('Loading data...', end='')
    tic = time.time()
    train2 = pd.read_csv('../input/train_active.csv', usecols=use_cols, nrows= 1000000, skiprows=range(1, start))
    toc = time.time()
    print('Done in {:.1f}s'.format(toc-tic))
    train2['text'] = train2['param_1'].str.cat([train2.param_2,train2.param_3,train2.title,train2.description], sep=' ',na_rep='')
    train2.drop(use_cols, axis = 1, inplace=True)
    train2 = train2['text'].values

    train2 = [text_to_word_sequence(text) for text in tqdm(train2)]
    return train2
    
for k in range(15):
    update = False
    if k != 0:
        update = True
    train = load_text(k*1000000+1)
    model.build_vocab(train, update=update)
    model.train(train, total_examples=model.corpus_count, epochs=3)
'''

target='title'
target_dim = 100

model = Word2Vec(size=target_dim, window=5, max_vocab_size=500000, workers=3)

files = ['train.csv', 'test.csv', 'train_active.csv', 'test_active.csv']

update = False
for f in files:
    for k in range(100):
        if f==files[0] and k == 0:
            update = False
            print('Refresh vocab!')
        else:
            update = True
            
        train = text_generator2(f, target=='title', k*2000000+1)
        if len(train) == 0:
            break
          
        model.build_vocab(train, update=update)
        model.train(train, total_examples=model.corpus_count, epochs=8)
        del train; gc.collect()
        
model.save('avito_{}_{}.w2v'.format(target, target_dim))