"""
File name: preprocessing.py

Creation Date: Sat 31 Jul 2021

Description:
    
    Fetches data and preprocesses it. The data contains pairs of english and 
    spanish sentences. 


"""

# Python Libraries
# -----------------------------------------------------------------------------
import string
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Local Application Modules
# -----------------------------------------------------------------------------

# creating a DataFrame
    

data_filepath = "../data/spa.txt"


def get_sentence_pairs():

    with open(data_filepath, "r") as f:
        raw = f.read()


    raw = raw.split('\n')
    pairs = [sentence.split('\t') for sentence in raw]

    pairs = pairs[1000:20000]
    print(pairs[0])
    return pairs

def clean_sentences(sentence):
    print(sentence)
    print(type(sentence))
    sentence = sentence.str.lower()
    string_punctuation = string.punctuation +  "¡" + '¿'
    return sentence.str.translate(str.maketrans('', '', string_punctuation))

def tokenize(sentences):
    text_tokenizer = Tokenizer()
    text_tokenizer.fit_on_texts(sentences)

    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer


def get_preprocessed_sentences_and_tokenizer(pairs):
    df = pd.DataFrame(pairs, columns=['english', 'spanish', 'misc'])
    df = df.drop("misc", axis=1)

    df[["english", "spanish"]] = df[["english", "spanish"]].astype(str)
    df[["english", "spanish"]] = df[["english", "spanish"]].apply(clean_sentences)
    print(df.info())
    print(df.head())


    spa_tokenized, spa_tokenizer = tokenize(list(df["spanish"]))
    eng_tokenized, eng_tokenizer = tokenize(list(df["english"]))

    print('Maximum length spanish sentence: {}'.format(len(max(spa_tokenized,key=len))))
    print('Maximum length english sentence: {}'.format(len(max(eng_tokenized,key=len))))


# Check language length
    spanish_vocab = len(spa_tokenizer.word_index) + 1
    english_vocab = len(eng_tokenizer.word_index) + 1
    print("Spanish vocabulary is of {} unique words".format(spanish_vocab))
    print("English vocabulary is of {} unique words".format(english_vocab))

## Padding with 0s

    max_spanish_len = int(len(max(spa_tokenized, key=len)))
    max_english_len = int(len(max(eng_tokenized, key=len)))

    spa_pad_sentence = pad_sequences(spa_tokenized, max_spanish_len, padding="post")
    eng_pad_sentence = pad_sequences(eng_tokenized, max_english_len, padding="post")

    spa_pad_sentence = spa_pad_sentence.reshape(*spa_pad_sentence.shape, 1)
    eng_pad_sentence = eng_pad_sentence.reshape(*eng_pad_sentence.shape, 1)

    return spa_pad_sentence, eng_pad_sentence,\
            spa_tokenizer, eng_tokenizer



