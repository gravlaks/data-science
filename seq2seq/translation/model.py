"""
File name: model.py

Creation Date: Sat 31 Jul 2021

Description:

"""

# Python Libraries
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed, Activation,\
        RepeatVector, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

# Local Application Modules
# -----------------------------------------------------------------------------
from preprocessing import get_preprocessed_sentences_and_tokenizer, get_sentence_pairs



def get_model(max_spanish_len, max_english_len, spanish_vocab, english_vocab, embedding_dim):
    print(max_spanish_len)
    print(max_english_len)
    print(spanish_vocab)
    print(embedding_dim)
    input_layer = Input(shape=(max_spanish_len, ))
    embedding = Embedding(input_dim=spanish_vocab, output_dim=embedding_dim,)(input_layer)

    encoder = LSTM(64, return_sequences=False, dropout=0.2)(embedding)
    repeat_encoder = RepeatVector(max_english_len)(encoder)

    decoder = LSTM(64, return_sequences=True, dropout=0.2)(repeat_encoder)
    dense = TimeDistributed(Dense(english_vocab, activation="softmax"))(decoder)

    encoder_decoder_model = Model(input_layer, dense)
    
    encoder_decoder_model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(1e-3),
            metrics=['accuracy']
            )
    return encoder_decoder_model


pairs = get_sentence_pairs()
spa_pad_sentence, eng_pad_sentence,\
            spa_tokenizer, eng_tokenizer = get_preprocessed_sentences_and_tokenizer(pairs)

embedding_dim = 128
model = get_model(len(spa_pad_sentence[0]),len(eng_pad_sentence[0]), len(spa_pad_sentence), len(eng_pad_sentence), embedding_dim)
print(model.summary())

model.fit(spa_pad_sentence, eng_pad_sentence, batch_size = 30, epochs = 200, validation_split=0.2)

def output_to_sentence(output, tokenizer):

    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '<empty>'
    return " ".join([index_to_words[idx] for idx in np.argmax(output, 1)])
index = 14
model.save("models/lstm_simple")
print("The english sentence is: {}".format(pairs[index][0]))
print("The spanish sentence is: {}".format(pairs[index][1]))
print('The predicted sentence is :')
print(output_to_sentence(model.predict(spa_pad_sentence[index:index+1])[0], eng_tokenizer))
print(model.predict(spa_pad_sentence[index:index+1]))
