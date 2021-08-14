import numpy as np
import tensorflow
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import plot_model
from numpy.random import seed
from read_data import read_data_from_csv
from prepare_data import transform_data, get_dictionary_map

seed(1)
tensorflow.random.set_seed(2)

def get_input_dim():
    data = read_data_from_csv()
    data_group = transform_data(data)
    tag2idx, idx2tag = get_dictionary_map(data, 'tag')
    input_dim = len(list(set(data['Word'].to_list())))+1
    output_dim = 64
    input_length = max([len(s) for s in data_group['Word_idx'].tolist()])
    n_tags = len(tag2idx)
    return input_dim, output_dim, input_length, n_tags

def get_lstm_model():
    model = Sequential()
    input_dim, output_dim, input_length, n_tags = get_input_dim()
    # Add Embedding layer
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))

    # Add bidirectional LSTM
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode = 'concat'))

    # Add LSTM
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

    # Add timeDistributed Layer
    model.add(TimeDistributed(Dense(n_tags, activation="relu")))

    #Optimiser 
    # adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def train_model(X, y, model):
    loss = list()
    for i in range(25):
        # fit model for one epoch on this sequence
        hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss