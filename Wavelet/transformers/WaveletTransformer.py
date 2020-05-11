from sklearn.base import TransformerMixin
import pandas as pd
from time import time
from nltk.util import ngrams
import os
import glob
import warnings
import pandas as pd
import numpy as np
from wavelet.Reading_log_all import Reading_log_all

from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
random_state = 22

from wavelet.eventMatrixCreation import FeatureValue_Representation, Feature_Ranking, Event_Matrxi_Creation


class WaveletTransformer(TransformerMixin):

    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col  # case id
        self.cat_cols = cat_cols  # categorical cols
        self.num_cols = num_cols  # numeric cols
        self.fillna = fillna  # fillna

        self.columns = None

        self.fit_time = 0
        self.transform_time = 0





    def fit(self, X, y=None):
        return self

    ###### Using NN to encode feature vector for wavelet method
    def encode_data(self, data, enc_dim=10):
        # Normalise
        scaler = MinMaxScaler()

        data_scaled = scaler.fit_transform(data)

        input_dim = data.shape[1]  # 8
        encoding_dim = enc_dim

        # Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders
        input_layer = Input(shape=(input_dim,))
        encoder_layer_1 = Dense(6, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder_layer_2 = Dense(4, activation="tanh")(encoder_layer_1)
        encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)

        # Crear encoder model
        encoder = Model(inputs=input_layer, outputs=encoder_layer_3)
        encoded_data = pd.DataFrame(encoder.predict(data_scaled))
        encoded_data.columns = list(range(enc_dim))
        #print(f"\nencoded_data.columns:\n{encoded_data.shape}\n")
        # print(encode_data.head())

        return encoded_data


    def transform(self, X, y=None):

        start = time()
        log1, id1 = Reading_log_all(X)  # return all traces from any log,
        # taking much time
        log_featureValue_dic = FeatureValue_Representation(log1,id1)
        events_unique = Feature_Ranking(log_featureValue_dic)  # [('d', 0.178), ('f', 0.17), ('e', 0.0)]


        # TODO: check dimension for lastsatte method and the dimension for wavelet method.
        # List of lists into numpy array
        # https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
        ev_mat = Event_Matrxi_Creation(events_unique, log_featureValue_dic,id1)
        event_matrix1 = ev_mat
        #event_matrix1 = [x for x in Event_Matrxi_Creation(events_unique, log_featureValue_dic,id1) if x != []]
        #print(f"\nlen(event_matrix1):\n{len(event_matrix1)}\n")
        y = np.array([xi + [0.0000001] * (max(map(len, event_matrix1)) - len(xi)) for xi in event_matrix1]).astype(float)
        #print(f"\nevent_matrix1 4")

        #y = y.astype(float)
        #print(f"\nevent_matrix1 5")

        #print(f"\ny  before: {y[0]}\n")
        #print('\n',y.shape,'\n')
        if y.shape[0] == 0:
            pass
        else:
            #y = self.encode_data(y, int(y.shape[1]/40))
            y = self.encode_data(y)
        #print('\nafter_encoding', y.shape, '\n')
        self.transform_time = time() - start
        return y

    def get_feature_names(self):
        return self.columns

