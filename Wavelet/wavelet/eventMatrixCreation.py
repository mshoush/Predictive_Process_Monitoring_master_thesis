import math
import numpy as np
from numpy.linalg import inv
from collections import Counter
import operator
from functools import reduce

import numpy as np
from scipy import stats
import pprint
import operator
import collections
from scipy.stats import chisquare, chi2_contingency, t
from scipy.special import expit
from sklearn.metrics.cluster import contingency_matrix
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import gc
import copy
import tqdm

# Implementing the feature:value representation for sparse dataset
# Binarization and Vectorization
import tqdm

from wavelet.waveletComputeTransform import Seq2Vec


def FeatureValue_Representation(log,id1):
    obj1 = Seq2Vec()  # class to get vec from seq
    log_tot = obj1.Log_Normalization(log)

    log_featureValue_dic = {}

    # for trace in log:
    for i in tqdm.tqdm(range(len(log_tot))):
        if i <= (len(log) - 1):
            trace = log_tot[i]
            obj = Seq2Vec()  # Creating an object for the trace
            obj.unique_Events_log = obj1.unique_Events_log  # Giving the unique name of event log
            obj.Start(trace)  # Starting Sequence to vector
            # We only select those key, foe which the list is not totally zero
            nonzero_keys = [key for key in obj.dic_WaveletCfVector if
                            np.sum(np.absolute(obj.dic_WaveletCfVector[key])) != 0]
            #log1_id = list(id1.keys())  # list(fdist1.keys())
            log_featureValue_dic[list(id1.keys())[i]] = dict((k, obj.dic_WaveletCfVector[k]) for k in nonzero_keys)


        else:
            pass
    return log_featureValue_dic


# Creating a dictionary of feature-value. See the definition of the function
# Binarization and Vectorization.
# log_featureValue_dic = FeatureValue_Representation(log1)


# print(log_featureValue_dic)


################################################################################
# This function ranks the events in terms of discrimination (Fisher score)
# we don't need it, it will return the unique events.
def Feature_Ranking(log_featureValue_dic):
    # Finding unique events
    events_unique = set([e for trace in log_featureValue_dic for e in log_featureValue_dic[trace]])
    return events_unique


# events_unique = Feature_Ranking(log_featureValue_dic=log_featureValue_dic)  # [('d', 0.178), ('f', 0.17), ('e', 0.0)]


# len(events_unique)

################################################################################
# This function given a set of events create two matrices (related to event log1 and log2)
def Event_Matrxi_Creation(event_list, log_featureValue_dic,id1):
    # dimension of each element vector
    #t_id = list(log_featureValue_dic.keys())[0]
    #el = list(log_featureValue_dic[t_id].keys())[0]
    #event_dim = len(log_featureValue_dic[t_id][el])

    # Iterate over the first event log
    event_matrix1 = []
    for trace in log_featureValue_dic:
        # Fisrt finding how many times it happened
        freq = len(id1[trace].split(","))
        temp = []
        for e in event_list:
            if e in log_featureValue_dic[trace]:
                temp += log_featureValue_dic[trace][e]

        [event_matrix1.append(t) for t in [temp] * freq]

    return event_matrix1


