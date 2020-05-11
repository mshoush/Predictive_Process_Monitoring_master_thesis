from sklearn.base import TransformerMixin
import pandas as pd
from time import time
from nltk.util import ngrams
import os
import glob
import warnings
import pandas as pd
import numpy as np


def Reading_log_all(X):
    # print("Reading All")

    ################################################################################
    # get different cases (Case Id) from each data file(log)
    def get_case_name(X):
        # case_id_col = 'orig_case_id'
        case_name = X['orig_case_id'].unique()
        return case_name  # return unique cases from the log

    #################################################################################
    # based on the case name (Case ID) we'll get the trace (seq of events of the same case)
    # return all traces from any data file (log)
    def get_log_open_cases(case_name, X):
        log = []
        [log.append(list(X[X['orig_case_id'] == case]['open_cases'])) for case in case_name]

        return log  # getting the trace for each case from the log

    ##################################################################################

    case_name = get_case_name(X)
    # print(case_name)
    log_temp = get_log_open_cases(case_name, X)

    # finding unique traces
    total_trace = []
    dictionary_log = dict()

    k = 0
    for i in range(len(log_temp)):
        if log_temp[i] not in total_trace:
            total_trace.append(log_temp[i])
            # This dictionary is only for tracking which traces are related each uniqe trace
            dictionary_log[k] = str(case_name[i])
            k += 1
        elif log_temp[i] in total_trace:
            trace_index = total_trace.index(log_temp[i])
            dictionary_log[trace_index] = dictionary_log[trace_index] + ',' + str(case_name[i])

    def Ngram_Compute(total_trace, k=2):
        temp = []
        [temp.append(list(ngrams(trace, k))) for trace in total_trace]

        return temp

    return Ngram_Compute(total_trace, k=2), dictionary_log  # return all traces from any log,
