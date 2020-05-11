"""
This an extended version of code done by "Irene" after adding wavelet, moreover you can find the original code at below link:

https://github.com/irhete/predictive-monitoring-thesis

This script used to compute the execution time on the basis of wavelet encoding.

"tp run this script:"
        python experiments_performance.py <data set> <bucketing_encoding> <classifier>

        Ex:
            python experiments_performance.py production single_waveletLast  catboost  1

"Author:"
        Mahmoud Kamel Shoush
        mahmoud.shoush@ut.ee
"""

import time
import os
import sys
from sys import argv
import pickle
import csv

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from DatasetManager import DatasetManager
import EncoderFactory
import BucketFactory
import ClassifierFactory

PARAMS_DIR = "cv_results_revision"
RESULTS_DIR = "results_performance"

dataset_ref = argv[1]
method_name = argv[2]
cls_method = argv[3]
n_iter = int(argv[4])

gap = 1

if "prefix_index" in method_name or "prefix_waveletIndex" in method_name:
    bucket_method, cls_encoding, nr_events = method_name.split("_")
    nr_events = int(nr_events)
else:
    bucket_method, cls_encoding = method_name.split("_")
    nr_events = None



if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    #"sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
    "sepsis_cases": ["sepsis_cases_4"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"],
    "wavelet": ["wavelet"],
    "waveletLast": ["wavelet"],
    "waveletAgg":["wavelet"],
    "waveletIndex":["wavelet"]
}

def add_features(df_wavelet, df_last, numberOfFeatures):
    repeat_arr = df_wavelet.iloc[:, 0:numberOfFeatures].values.tolist()  # wavelet
    df_repeated = pd.DataFrame(repeat_arr * int((len(df_wavelet) / len(repeat_arr) + 1)))
    newdf = df_last.join(df_repeated, lsuffix='_left', rsuffix='_right')
    return newdf


datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
    
train_ratio = 0.8
random_state = 22
min_cases_for_training = 1

# create results directory
if not os.path.exists(os.path.join(RESULTS_DIR)):
    os.makedirs(os.path.join(RESULTS_DIR))
    
for dataset_name in datasets:
    
    if bucket_method != "prefix":
        # load optimal params
        optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
        if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
            continue

        with open(optimal_params_filename, "rb") as fin:
            args = pickle.load(fin)
    
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    # split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    overall_class_ratio = dataset_manager.get_class_ratio(train)
    
    # generate test prefix log
    start_test_prefix_generation = time.time()
    # generate prefixes
    if nr_events is not None:
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, nr_events, nr_events)
    else:
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

    test_prefix_generation_time = time.time() - start_test_prefix_generation
            
    offline_total_times = []
    online_event_times = []
    
    for ii in range(n_iter):
        print("Starting iteration %s ..." % ii)
        time_train = 0
        
        # create train prefix log
        start = time.time()
        # generate prefixes
        if nr_events is not None:
            dt_train_prefixes = dataset_manager.generate_prefix_data(train, nr_events, nr_events)
        else:
            dt_train_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

        time_train += time.time() - start
            
        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method': bucket_encoding, 
                         'case_id_col': dataset_manager.case_id_col, 
                         'cat_cols': [dataset_manager.activity_col], 
                         'num_cols': [], 
                         'random_state': random_state}
        if bucket_method == "cluster":
            bucketer_args["n_clusters"] = int(args["n_clusters"])
        cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}
        bucketer = BucketFactory.get_bucketer(bucket_method,cls_method, **bucketer_args)

        start = time.time()
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
        time_train += time.time() - start

        bucket_assignments_test = bucketer.predict(dt_test_prefixes)

        for bucket in set(bucket_assignments_test):
            if bucket_method == "prefix":
                # load optimal params
                optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name, bucket))
                if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
                    continue

                with open(optimal_params_filename, "rb") as fin:
                    args = pickle.load(fin)
            
            #args["n_estimators"] = 500
            
            # select prefixes for the given bucket
            relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_train_cases_bucket)
            train_y = dataset_manager.get_label_numeric(dt_train_bucket)
            test_y = dataset_manager.get_label_numeric(dt_test_bucket)

            if cls_encoding == 'waveletLast' or cls_encoding == 'waveletAgg' or cls_encoding == 'waveletIndex':
                pass
                # initialize pipeline for sequence encoder and classifier
                start = time.time()
                if cls_encoding == 'waveletLast' or cls_encoding == 'waveletAgg' or cls_encoding == 'waveletIndex':
                    if cls_encoding == "waveletLast" or cls_encoding == "waveletAgg" or cls_encoding == "waveletIndex":
                        if cls_encoding == "waveletLast":
                            encoding = "laststate"
                        elif cls_encoding == "waveletAgg":
                            encoding = "agg"
                        else:
                            encoding = "index"

                # initialize pipeline for sequence encoder and classifier
                feature_combiner_last = FeatureUnion(
                    [(method, EncoderFactory.get_encoder(bucket_method, method, cls_method, **cls_encoder_args)) for
                     method
                     in
                     encoding_dict[encoding]],
                    n_jobs=-1)
                feature_combiner_wavelet = FeatureUnion(
                    [(method, EncoderFactory.get_encoder(bucket_method, method, cls_method, **cls_encoder_args)) for
                     method
                     in
                     encoding_dict['wavelet']],
                    n_jobs=-1)


                cls = ClassifierFactory.get_classifier(cls_method, args, random_state, min_cases_for_training,
                                                       overall_class_ratio)

                # fit pipeline
                if cls_method == 'catboost' and bucket_method != 'cluster':
                    with open('outfile' + '_' + cls_method + '_' + cls_encoding, 'rb') as fp:
                        types = pickle.load(fp)
                    newdf = pd.DataFrame(types).reset_index()
                    types = newdf[0]
                    os.remove('outfile' + '_' + cls_method + '_' + cls_encoding)


                    if dt_train_bucket.shape[0] == 1 or dt_test_bucket.shape[0] == 1:
                        continue
                    else:
                        # pass

                        dt_train_bucket_encoded_last = feature_combiner_last.transform(dt_train_bucket)
                        dt_train_bucket_encoded_wavelet = feature_combiner_wavelet.transform(dt_train_bucket)
                        dt_train_bucket_encoded = add_features(pd.DataFrame(dt_train_bucket_encoded_wavelet), \
                                                               pd.DataFrame(dt_train_bucket_encoded_last),
                                                               numberOfFeatures=10)
                        dt_train_bucket_encoded.columns = list(range(dt_train_bucket_encoded.shape[1]))

                        dt_test_bucket_encoded_last = feature_combiner_last.transform((dt_test_bucket))
                        dt_test_bucket_encoded_wavelet = feature_combiner_wavelet.transform((dt_test_bucket))
                        dt_test_bucket_encoded = add_features(pd.DataFrame(dt_test_bucket_encoded_wavelet), \
                                                              pd.DataFrame(dt_test_bucket_encoded_last),
                                                              numberOfFeatures=10)
                        dt_test_bucket_encoded.columns = list(range(dt_test_bucket_encoded.shape[1]))

                        i = 0
                        for col in dt_train_bucket_encoded:
                            dt_train_bucket_encoded[col] = dt_train_bucket_encoded[col].astype(types[i])
                            dt_test_bucket_encoded[col] = dt_test_bucket_encoded[col].astype(types[i])
                            i += 1
                        cls.fit(dt_train_bucket_encoded, train_y,
                                list(dt_train_bucket_encoded.select_dtypes(include=['object', 'category']).columns))

                else:

                    if dt_train_bucket.shape[0] == 1:
                        continue
                    else:
                        dt_train_bucket_encoded_last = feature_combiner_last.transform(dt_train_bucket)
                        dt_train_bucket_encoded_wavelet = feature_combiner_wavelet.transform(dt_train_bucket)

                        pd.DataFrame(dt_train_bucket_encoded_last).to_csv("folds_train_last.csv", sep=";",
                                                                          index=False)
                        pd.DataFrame(dt_train_bucket_encoded_wavelet).to_csv("folds_train_wavelet.csv", sep=";",
                                                                             index=False)

                        dt_train_bucket_encoded_last = pd.read_csv("folds_train_last.csv", sep=";")
                        os.remove("folds_train_last.csv")
                        dt_train_bucket_encoded_wavelet = pd.read_csv("folds_train_wavelet.csv", sep=";")
                        os.remove("folds_train_wavelet.csv")
                        import time

                        time.sleep(5)

                        dt_train_bucket_encoded = add_features(dt_train_bucket_encoded_wavelet,
                                                               dt_train_bucket_encoded_last, numberOfFeatures=10)
                        dt_train_bucket_encoded.columns = list(range(dt_train_bucket_encoded.shape[1]))

                        del dt_train_bucket_encoded_last, dt_train_bucket_encoded_wavelet

                    if dt_test_bucket.shape[0] == 1:
                        continue
                    else:
                        dt_test_bucket_encoded_last = feature_combiner_last.transform(dt_test_bucket)
                        dt_test_bucket_encoded_wavelet = feature_combiner_wavelet.transform(dt_test_bucket)

                        pd.DataFrame(dt_test_bucket_encoded_last).to_csv("folds_test_last.csv", sep=";",
                                                                         index=False)
                        pd.DataFrame(dt_test_bucket_encoded_wavelet).to_csv("folds_test_wavelet.csv", sep=";",
                                                                            index=False)

                        # read encoded data
                        dt_test_bucket_encoded_last = pd.read_csv("folds_test_last.csv", sep=";")
                        os.remove("folds_test_last.csv")
                        dt_test_bucket_encoded_wavelet = pd.read_csv("folds_test_wavelet.csv", sep=";")
                        os.remove("folds_test_wavelet.csv")

                        dt_test_bucket_encoded = add_features(dt_test_bucket_encoded_wavelet,
                                                              dt_test_bucket_encoded_last,
                                                              numberOfFeatures=10)
                        dt_test_bucket_encoded.columns = list(range(dt_test_bucket_encoded.shape[1]))

                        del dt_test_bucket_encoded_last, dt_test_bucket_encoded_wavelet

                    if dt_test_bucket_encoded.shape[1] < dt_train_bucket_encoded.shape[1]:
                        dt_train_bucket_encoded = dt_train_bucket_encoded.iloc[:, :dt_test_bucket_encoded.shape[1]]
                    else:
                        dt_test_bucket_encoded = dt_test_bucket_encoded.iloc[:, :dt_train_bucket_encoded.shape[1]]

                    if cls_method == 'svm' or cls_method == 'logit' or cls_method == 'rf':

                        dt_train_bucket_encoded = dt_train_bucket_encoded.fillna(0)
                        print(dt_train_bucket_encoded.shape)
                        dt_test_bucket_encoded = dt_test_bucket_encoded.fillna(0)
                        print(dt_test_bucket_encoded.shape)
                        print(len(train_y))
                        train_y = train_y[0:dt_train_bucket_encoded.shape[0]]
                        print(len(train_y))
                        if len(train_y) < dt_train_bucket_encoded.shape[0]:
                            dt_train_bucket_encoded = dt_train_bucket_encoded.iloc[:len(train_y), :]
                        elif len(train_y) > dt_train_bucket_encoded.shape[0]:
                            train_y = train_y[0:dt_train_bucket_encoded.shape[0]]
                        else:
                            pass
                        cls.fit(dt_train_bucket_encoded, train_y)


                    else:
                        print(dt_test_bucket_encoded.shape)
                        print(dt_test_bucket_encoded.shape)
                        dt_train_bucket_encoded = dt_train_bucket_encoded.values
                        dt_test_bucket_encoded = dt_test_bucket_encoded.values
                        cls.fit(dt_train_bucket_encoded, train_y)


                time_train += time.time() - start

                test_all_grouped = dt_test_bucket.groupby('orig_case_id')
                for _, group in test_all_grouped:
                    start = time.time()
                    _ = bucketer.predict(group)
                    if group.shape[0] == 1:
                        continue
                    else:
                        group_last = feature_combiner_last.fit_transform(group)
                        group_wavelet = feature_combiner_wavelet.fit_transform(group)
                        group = add_features(pd.DataFrame(group_wavelet),
                                             pd.DataFrame(group_last),numberOfFeatures=10)
                        group.columns = list(range(group.shape[1]))
                        print(group.shape)
                        group.fillna(0, inplace=True)
                        #group = np.array(group)
                        dd = dt_train_bucket_encoded.shape[1] - group.shape[1]
                        if dd < 0:
                            dd = dd * (-1)
                        print(dd)
                        group = np.hstack((group, np.zeros((group.shape[0], dd))))
                        pred = cls.predict_proba(group)
                        time_test = (time.time() - start) / len(group)
                        online_event_times.append(time_test)
                offline_total_times.append(time_train)


            else:
                pass
                # initialize pipeline for sequence encoder and classifier
                start = time.time()
                feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(bucket_method, method,cls_method, **cls_encoder_args)) for method in methods])
                cls = ClassifierFactory.get_classifier(cls_method, args, random_state, min_cases_for_training,
                                                       overall_class_ratio)

                if cls_method == "svm" or cls_method == "logit":
                    pipeline = Pipeline([('encoder', feature_combiner), ('scaler', MinMaxScaler()), ('cls', cls)])
                else:
                    pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

                # fit pipeline
                if cls_method == 'catboost' and bucket_method!='cluster':
                    with open('outfile' + '_' + cls_method + '_' + cls_encoding, 'rb') as fp:
                        types = pickle.load(fp)
                    newdf = pd.DataFrame(types).reset_index()
                    types = newdf[0]
                    os.remove('outfile' + '_' + cls_method + '_' + cls_encoding)

                    dt_train_bucket_encoded = feature_combiner.transform(dt_train_bucket)
                    dt_test_bucket_encoded = feature_combiner.transform((dt_test_bucket))

                    dt_train_bucket_encoded = pd.DataFrame(dt_train_bucket_encoded,
                                                           columns=feature_combiner.get_feature_names())
                    dt_test_bucket_encoded = pd.DataFrame(dt_test_bucket_encoded,
                                                          columns=feature_combiner.get_feature_names())
                    i = 0
                    for col in dt_train_bucket_encoded:
                        dt_train_bucket_encoded[col] = dt_train_bucket_encoded[col].astype(types[i])
                        dt_test_bucket_encoded[col] = dt_test_bucket_encoded[col].astype(types[i])
                        i += 1
                    cls.fit(dt_train_bucket_encoded, train_y,
                            list(dt_train_bucket_encoded.select_dtypes(include=['object', 'category']).columns))

                else:
                    if dt_train_bucket.shape[0] == 1:
                        continue
                    else:

                        if cls_method == 'svm' or cls_method == 'logit' or cls_method == 'rf':
                            pass
                            dt_train_bucket_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
                            dt_train_bucket_encoded.fillna(0, inplace=True)
                            dt_test_bucket_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
                            dt_test_bucket_encoded.fillna(0, inplace=True)
                            pipeline.fit(dt_train_bucket, train_y)
                        else:
                            pipeline.fit(dt_train_bucket, train_y)


                # fit pipeline
                #pipeline.fit(dt_train_bucket, train_y)
                time_train += time.time() - start

                # predict separately for each prefix
                test_all_grouped = dt_test_bucket.groupby(dataset_manager.case_id_col)
                for _, group in test_all_grouped:
                    start = time.time()
                    _ = bucketer.predict(group)
                    pred = pipeline.predict_proba(group)
                    time_test = (time.time() - start) / len(group)
                    online_event_times.append(time_test)
            offline_total_times.append(time_train)

    offline_total_times = np.array(offline_total_times)
    online_event_times = np.array(online_event_times)
    
    # write results
    outfile = os.path.join(RESULTS_DIR, "results_performance_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        spamwriter.writerow(["dataset", "bucket_enc", "text_method_enc", "cls", "metric", "score"])
        spamwriter.writerow([dataset_name, method_name, "no_text", cls_method, "offline_total_avg", offline_total_times.mean()])
        spamwriter.writerow([dataset_name, method_name, "no_text", cls_method, "offline_total_std", offline_total_times.std()])
        spamwriter.writerow([dataset_name, method_name, "no_text", cls_method, "online_event_avg", online_event_times.mean()])
        spamwriter.writerow([dataset_name, method_name, "no_text", cls_method, "online_event_std", online_event_times.std()])
