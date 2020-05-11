"""
This an extended version of code done by "Irene" after adding wavelet, moreover you can find the original code at below link:

https://github.com/irhete/predictive-monitoring-thesis

This script used to tune model and select the best hyper-parameters for each classification model on the
basis of wavelet encoding.

"tp run this script:"
        python experiments_optimize_params.py <data set> <bucketing_encoding> <classifier> <nr_iteration>

        Ex:
            python experiments_optimize_params.py production single_waveletLast  catboost 10

"Author:"
        Mahmoud Kamel Shoush
        mahmoud.shoush@ut.ee
"""

import time
import os
import shutil
from sys import argv
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope

from DatasetManager import DatasetManager
import EncoderFactory
import ClassifierFactory


def add_features(df_wavelet, df_last, numberOfFeatures):
    repeat_arr = df_wavelet.iloc[:, 0:numberOfFeatures].values.tolist()  # wavelet
    df_repeated = pd.DataFrame(repeat_arr * int((len(df_wavelet) / len(repeat_arr) + 1)))
    newdf = df_last.join(df_repeated)
    return newdf


def get_types(data):
    types = []
    [types.append(str(data[col].dtype)) for col in data.columns]
    return data.dtypes

def create_and_evaluate_model(args):
    global trial_nr, all_results
    trial_nr += 1
    
    print("Trial %s out of %s" % (trial_nr, n_iter))
    
    start = time.time()
    score = 0

    for cv_iter in range(n_splits):

        if cls_encoding == "waveletLast" or cls_encoding == "waveletAgg" or cls_encoding == "waveletIndex":
            # read encoded data
            dt_train_last = pd.read_csv(os.path.join(folds_dir, "fold%s_train_last.csv" % cv_iter), sep=";")
            dt_test_last = pd.read_csv(os.path.join(folds_dir, "fold%s_test_last.csv" % cv_iter), sep=";")

            dt_train_wavelet = pd.read_csv(os.path.join(folds_dir, "fold%s_train_wavelet.csv" % cv_iter), sep=";")
            dt_test_wavelet = pd.read_csv(os.path.join(folds_dir, "fold%s_test_wavelet.csv" % cv_iter), sep=";")

            dt_train = add_features(dt_train_wavelet, dt_train_last,numberOfFeatures=10)
            dt_train.columns = list(range(dt_train.shape[1]))
            dt_test = add_features(dt_test_wavelet, dt_test_last, numberOfFeatures=10)
            dt_test.columns = list(range(dt_test.shape[1]))




            with open(os.path.join(folds_dir, "fold%s_train_y.csv" % cv_iter), "rb") as fin:
                train_y = np.array(pickle.load(fin))
            with open(os.path.join(folds_dir, "fold%s_test_y.csv" % cv_iter), "rb") as fin:
                test_y = np.array(pickle.load(fin))

            # fit classifier and predict
            cls = ClassifierFactory.get_classifier(cls_method, args, random_state, min_cases_for_training,
                                                   class_ratios[cv_iter])

            # print(set(get_types(dt_train)))
            if cls_method == 'catboost':
                with open('outfile' + '_' + cls_method + '_' + cls_encoding, 'wb') as fp:
                    pickle.dump(get_types(dt_train), fp)

                cls.fit(dt_train, train_y, list(dt_train.select_dtypes(include=['object', 'category']).columns))
            else:

                if cls_method == 'svm' or cls_method == 'logit' or cls_method == 'rf':
                    pass
                    dt_train.replace([np.inf, -np.inf], np.nan, inplace=True)
                    dt_train.fillna(0, inplace=True)
                    dt_test.replace([np.inf, -np.inf], np.nan, inplace=True)
                    dt_test.fillna(0, inplace=True)
                    cls.fit(dt_train,train_y)

                else:
                    cls.fit(dt_train, train_y)
            preds = cls.predict_proba(dt_test)

            if len(set(test_y)) >= 2:
                score += roc_auc_score(test_y, preds)
        else:

            # read encoded data
            dt_train = pd.read_csv(os.path.join(folds_dir, "fold%s_train.csv" % cv_iter), sep=";")
            #print(dt_train.info())
            dt_test = pd.read_csv(os.path.join(folds_dir, "fold%s_test.csv" % cv_iter), sep=";")

            with open(os.path.join(folds_dir, "fold%s_train_y.csv" % cv_iter), "rb") as fin:
                train_y = np.array(pickle.load(fin))
            with open(os.path.join(folds_dir, "fold%s_test_y.csv" % cv_iter), "rb") as fin:
                test_y = np.array(pickle.load(fin))

            # fit classifier and predict
            cls = ClassifierFactory.get_classifier(cls_method, args, random_state, min_cases_for_training,
                                                   class_ratios[cv_iter])
            #print(set(get_types(dt_train)))
            if cls_method=='catboost':

                cls.fit(dt_train, train_y, list(dt_train.select_dtypes(include=['object', 'category']).columns))
            else:
                if cls_method == 'svm' or cls_method == 'logit' or cls_method == 'rf':
                    pass
                    dt_train.replace([np.inf, -np.inf], np.nan, inplace=True)
                    dt_train.fillna(0, inplace=True)
                    dt_test.replace([np.inf, -np.inf], np.nan, inplace=True)
                    dt_test.fillna(0, inplace=True)
                    # dt_test = dt_test.astype((np.float))
                    cls.fit(dt_train, train_y)
                else:
                    cls.fit(dt_train, train_y)

                #cls.fit(dt_train, train_y)

            preds = cls.predict_proba(dt_test)

            if len(set(test_y)) >= 2:
                score += roc_auc_score(test_y, preds)
    
    # save current trial results
    for k, v in args.items():
        all_results.append((trial_nr, k, v, -1, score / n_splits))

    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


dataset_ref = argv[1]
method_name = argv[2]
cls_method = argv[3]
n_iter = int(argv[4])

train_ratio = 0.8
n_splits = 3
random_state = 22
min_cases_for_training = 1

if n_splits == 1:
    PARAMS_DIR = "val_results_unstructured"
else:
    PARAMS_DIR = "cv_results_revision"

# create directory
if not os.path.exists(os.path.join(PARAMS_DIR)):
    os.makedirs(os.path.join(PARAMS_DIR))

if "prefix_index" in method_name or "prefix_waveletIndex" in method_name:
    bucket_method, cls_encoding, nr_events = method_name.split("_")
    nr_events = int(nr_events)
else:
    bucket_method, cls_encoding = method_name.split("_")
    nr_events = None

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

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
    
for dataset_name in datasets:
    
    folds_dir = "folds_%s_%s_%s" % (dataset_name, cls_method, method_name)
    if not os.path.exists(os.path.join(folds_dir)):
        os.makedirs(os.path.join(folds_dir))
    
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    
    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    # split into training and test
    train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    del data
    
    # prepare chunks for CV
    class_ratios = []
    cv_iter = 0
    if n_splits == 1:
        if dataset_ref in ["github"]:
            train, _ = dataset_manager.split_data(train, train_ratio=0.15/train_ratio, split="random", seed=22)
            # train will be 0.1 of original data and val 0.05
            train_chunk, test_chunk = dataset_manager.split_val(train, val_ratio=0.33, split="random", seed=22)
        else:
            train_chunk, test_chunk = dataset_manager.split_val(train, 0.2, split="random", seed=22)
        
        class_ratios.append(dataset_manager.get_class_ratio(train_chunk))

        # generate prefixes
        if nr_events is not None:
            dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, nr_events, nr_events)
            dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, nr_events, nr_events)
        else:
            dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, min_prefix_length, max_prefix_length)
            dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length)

        # encode data for classifier
        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(bucket_method, method, cls_method, **cls_encoder_args)) for method in methods])
        # if cls_method == "svm" or cls_method == "logit":
        #     feature_combiner = Pipeline([('encoder', feature_combiner), ('scaler', MinMaxScaler())])

        dt_train_encoded = feature_combiner.fit_transform(dt_train_prefixes)

        pd.DataFrame(dt_train_encoded).to_csv(os.path.join(folds_dir, "fold%s_train.csv" % cv_iter), sep=";", index=False)
        del dt_train_encoded

        dt_test_encoded = feature_combiner.transform(dt_test_prefixes)
        pd.DataFrame(dt_test_encoded).to_csv(os.path.join(folds_dir, "fold%s_test.csv" % cv_iter), sep=";", index=False)
        del dt_test_encoded

        # labels
        train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
        with open(os.path.join(folds_dir, "fold%s_train_y.csv" % cv_iter), "wb") as fout:
            pickle.dump(train_y, fout)

        test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
        with open(os.path.join(folds_dir, "fold%s_test_y.csv" % cv_iter), "wb") as fout:
            pickle.dump(test_y, fout)

    else:
        if cls_encoding == 'waveletLast' or cls_encoding == 'waveletAgg' or cls_encoding == 'waveletIndex':
            if cls_encoding == "waveletLast" or cls_encoding == "waveletAgg" or cls_encoding == "waveletIndex":
                if cls_encoding == "waveletLast":
                    encoding = "laststate"
                elif cls_encoding == "waveletAgg":
                    encoding = "agg"
                else:
                    encoding = "index"
            for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
                class_ratios.append(dataset_manager.get_class_ratio(train_chunk))

                # generate prefixes
                if nr_events is not None:
                    dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, nr_events, nr_events)
                    dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, nr_events, nr_events)
                else:
                    dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, min_prefix_length,
                                                                             max_prefix_length)
                    dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, min_prefix_length,
                                                                            max_prefix_length)



                feature_combiner_last = FeatureUnion(
                    [(method, EncoderFactory.get_encoder(bucket_method, method, cls_method, **cls_encoder_args)) for method in
                     encoding_dict[encoding]],
                    n_jobs=-1)
                feature_combiner_wavelet = FeatureUnion(
                    [(method, EncoderFactory.get_encoder(bucket_method,method, cls_method, **cls_encoder_args)) for method in
                     encoding_dict['wavelet']],
                    n_jobs=-1)



                dt_train_encoded_last = feature_combiner_last.fit_transform(dt_train_prefixes)
                dt_train_encoded_wavelet = feature_combiner_wavelet.fit_transform(dt_train_prefixes)



                pd.DataFrame(dt_train_encoded_last).to_csv(os.path.join(folds_dir, "fold%s_train_last.csv" % cv_iter), sep=";",
                                                      index=False)
                pd.DataFrame(dt_train_encoded_wavelet).to_csv(os.path.join(folds_dir, "fold%s_train_wavelet.csv" % cv_iter), sep=";",
                                                      index=False)
                del dt_train_encoded_last, dt_train_encoded_wavelet

                dt_test_encoded_last = feature_combiner_last.transform(dt_test_prefixes)
                dt_test_encoded_wavelet = feature_combiner_wavelet.transform(dt_test_prefixes)

                pd.DataFrame(dt_test_encoded_last).to_csv(os.path.join(folds_dir, "fold%s_test_last.csv" % cv_iter), sep=";",
                                                     index=False)
                pd.DataFrame(dt_test_encoded_wavelet).to_csv(os.path.join(folds_dir, "fold%s_test_wavelet.csv" % cv_iter), sep=";",
                                                     index=False)
                del dt_test_encoded_last, dt_test_encoded_wavelet

                # labels
                train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
                with open(os.path.join(folds_dir, "fold%s_train_y.csv" % cv_iter), "wb") as fout:
                    pickle.dump(train_y, fout)

                test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
                with open(os.path.join(folds_dir, "fold%s_test_y.csv" % cv_iter), "wb") as fout:
                    pickle.dump(test_y, fout)

                cv_iter += 1
        else:
            #pass
            for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
                class_ratios.append(dataset_manager.get_class_ratio(train_chunk))

                # generate prefixes
                if nr_events is not None:
                    dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, nr_events, nr_events)
                    dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, nr_events, nr_events)
                else:
                    dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, min_prefix_length, max_prefix_length)
                    dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length)

                # encode data for classifier
                feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(bucket_method, method,cls_method, **cls_encoder_args)) for method in methods])
                if cls_method == "svm" or cls_method == "logit":
                    feature_combiner = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler())])

                dt_train_encoded = feature_combiner.fit_transform(dt_train_prefixes)


                pd.DataFrame(dt_train_encoded).to_csv(os.path.join(folds_dir, "fold%s_train.csv" % cv_iter), sep=";", index=False)
                del dt_train_encoded

                dt_test_encoded = feature_combiner.transform(dt_test_prefixes)
                pd.DataFrame(dt_test_encoded).to_csv(os.path.join(folds_dir, "fold%s_test.csv" % cv_iter), sep=";", index=False)
                del dt_test_encoded

                # labels
                train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
                with open(os.path.join(folds_dir, "fold%s_train_y.csv" % cv_iter), "wb") as fout:
                    pickle.dump(train_y, fout)

                test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
                with open(os.path.join(folds_dir, "fold%s_test_y.csv" % cv_iter), "wb") as fout:
                    pickle.dump(test_y, fout)

                cv_iter += 1

    del train
        
    # set up search space
    if cls_method == 'catboost' and cls_method!='cluster':
        space = {
            'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.8),
            'one_hot_max_size': scope.int(hp.quniform('one_hot_max_size', 4, 255, 1)),
            'max_depth': scope.int(hyperopt.hp.quniform('max_depth', 6, 16, 1)),
            'colsample_bylevel': hyperopt.hp.uniform('colsample_bylevel', 0.5, 1.0),
            'bagging_temperature': hyperopt.hp.uniform('bagging_temperature', 0.0, 100),
            'random_strength': hyperopt.hp.uniform('random_strength', 0.0, 100),
            'scale_pos_weight': hyperopt.hp.uniform('scale_pos_weight', 1.0, 16.0),
            'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
            'n_clusters': scope.int(hp.quniform('n_clusters', 2, 6, 1)),
            'n_estimators': hp.choice('n_estimators', [500, 1000])
            # change 16.0 to n_negative / n_poistive
        }



    elif cls_method == "rf":
        space = {'max_features': hp.uniform('max_features', 0, 1),
                 'n_estimators': hp.choice('n_estimators', [500, 1000]),
                 'n_clusters': scope.int(hp.quniform('n_clusters', 2, 6, 1)),
                 }
        
    elif cls_method == "xgboost":
        space = {'learning_rate': hp.uniform("learning_rate", 0, 1),
                 'subsample': hp.uniform("subsample", 0.5, 1),
                 'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                 'n_estimators': hp.choice('n_estimators', [500, 1000]),
                 'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                 'n_clusters': scope.int(hp.quniform('n_clusters', 2, 6, 1)),
                 'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}
        
    elif cls_method == "logit":
        space = {'C': hp.uniform('C', -15, 15),
                 'n_clusters': scope.int(hp.quniform('n_clusters', 2, 6, 1)),
                 }
        
    elif cls_method == "svm":
        space = {'C': hp.uniform('C', -15, 15),
                 'n_clusters': scope.int(hp.quniform('n_clusters', 2, 6, 1)),
                 'gamma': hp.uniform('gamma', -15, 15)}
        
    # optimize parameters
    trial_nr = 0
    trials = Trials()
    all_results = []
    best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)

    # extract the best parameters
    best_params = hyperopt.space_eval(space, best)
    
    # write to file
    outfile = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)
        
    dt_results = pd.DataFrame(all_results, columns=["iter", "param", "value", "nr_events", "score"])
    dt_results["dataset"] = dataset_name
    dt_results["cls"] = cls_method
    dt_results["method"] = method_name
    
    outfile = os.path.join(PARAMS_DIR, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    dt_results.to_csv(outfile, sep=";", index=False)

    shutil.rmtree(folds_dir)