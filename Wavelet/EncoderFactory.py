from transformers.StaticTransformer import StaticTransformer
from transformers.LastStateTransformer import LastStateTransformer
from transformers.AggregateTransformer import AggregateTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer
import transformers.TextTransformers as ttf
from transformers.WaveletTransformer import WaveletTransformer


def get_encoder(bucket, method,cls_method,case_id_col=None, static_cat_cols=None, static_num_cols=None, dynamic_cat_cols=None, dynamic_num_cols=None, fillna=True, max_events=None, activity_col=None, resource_col=None, timestamp_col=None, scale_model=None, text_transformer_args=None):

    if method == "static":
        return StaticTransformer(bucket=bucket, cls_method=cls_method,case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)

    elif method == "last" or method == "laststate":
        return LastStateTransformer(bucket=bucket, cls_method=cls_method, case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)

    # TODO Add wavelet transformer for encoding data WaveletTransformer
    elif method == "wavelet":
        #laststate = LastStateTransformer(cls_method, case_id_col=case_id_col, cat_cols=dynamic_cat_cols,
        #                             num_cols=dynamic_num_cols,
        #                             fillna=fillna)
        wavelet = WaveletTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                 fillna=fillna)
        return wavelet

    elif method == "agg":
        return AggregateTransformer(bucket=bucket, cls_method=cls_method,case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, boolean=False)

    elif method == "bool":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, boolean=True)
    
    elif method == "index":
        return IndexBasedTransformer(bucket=bucket,cls_method=cls_method,case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                     max_events=max_events, fillna=fillna)
    
    elif method == "lda":
        return ttf.LDATransformer(**text_transformer_args)
    
    elif method == "pv":
        return ttf.PVTransformer(**text_transformer_args)
    
    elif method == "bong":
        return ttf.BoNGTransformer(**text_transformer_args)
    
    elif method == "nb":
        return ttf.NBLogCountRatioTransformer(**text_transformer_args)

    else:
        print("Invalid encoder type")
        return None
