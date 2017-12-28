#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import sys
import pandas as pd
import numpy as np
import time
sys.path.append("../")

from aux import sharing
from classification import nhst_testing
from clustering import em2
from classification import evaluation


from sklearn.linear_model import LogisticRegression



def logistic_regression(features, train_ids, test_ids, train_preds):
    """Learns classes (=train_preds) from features of train_ids
        and predicts them for features of test_ids.
    """
    train_preds = np.asarray(train_preds>0.5, dtype=int)
    
    #extract and reorder features matrix
    feature_columns = [c for c in features.columns if c not in ["group", "id"]] #remove id and group info from features
    x = np.asarray(features[feature_columns]) 
    id2pos = dict((uid, pos) for pos, uid in enumerate(features["id"]))
    
    train_x = x[[id2pos[i] for i in train_ids] ,:]
    test_x = x[[id2pos[i] for i in test_ids] ,:]

    lr = LogisticRegression()
    lr.fit(train_x, train_preds)
    
    train_preds = lr.predict_proba(train_x)[:,1]
    test_preds = lr.predict_proba(test_x)[:,1]

    #preliminary evaluation (if group information available)
    if "group" in features.columns:
        id2true = dict(zip(features["id"], features["group"]))
        train_true, train_preds1 = np.array([id2true[i] for i in train_ids]), np.asarray(train_preds>0.5, dtype=int)
        logging.info("[simulate] logistic_regression:\n%s" % pd.crosstab(train_true, train_preds1, rownames=['Actual'], colnames=['Predicted']))                     
        test_true, test_preds1 = np.array([id2true[i] for i in test_ids]), np.asarray(test_preds>0.5, dtype=int)
        logging.info("[simulate] logistic_regression:\n%s" % pd.crosstab(test_true, test_preds1, rownames=['Actual'], colnames=['Predicted']))                     
         
    return train_preds, test_preds


    
    
    