#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Measuring classification quality."""

import numpy as np
from sklearn.metrics import auc, roc_curve


def confusion_matrix(true, preds, threshold=0.5):
    #true = np.asarray(np.round((true)), dtype=int)
    #preds = np.asarray(np.round((preds)), dtype=int)
    true  = np.asarray((np.asarray(true)>threshold), dtype=int)
    preds = np.asarray((np.asarray(preds)<=threshold), dtype=int)
    TP = sum((true == 1) & (preds == 1))
    FP = sum((true == 0) & (preds == 1))
    TN = sum((true == 0) & (preds == 0))
    FN = sum((true == 1) & (preds == 0))
    return TP, FP, TN, FN


def AUC(true, preds):
    assert len(true)==len(preds)
    if len(true)==0: return None
    #xs, ys = [], []
    #for threshold in sorted(set(preds)):
    #    TP, FP, TN, FN = confusion_matrix(true, preds, threshold)
    #    FPR = float(FP)/(FP+TN)
    #    TPR = float(TP)/(TP+FN)
    #    xs.append(FPR)
    #    ys.append(TPR)
    true = np.asarray(true, dtype=int)
    preds = np.asarray(preds, dtype=float)
    xs, ys, _ = roc_curve(true, preds, pos_label=1)
    return auc(xs, ys)
        

def summary(true, preds):
    true, preds = np.asarray(true), np.asarray(preds)
    results = {}
    
    TP, FP, TN, FN = confusion_matrix(true, preds)
    results.update({"EVAL_TP": TP, "EVAL_FP": FP, "EVAL_TN": TN, "EVAL_FN": FN})
    
    results["AUC"] = AUC(true, preds)
    
    results["EVAL_RMSE"] = np.sqrt(sum((true-preds)*(true-preds))/len(true)) if len(true)>0 else None
    
    true0, true1   = true[true==0],  true[true==1]
    preds0, preds1 = preds[true==0], preds[true==1]
    results["EVAL_RMSE0"] = np.sqrt(sum((true0-preds0)*(true0-preds0))/len(true0)) if len(true0)>0 else None
    results["EVAL_RMSE1"] = np.sqrt(sum((true1-preds1)*(true1-preds1))/len(true1)) if len(true1)>0 else None

    return results

