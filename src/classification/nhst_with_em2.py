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

from multiprocessing import Pool


def perform_nhst_test(params):
    uid, usamples, context = params 
    pvalue, llr_true, _, _, l0, l1, l2 = nhst_testing.badge_change_test(usamples, test=context.test, badge_name=context.badge_name)
    logging.debug("[nhst_classification][%s] testing user id=%4s usamples=%4s => l0=%6s l1=%6s l2=%6s | p=%.3f llr=%.3f " % 
                  (context.test, uid, len(usamples), "%.3f" % l0, "%.3f" % l1, "%.3f" % l2, pvalue, llr_true))        
    return pvalue


class NHST_EM2_Classifier:
    """Classifies users in 2 phases (steps): 
        first, initial NHST classification is performed,
        then, EM2 clustering in covariates' space is used to refine results.
    """

    def __init__(self, 
                 test=nhst_testing.wilks_pvalue,
                 pvalue_threshold=0.05,
                 cpus=1,
                 FPR=0.25, FNR=0.4, K0=1, K1=1,
                 sigma=1.0, kappa=1.0, 
                 badge_name="switch_time",                                 
                 fix_nhst_train_classification = None):
        self.test = test
        self.badge_name = badge_name
        self.pvalue_threshold = pvalue_threshold
        self.cpus = cpus
        
        self.FPR=FPR 
        self.FNR=FNR
        self.K0=K0
        self.K1=K1
        self.sigma=sigma
        self.kappa=kappa                
        
        self.fix_nhst_train_classification = fix_nhst_train_classification
        self.nhst_train_preds = None
        self.nhst_test_preds = None
        self.nhst_elapsed_time = None
        self.clustering_params = None
        
        self.method_prefix = "WT" if self.test==nhst_testing.wilks_pvalue else "BT"

    def _report_intermediate_nhst_classification_results(self, features, train_ids, preds):
        """Preliminary evaluation (if group information available in features)."""
        if "group" not in features.columns: return
        id2true = dict(zip(features["id"], features["group"]))
        
        train_true = np.array([id2true[i] for i in train_ids])
        logging.info("[nhst_classification][%s] p-values train classification:\n%s" % 
                     (self.test, pd.crosstab(train_true, preds, rownames=['Actual'], colnames=['Predicted'])))

    def _report_intermediate_classification_results(self, features, train_ids, test_ids, train_preds, test_preds):
        """Preliminary evaluation (if group information available in features)."""
        if "group" not in features.columns: return
        id2true = dict(zip(features["id"], features["group"]))
        
        train_true, train_preds = np.array([id2true[i] for i in train_ids]), train_preds.round().astype(int)
        logging.info("[nhst_with_em2][nhst_classification] << em2 train classification:\n%s" % pd.crosstab(train_true, train_preds, rownames=['Actual'], colnames=['Predicted']))
        
        test_true, test_preds = np.array([id2true[i] for i in test_ids]), test_preds.round().astype(int)
        logging.info("[nhst_with_em2][nhst_classification] << em2 test classification:\n%s" % pd.crosstab(test_true, test_preds, rownames=['Actual'], colnames=['Predicted']))
    
    def nhst_classification(self, features, samples, train_ids, test_ids):
        start_time = time.time()
        logging.info("[nhst_with_em2][nhst_classification] >> NHST classification started.")
        
        if self.fix_nhst_train_classification is not None: #if fixed 
            train_preds = self.fix_nhst_train_classification  
        else:
            #perform the test for each user    
            configurations = [(uid, samples[samples["id"] == uid], self) for uid in train_ids]
            logging.debug("[nhst_classification] running on %i cpu(s)" % self.cpus)    
            if self.cpus>1:                     
                processes = Pool(self.cpus)
                pvalues = processes.map(perform_nhst_test, configurations)
            else:            
                pvalues = []
                for i, c in enumerate(configurations):
                    if i%(1+len(configurations)//20)==0: 
                        logging.debug("[nhst_classification] %i/%i" % (i+1, len(configurations)))
                    pvalues.append(perform_nhst_test(c))
            train_preds = (np.asarray(pvalues)<self.pvalue_threshold).astype(float) #is significant?
        test_preds = np.ones(len(test_ids))*0.5 #we know nothing about the test data 
        
        self._report_intermediate_nhst_classification_results(features, train_ids, train_preds)
        self.nhst_train_preds, self.nhst_test_preds = train_preds, test_preds                             
        self.nhst_elapsed_time = time.time()-start_time
        return train_preds, test_preds

    @staticmethod
    def prepare_features_matrix(features, train_ids, test_ids):
        #extract and reorder features matrix
        feature_columns = [c for c in features.columns if c not in ["group", "id"]] #remove id and group info from features
        x = np.asarray(features[feature_columns]) 
        id2pos = dict((uid, pos) for pos, uid in enumerate(features["id"]))
        x = x[[id2pos[i] for i in train_ids+test_ids] ,:]        
        return x
    
    def two_step_classification(self, features, samples, train_ids, test_ids):            
        train_preds, _ = self.nhst_classification(features, samples, train_ids, test_ids)
        attracted, not_attracted = sum(train_preds), len(train_preds)-sum(train_preds)    
        logging.debug("[two_step_classification] nhst attracted=%s not_attracted=%s" % (attracted, not_attracted) )
        
        #############################################################################################
        #TODO more than two clusters & Dirchlet priors for K>2
        start_time = time.time()
        logging.info("[two_step_classification] EM2 refining started.")
        assert self.K0==1 and self.K1==1    
    
        #parameters initialization
        group_prior_pi = self.sigma*np.array([[not_attracted*(1-self.FNR),    not_attracted*self.FNR], 
                                              [attracted*self.FPR,            attracted*(1-self.FPR)], 
                                              [1, 1]])      
        user_group_assignment = list(train_preds.round().astype(int))+[2 for _ in test_ids] 
        train_weight = self.kappa*float(len(test_ids))/len(train_ids) if self.kappa and len(test_ids)>0 else 1.0
        weights = np.array([train_weight for _ in train_ids]+[1.0 for _ in test_ids])
        logging.debug("[two_step_classification] group_prior_pi=%s train_weight=%f" % 
                      ("; ".join(map(str, group_prior_pi)), train_weight))      
        
        features_matrix = self.prepare_features_matrix(features, train_ids, test_ids)
        z, self.clustering_params = em2.normal_em(features_matrix, user_group_assignment, group_prior_pi, weights)        
        train_preds, test_preds = z[:,1][0:len(train_ids)], z[:,1][len(train_ids): ]

        self._report_intermediate_classification_results(features, train_ids, test_ids, train_preds, test_preds)            
        self.train_preds, self.test_preds = train_preds, test_preds
        self.elapsed_time = time.time()-start_time
        return train_preds, test_preds
    
    
#backward compatibility version
def nhst_classification(features, samples, train_ids, test_ids,
                                             test=nhst_testing.wilks_pvalue, 
                                             badge_name="switch_time", 
                                             pvalue_threshold=0.05, cpus=1):
    classifier = NHST_EM2_Classifier( test=test,
                                      badge_name=badge_name,
                                      pvalue_threshold=pvalue_threshold,
                                      cpus=cpus)
    return classifier.nhst_classification(features, samples, train_ids, test_ids)
    
            
    
#backward compatibility version    
def two_step_classification(features, samples, train_ids, test_ids,
                            test=nhst_testing.wilks_pvalue,
                            badge_name="switch_time",
                            pvalue_threshold=0.05,
                            FPR=0.25, FNR=0.4, K0=1, K1=1,
                            sigma=1.0, kappa=1.0,
                            cpus=1, train_preds=None):
    
    classifier = NHST_EM2_Classifier( test=test,
                                      badge_name=badge_name,
                                      pvalue_threshold=pvalue_threshold,
                                      cpus=cpus,
                                      FPR=FPR, FNR=FNR, K0=K0, K1=K1,
                                      sigma=sigma, kappa=kappa,                 
                                      fix_nhst_train_classification=train_preds)
    
    train_preds, test_preds = classifier.two_step_classification(features, samples, train_ids, test_ids)
    
    #save nhst intermediate results
    ##if "train_group" in sharing.VAR:
    ##    true = sharing.VAR["train_group"]
    ##    sharing.report_d(evaluation.summary(true, classifier.nhst_train_preds), classifier.method_prefix + "_TR_")                    
    two_step_classification.intermediate_results = {"nhst_preds": classifier.nhst_train_preds, "nhst_train_preds": classifier.nhst_train_preds} 
    sharing.report_kv(classifier.method_prefix+"_TIME", classifier.nhst_elapsed_time)
    
    #save clustering intermediate results 
    clustering_params = classifier.clustering_params
    init_z = clustering_params["init_z"]
    clustering_params["init_train"], clustering_params["init_test"] = init_z[:,1][0:len(train_ids)], init_z[:,1][len(train_ids): ]
    two_step_classification.intermediate_results.update(clustering_params)
    
    return train_preds, test_preds
    
