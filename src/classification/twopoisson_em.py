#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import sys
import pandas as pd
import numpy as np
sys.path.append("../")

from clustering.twopoisson_em_with_covariates import TwoPoissonEM as TwoPoissonEMCOV
from clustering.twopoisson_em_with_covariates_nn import TwoPoissonEM_NN as TwoPoissonEMCOVNN
from aux import sharing

def _f1(f):
    if f is None: return "None"
    if type(f)==float: return "%.1f" % f
    return f

def _print_z(z):
    for i in range(20):
        if (i+1)*25>z.shape[0]:
            break
        for k in range(z.shape[1]):        
            rows = i*25+np.array(range(25))
            logging.debug(("z%i: " % k) + " ".join(("%.2f" % v) for v in z[rows,k]))
        logging.debug("---")
        
        
def _report_train_results(pem, features, train_ids, level=logging.DEBUG):    
    if "group" not in features.columns: return
    id2true = dict(zip(features["id"], features["group"]))
    true = np.array([id2true[i] for i in train_ids])
    preds = (pem.z.max(1) == pem.z[:,1]).astype(int)
    logging.log(level, "[twopoisson_em][em_fitting] TwoPoissonEMCOV train classification quality:\n%s" % 
                  pd.crosstab(true, preds, rownames=['Actual'], colnames=['Predicted']))

def _report_test_results(features, test_ids, test_preds):
    if "group" not in features.columns: return
    id2true = dict(zip(features["id"], features["group"]))
    true = np.array([id2true[i] for i in test_ids])
    test_preds = np.array([int(np.round(v)) for v in test_preds])
    logging.info("[twopoisson_em][em_fitting] TwoPoissonEMCOV test classification quality:\n%s" % pd.crosstab(true, test_preds, rownames=['Actual'], colnames=['Predicted']))


class TwoPoisson_Classifier:
    """Classifies users via clustering of Poisson processes
        with covariates used to model clusters' priors.
    """    
        
    def __init__(self, 
                 MAX_IT=5, 
                 LL_TOL=10**(-3),
                 multistart_em = True, 
                 badge_name="switch_time",
                 priors_model="logistic"):
        self.MAX_IT = MAX_IT
        self.LL_TOL = LL_TOL
        self.badge_name = badge_name
    
        self.multistart_em = multistart_em    
        self.pem = None #store ll object for debugging
        self.priors_model = TwoPoissonEMCOVNN if priors_model=="nn" else TwoPoissonEMCOV
    
    
    def fitting(self, features, 
                train_users_before, train_users_after, 
                train_x, train_ids):
        logging.info("[twopoisson_em][em_fitting] TwoPoissonEMCOV fitting started.")        
        self.pem = pem = self.priors_model(train_users_before, train_users_after, train_x)                
        _report_train_results(pem, features, train_ids)
        
        prev_ll = None    
        for it in range(self.MAX_IT):            
            pem.maximization()
            pem.expectation()
            
            _report_train_results(pem, features, train_ids)
            ll = pem.LL()
            z1 = pem.positives_proportion()
            
            logging.info("[twopoisson_em][em_fitting][#E=%s,#M=%s] prevLL=%s LL=%s z1=%s" % (it, it, _f1(prev_ll), _f1(ll), _f1(z1)))
            if prev_ll is not None and abs((ll-prev_ll)/prev_ll)<self.LL_TOL:
                logging.info("[twopoisson_em][em_fitting] The algorithm has converged in %i iteratons." % (it+1))
                return pem
            if z1>=0.99 or z1<=0.01:
                logging.warn("[twopoisson_em][em_fitting] algorithm got stuck with one class (z1=%s). interrupting." % _f1(z1))
                return pem
            prev_ll = ll
        
        logging.info("[twopoisson_em][em_fitting] The algorithm has not converged in %i iteratons." % (it+1))
        return pem

    
    def multistart_fitting(self, features, train_users_before, train_users_after, 
                           train_x, train_ids, 
                           NUMSTART=3, MAXREP=30):
        ll_pem = []
        for counter in range(MAXREP*NUMSTART):
            pem = self.fitting(features, train_users_before, train_users_after, train_x, train_ids)
            z1 = pem.positives_proportion()
            if z1<=0.99 and z1>=0.01: 
                ll_pem.append( (pem.LL(), pem) ) 
            if len(ll_pem)>=NUMSTART: break
        
        if counter>=(MAXREP*NUMSTART)-1:
            logging.error("[twopoisson_em][multistart_fitting] STUCK IN THE INFINITE LOOP!")
            sharing.report_kv("ERROR", "[twopoisson_em][multistart_fitting] STUCK IN THE INFINITE LOOP!")
            ll_pem.append( (pem.LL(), pem) ) #return something anyways
            
        ll_pem = sorted(ll_pem)
        logging.debug("[twopoisson_em][multistart_fitting] ll_pem=%s" % ll_pem)
        return ll_pem[-1][1]
    
    
    def prepare_train_users(self, samples, train_ids):
        """Returns two lists of user tuples (count, lifetime)
            representing users before and after they got a badge."""
        selected = set(train_ids)
       
        user2start, user2switch, user2max = {}, {}, {}
        for uid, tp, t in zip(samples["id"], samples["type"], samples["time"]):
            if uid not in selected: continue
            if tp=="start_time":  user2start[uid] = t
            if tp=="max_time":    user2max[uid] = t
            if tp==self.badge_name:    user2switch[uid] = t
            
        user2before, user2after = {}, {}
        for uid, tp, t in zip(samples["id"], samples["type"], samples["time"]):
            if uid not in selected: continue        
            if   t<user2switch[uid] and t>=user2start[uid]: user2before[uid] = user2before.get(uid, 0) + 1 
            elif t<=user2max[uid]:                          user2after[uid]  = user2after.get(uid, 0)  + 1
            
        train_users_before = [(user2before.get(uid, 0), user2switch[uid]-user2start[uid]) for uid in train_ids]       
        train_users_after  = [(user2after.get(uid, 0),  user2max[uid]-user2switch[uid])   for uid in train_ids]  
             
        return train_users_before, train_users_after

    @staticmethod
    def prepare_features_matrices(features, train_ids, test_ids):
        id2pos = dict((uid, pos) for pos, uid in enumerate(features["id"]))
        x = np.asarray( features[features.columns.difference(["group", "id"])] ) #remove id and group info from features
        train_x = x[[id2pos[i] for i in train_ids] ,:]
        test_x  = x[[id2pos[i] for i in test_ids] ,:]
        return train_x, test_x

    def twopoisson_em_classification(self, features, samples, train_ids, test_ids,
                                     id2processes = None):    
        assert id2processes==None, "This option is not supported anymore!"
        #if id2processes is not None:
        #    train_processes     = [id2processes[i] for i in train_ids]
        #    train_users_before  = [(p.get_numactions_before_switch(), p.switch_time()-p.start_time()) for p in train_processes]
        #    train_users_after   = [(p.get_numactions_after_switch(), p.max_time()-p.switch_time()) for p in train_processes]
        #else:
        train_users_before, train_users_after = self.prepare_train_users(samples, train_ids)
        train_x, test_x = self.prepare_features_matrices(features, train_ids, test_ids)
                        
        if self.multistart_em:
            self.pem = pem = self.multistart_fitting(features, train_users_before, train_users_after, train_x, train_ids)
        else:
            self.pem = pem = self.fitting(features, train_users_before, train_users_after, train_x, train_ids)
        
        _report_train_results(pem, features, train_ids, level=logging.INFO)
        logging.info("[twopoisson_em] >>>LL=%.1f" % pem.LL())
    
        test_preds = np.array(pem.predict(test_x))
        _report_test_results(features, test_ids, test_preds)    
        
        return pem.positives(), test_preds
    
    
def twopoisson_em_classification(features, samples, train_ids, test_ids,
                                 id2processes = None, badge_name="switch_time"):    
    
    classifier = TwoPoisson_Classifier(badge_name=badge_name)
    
    return classifier.twopoisson_em_classification(features, samples, train_ids, test_ids,
                                                   id2processes=id2processes)
    
    
    