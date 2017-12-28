#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg') 

from multiprocessing import Pool

import json
import time
#import pandas as pd
import numpy as np
import argparse
import os
import logging
import random
import traceback

import sys
sys.path.append("../")

from aux import parsing, sharing

from classification import nhst_testing
from classification import nhst_with_em2, twopoisson_em
from classification.logistic import logistic_regression
from classification import evaluation


from synthetic.sample_data import data_generator
from synthetic.config import prepare_configurations

LOG_DBGINFO = 15
    


def simulation(params):
    #DBG CODE: params = {}; logging.getLogger().setLevel(0)

    seed = params.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)    
    logging.info("[simulation][>>>] PARAMS = %s" % params)
    
    logging.info("[simulation] DATA GENERATION")
    train_ids, test_ids, features, samples, processes, group = data_generator(params)        
    train_group, test_group = group[train_ids], group[test_ids]

    _prfx = lambda dct, prefix="": dict((prefix+str(k), v) for k, v in dct.items())
    sigma = params.get("sigma", 1.0)
    results = {}

    ###########################################################################

    #############################################
    logging.info("[simulation] CLASSIFICATION: twopoisson_em_classification")
    start_time = time.time()
    #train_preds, test_preds = twopoisson_em.twopoisson_em_classification(features, samples, train_ids, test_ids, badge_name="switch_time")#dict(zip(ids, processes)),
    poisson = twopoisson_em.TwoPoisson_Classifier(badge_name="switch_time", 
                                                  priors_model=params.get("twopoisson_priors", "logistic"),
                                                  multistart_em=params.get("multistart_em", True))
    train_preds, test_preds = poisson.twopoisson_em_classification(features, samples, train_ids, test_ids)
    
    results.update(_prfx(evaluation.summary(train_group, train_preds), "2P_TR_"))
    results.update(_prfx(evaluation.summary(test_group, test_preds), "2P_TS_"))
    results["2P_TIME"] = time.time()-start_time
    

    #############################################
    logging.info("[simulation] CLASSIFICATION: two_step_classification with theoretic")
    start_time = time.time()
    nhst_em2_theoretic = nhst_with_em2.NHST_EM2_Classifier(test=nhst_testing.wilks_pvalue,
                                                           badge_name="switch_time",
                                                           pvalue_threshold=0.05,
                                                           sigma=sigma)
    train_preds, test_preds = nhst_em2_theoretic.two_step_classification(features, samples, train_ids, test_ids)

    results.update(_prfx(evaluation.summary(train_group, train_preds), "2SW_TR_"))
    results.update(_prfx(evaluation.summary(test_group,  test_preds), "2SW_TS_"))
    results.update(_prfx(evaluation.summary(train_group, nhst_em2_theoretic.nhst_train_preds), "WT_TR_"))
    results.update(_prfx(evaluation.summary(test_group,  nhst_em2_theoretic.nhst_test_preds), "WT_TS_"))
    results["2SW_TIME"] = time.time()-start_time
    results["WT_TIME"]  = nhst_em2_theoretic.nhst_elapsed_time

    #############################################
    logging.info("[simulation] CLASSIFICATION: two_step_classification with bootstrap")
    start_time = time.time()
    nhst_em2_bootstrap = nhst_with_em2.NHST_EM2_Classifier(test=nhst_testing.bootstrap_pvalue,
                                                           badge_name="switch_time",
                                                           pvalue_threshold=0.05,
                                                           sigma=sigma)
    train_preds, test_preds = nhst_em2_bootstrap.two_step_classification(features, samples, train_ids, test_ids)
    sb_bootstrap_nhst_preds = nhst_em2_bootstrap.nhst_train_preds #nhst_with_em2.two_step_classification.intermediate_results["nhst_train_preds"]
    sb_bootstrap_preds = train_preds
    
    results.update(_prfx(evaluation.summary(train_group, train_preds),"2SB_TR_"))
    results.update(_prfx(evaluation.summary(test_group,  test_preds),"2SB_TS_"))
    results.update(_prfx(evaluation.summary(train_group, nhst_em2_bootstrap.nhst_train_preds), "BT_TR_"))
    results.update(_prfx(evaluation.summary(test_group,  nhst_em2_bootstrap.nhst_test_preds), "BT_TS_"))
    results["2SB_TIME"] = time.time()-start_time    
    results["BT_TIME"]  = nhst_em2_bootstrap.nhst_elapsed_time

    #############################################
    logging.info("[simulation] CLASSIFICATION: logistic_regression on bootstrap-NHST")
    start_time = time.time()
    train_preds, test_preds = logistic_regression(features, train_ids, test_ids, sb_bootstrap_nhst_preds)
    
    results.update(_prfx(evaluation.summary(train_group, train_preds),"LR0_TR_"))
    results.update(_prfx(evaluation.summary(test_group, test_preds),"LR0_TS_"))
    results["LR0_TIME"] = time.time()-start_time

    #############################################
    logging.info("[simulation] CLASSIFICATION: logistic_regression on 2-phase-bootstrap-NHST")
    start_time = time.time()
    train_preds, test_preds = logistic_regression(features, train_ids, test_ids, sb_bootstrap_preds)
    
    results.update(_prfx(evaluation.summary(train_group, train_preds),"LR1_TR_"))
    results.update(_prfx(evaluation.summary(test_group, test_preds),"LR1_TS_"))
    results["LR1_TIME"] = time.time()-start_time

    ###########################################################################

    results.update(params)
    results.update(sharing.report_retrieve())
    #logging.debug("[simulation] RESULTS = %s" % results)
    logging.info("[simulation][>>>] RESULTS: %s" %
                  " ".join(["%s=%s" % (k,v) for k,v in sorted(results.items())]))
    json.dump(dict((parsing.tcast(k), parsing.tcast(v)) for k,v in results.items()), 
              fp=open("/tmp/simulation%s_ix%s.json" % (params.get("experiment_id", os.getpid()),
                                                        params.get("ix", time.time())), "w"))
    return results     


def simulation_safe(params):
    try:
        return simulation(params)
    except Exception as e:
        params["ERROR"] = str(e)
        msg = "%s\nPARAMS:\n%s\nREPORT:\n%s\nMSG: %s\nTRACEBACK: %s" % \
                (time.strftime("%Y-%m-%d %H:%M"), params, sharing.report_retrieve(), e, traceback.format_exc())
        logging.error(("[simulation] ERROR: %s" % msg).replace("\n", "\n[simulation] ERROR:"))
        f = open("ERROR_pid%i_%i.log" % (os.getpid(), random.randint(0,100000)), "w")
        f.write(msg)
        f.close()
        if logging.getLogger("output").getEffectiveLevel()<=logging.DEBUG: raise  
            #import pdb; pdb.set_trace() #run debugger        
    return params 
        


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


def prepare_params(args):
    params = parsing.parse_dictionary(args.params)
    try:    verbose = args.verbose
    except: verbose = False    
    try:    debug = args.debug
    except: debug = False    
        
    fmt = '[%(process)4d][%(asctime)s][%(levelname)-5s][%(module)s:%(lineno)d/%(funcName)s] %(message)s'
    if args.output is not None:
        if args.logfile:            
            if not debug and not verbose:
                errfile = args.output+".err.log"
                sys.stderr = open(errfile, 'w')  
                print("logging errs to %s" % (errfile))                        
            logfile = "%s.log" % (args.output)
            logging.basicConfig(filename=logfile, level=logging.DEBUG, format=fmt)
            print("logging to %s" % (logfile))
        else:
            logging.basicConfig(format=fmt)
    else:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
                
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARN)    
    if verbose or debug:
        print_handler = logging.StreamHandler()
        print_handler.setLevel(level)
        logging.getLogger().addHandler(print_handler)
    logging.getLogger("output").setLevel(level)
        
    if args.output is None:
        args.output = "%s_%i" % (os.path.basename(__file__), os.getpid())
    params["verbose"] = verbose or debug
    params["debug"] = debug
    params["experiment_id"] = os.getpid()
    return params


def print_progress(i, numactions, start_time=None):
    #sys.stdout.write('\rPROGRESS {0:%}'.format(float(i+1)/len(configurations)))
    if start_time is None:
        if print_progress.start_time is None:
            print_progress.start_time = time.time()
    else:
        print_progress.start_time = start_time
    start_time = print_progress.start_time
    
    if i==0:
        sys.stdout.write('\rPROGRESS: 0.00%')   
        sys.stdout.flush()     
        return

    progress = float(i) / numactions
    elapsed = (time.time() - start_time) / 60.0
    speed = progress / elapsed
    left =  (float(numactions - i) / numactions) /speed
    sys.stdout.write('\rPROGRESS: %.2f%%, ELAPSED: %.2fmin, REMAINING: %.2fmin, SPEED: %.2f%%/min         ' % 
                     (100.0*progress, elapsed, left, 100.0*speed))
    sys.stdout.flush()
print_progress.start_time = None


def main():    

    parser = argparse.ArgumentParser(description="""Synthetic experiments simulator.""")
    
    parser.add_argument("-v", "--verbose", dest='verbose', help="print additional info", 
                        required=False, default=False, action='store_true')
    parser.add_argument("-d", "--debug", dest='debug', help="print additional debug info", 
                        required=False, default=False, action='store_true')
     
    parser.add_argument("-o", "--output", dest='output', 
                        help="output files prefix", 
                        required=False, 
                        default="%s" % (os.path.basename(__file__)))
                        #default="%s_%i" % (os.path.basename(__file__), os.getpid()))
    parser.add_argument("-l", "--logfile", dest='logfile', 
                        help="don't create log file", 
                        required=False, default=True, action="store_false")
    parser.add_argument("-c", "--cpu", dest='cpu', 
                        help="num processes to use", 
                        required=False, type=int, default=1)    
      
    parser.add_argument("-p", "--params", dest='params', 
                        help="comma-separated params: option=value", 
                        required=False, default="")
            
    args = parser.parse_args(sys.argv[1: ])    
    if args.debug: args.verbose = True
    
    params = prepare_params(args)    
    
    #########################################################################
    
    configurations = prepare_configurations(params)
    logging.info("[simulation] %i configurations to simulate on %i cpus" % (len(configurations), args.cpu))
    #configurations = [params]       
    results = []
    print_progress(0, len(configurations))
    if args.cpu>1:         
        processes = Pool(args.cpu)
        #results = processes.map(simulation_safe, configurations)
        for i, result in enumerate(processes.imap_unordered(simulation_safe, configurations)):
            print_progress(i+1, len(configurations))
            results.append(result)
    else:
        for i, c in enumerate(configurations):
            results.append(simulation_safe(c))
            print_progress(i+1, len(configurations))
    sys.stdout.write("\n")        
        
    #########################################################################
    
    print("[simulation] saving results to %s" % (args.output+".json"))
    results = [dict((parsing.tcast(k), parsing.tcast(v)) for k,v in r.items()) for r in results]
    json.dump(results, fp=open(args.output+".json", "w"))
    logging.debug("[simulation] "+args.output+".json: "+json.dumps(results))

    #########################################################################
    #########################################################################

if __name__=="__main__":
    main()
    
    
    