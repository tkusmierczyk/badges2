#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import copy


def prepare_configurations(params):
    
    TFs = [params.get("TF", 0.5)]       #fraction of Test Users 
    
    Fs = [params.get("F", 0.5)]         #F = Fraction switching
    FTSs = [params.get("FTS", None)]    #FTS = Fraction switching in the test set; if None use F
        
    ishifts    = [params.get("ishift", 1.0)]
    covmshifts = [params.get("covmshift", 1.0)]
    trends = [params.get("trend", 0)]
    
    numtrials = params.get("numtrials", 10)
    
    mode = params.get("mode", 0)
    if mode==0:
        Fs = [0.05, 0.5, 0.95]
        ishifts    = [1.0, 2.0, 0.0, 3.0, 0.5, 1.5, 2.5]
        covmshifts = [1.0, 2.0, 0.0, 3.0, 0.5, 1.5, 2.5]

    #studying impact of differences in intensities and covariates between user groups:
    elif mode==101:
        #Fs         = [0.5]
        ishifts    = [1.0, 2.0, 0.0]
        covmshifts = [1.0, 2.0, 0.0, 3.0, 0.5, 1.5, 2.5]

    elif mode==102:
        #Fs         = [0.5]
        ishifts    = [3.0, 0.5, 1.5, 2.5]
        covmshifts = [1.0, 2.0, 0.0, 3.0, 0.5, 1.5, 2.5]
        
    elif mode=="trends": #trend strength
        trends     = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

    elif mode=="influence_frac": #fraction of impacted users       
        Fs         = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    elif mode=="proportion": #proportion between training and testing users       
        TFs        = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        
    elif mode=="test_influence_frac": #fraction of impacted users in the test set
        #keep fixed proportion of influenced in training but change in testing between 0.1 - 0.9
        FTSs       = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    #testing modes:
    elif mode==-1:
        #Fs = [0.5]
        if "ishift" not in params:      ishifts    = [1.0, 2.0]
        if "covmshift" not in params:   covmshifts = [1.0, 2.0]
        numtrials  = params.get("numtrials", 3)         
        
    elif mode==-101:
        if "F" not in params:           Fs         = [0.5, 0.75]
        if "ishift" not in params:      ishifts    = [1.0, 0.0]
        if "covmshift" not in params:   covmshifts = [1.0, 0.0]
        numtrials  = params.get("numtrials", 1) 
        
    values_summary = "trends=%s TFs=%s Fs=%s FTSs=%s ishifts=%s covmshifts=%s numtrials=%s" % \
                        (trends, TFs, Fs, FTSs, ishifts, covmshifts, numtrials)
    logging.info("[simulation] %s" % values_summary)
    
    ###############################################################################################
    
    def _set(cfg, key, value):
        if value: cfg[key] = value
    
    configurations = []
    for trend in trends:
        for trialno in range(numtrials):
            for ishift in ishifts:
                for covmshift in covmshifts:
                    for F in Fs:
                        for FTS in FTSs:
                            for TF in TFs:
                                cfg = copy.copy(params)
                                
                                cfg["TF"] = TF
                                cfg["F"] = F
                                _set(cfg, "FTS", FTS) #will be assigned only if set (=not None)
                                
                                cfg["trend"] = trend
                                cfg["covmshift"] = covmshift
                                cfg["ishift"] = ishift
                                
                                cfg["trialno"] = trialno
                                cfg["ix"] = len(configurations)
                                cfg["seed"] = params.get("seed", 123*(1+cfg["ix"]))
                                
                                configurations.append(cfg)
    print("%i cfgs: %s" % (len(configurations), values_summary))
    return configurations
