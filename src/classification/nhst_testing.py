#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Testing badges impact on users by applying NHST to temporal traces."""

import numpy as np
from scipy.stats import chi2
import logging

#import sys
#sys.path.append("../")


ATTENTION_HORIZON = 1000 #limits time window around badge
EPS = 10**(-18) #time margin from badge for bootstrap
NUM_SAMPLES = 300 #num runs of bootstrap
INF = float("inf")


def llr0(start, end, actions, badge, 
         verbose=False):
    actions = np.asarray(actions)
    
    t0 = (end-start)
    t1 = (badge-start)
    t2 = (end-badge)
    
    count0 = sum((actions>start) & (actions<end))
    count1 = sum((actions<badge) & (actions>start))
    count2 = sum((actions>badge) & (actions<end))
    
    l0 = count0 / t0
    ll0 = count0*np.log(l0) - l0*t0
    
    l1 = count1 / t1
    ll1 = count1*np.log(l1) - l1*t1 if count1>0 else 0.0
    
    l2 = count2 / t2
    ll2 = count2*np.log(l2) - l2*t2 if count2>0 else 0.0
    
    llr = ll2+ll1 - ll0
    
    if (verbose): 
        logging.info("start=%f end=%f badge=%f" % (start, end, badge))
        logging.info("na0=%i t0=%f na1=%i t1=%f na=%i t2=%f" % (count0, t0, count1, t1, count2, t2))
        logging.info("l0=%g ll0=%g  l1=%g l2=%g ll1=%g llr=%g" % (l0, ll0, l1, l2, ll1+ll2, llr))
    return llr, ll1, ll2, l0, l1, l2


def llr(start, end, actions, badge, verbose=False): 
    return llr0(start, end, actions, badge, verbose=verbose)[0]
  
  
def wilks_pvalue(start, end, actions, badge, verbose=False):
    actions = sorted(actions)
    llr_true, ll1, ll2, l0, l1, l2 = llr0(start, end, actions, badge, verbose=verbose)
    pvalue = 1-chi2.cdf(2*(llr_true), 1)
    return pvalue, llr_true, ll1, ll2, l0, l1, l2
    
    
def bootstrap_pvalue(start, end, actions, badge, 
                     n=NUM_SAMPLES, margin_events=1, attention_horizon=ATTENTION_HORIZON,
                     verbose=False):
    start, end = max(badge-attention_horizon, start), min(badge+attention_horizon, end)
    llr_true, ll1, ll2, l0, l1, l2 = llr0(start, end, actions, badge, verbose=verbose)
    actions = np.asarray(sorted(actions))
    actions = actions[(actions>=start) & (actions<=end)]
    
    if len(actions)<=margin_events:
        logging.info("WARNING: %i actions available with start=%f end=%f" % (len(actions), start, end))
        return 1-chi2.cdf(2*(llr_true), 1), llr_true, ll1, ll2, l0, l1, l2

    start1, end1 = actions[margin_events-1]+EPS, actions[-margin_events]-EPS
    llr_virtual = [ (llr(badge, end, actions, virtual_badge) if virtual_badge>badge else llr(start, badge, actions, virtual_badge))
                    for virtual_badge in np.random.uniform(low=start1, high=end1, size=n)]
    #print(llr_true, llr_virtual)
    bpvalue = float(sum(llr_true<np.asarray(llr_virtual))) / len(llr_virtual)
    return bpvalue, llr_true, ll1, ll2, l0, l1, l2
    
    
def badge_change_test(samples, 
                 test = wilks_pvalue,
                 badge_name="switch_time"):
    start = float(samples[samples["type"]=="start_time"]["time"].iloc[0])   
    end = float(samples[samples["type"]=="max_time"]["time"].iloc[0])
    badge = float(samples[samples["type"]==badge_name]["time"].iloc[0])
    actions = list(samples[samples["type"]=="action"]["time"])
    return test(start, end, actions, badge)


def llr01(start, end, actions, alpha, badge, verbose=False):
    """@TODO: WORK IN PROGRESS"""
    actions = np.asarray(actions)
    
    last_before = max(actions[actions<badge])
    first_after = min(list(actions[actions>badge])+[INF])
    actions_after = first_after<INF
    survival = min(first_after, end)-last_before
    
    count0 = sum( ((actions>start) & (actions<=last_before)) | ((actions>first_after) & (actions<end)) )  
    t0 = (last_before-start) + max(0, (end-first_after))

    l0 = count0 / t0
    ll0 = count0*np.log(l0)-l0*t0 + int(actions_after)*alpha-alpha*survival
        
    
    t1 = (last_before-start)
    t2 = max(0, (end-first_after))
    
    count1 = sum((actions>start) & (actions<=last_before))
    count2 = sum((actions>first_after) & (actions<end))
    
    l1 = count1 / t1
    ll1 = count1*np.log(l1) - l1*t1
    
    l2 = count2 / t2
    ll2 = count2*np.log(l2) - l2*t2 if l2>0 else 0.0
    
    llr = ll2+ll1 - ll0
    
    if (verbose): 
        logging.info("start=%f end=%f badge=%f" % (start, end, badge))
        logging.info("na0=%i t0=%f na1=%i t1=%f na=%i t2=%f" % (count0, t0, count1, t1, count2, t2))
        logging.info("l0=%g ll0=%g  l1=%g l2=%g ll1=%g llr=%g" % (l0, ll0, l1, l2, ll1+ll2, llr))
    return llr, ll1, ll2, l0, l1, l2

