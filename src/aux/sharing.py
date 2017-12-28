#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Structures used to exchange content/reports between modules."""


VAR = {}

PLOT = False

RESULTS = {}

REPORT = {}

def report_kv(key, value):
    REPORT[key] = value
    

def report_d(dct, label=""):
    for k, v in dct.items():
        report_kv(label+str(k), v)
        
        
def report_reset():
    REPORT = {}
    
    
def report_retrieve():
    return REPORT


