#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import json
import seaborn as sns; sns.set()

sys.path.append("../")


CONTROL_COLS = ["trend", "F", "FTR", "FTS", "TF", "ishift", "covmshift", "trialno", "ix"]
VALUE_COLS = ["2P_TR_AUC", "2P_TS_AUC", 
              "2SW_TR_AUC", "2SW_TS_AUC", 
              "2SB_TR_AUC", "2SB_TS_AUC",
              "BT_TS_AUC", "BT_TR_AUC",
              "WT_TS_AUC", "WT_TR_AUC"]
#VALUE_COLS = ["2SW_TS_AUC"]


def json2df(j):
    j2 = []
    for d in j:
        if "ERROR" in d:
            print("SKIPPING ERROR ENTRY: %s" % d)
        else:
            j2.append(d) 
    j = j2

    cols = sorted(set([k for d in j for k in d.keys()]))
    df = pd.DataFrame()
    for col in cols:
        df[col] = [d.get(col, None) for d in j]    
    return df


def retrieve_data(infiles):
    j = []
    for infile in infiles:
        el = json.load(open(infile))
        if type(el) == list:
            j.extend(el)
        else:
            j.append(el)
    
    df = json2df(j)
    return df


def save_summary_files(output, df, VALUE_COLS=VALUE_COLS, CONTROL_COLS=CONTROL_COLS):
    CONTROL_COLS = [c for c in CONTROL_COLS if c in df.columns]
    
    df = df[CONTROL_COLS + sorted(set(df.columns).difference(CONTROL_COLS))]
    df = df.sort_values(CONTROL_COLS)
    df.to_csv(output + ".tsv", sep="\t", index=False, header=True)
    print(df.head())
    
    VALUE_COLS = [c for c in VALUE_COLS if c in df.columns]
    df2 = df[CONTROL_COLS + VALUE_COLS]
    df2.to_csv(output + "_summary.tsv", sep="\t", index=False, header=True)
    print(df.head())
    
    df2 = df[CONTROL_COLS + [c for c in df.columns if "TIME" in c]]
    df2.to_csv(output + "_time_summary.tsv", sep="\t", index=False, header=True)
    print(df.head())
    return df, VALUE_COLS






    
    
    
    
    