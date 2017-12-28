#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd
import numpy as np
import logging
import json
import matplotlib
from scipy.stats import chi2
from matplotlib import pyplot
import seaborn as sns; sns.set()

sys.path.append("../")

from aux import parsing
from analytics import plots
from synthetic.simulation_plot import *


TREND2XLABEL = {"trend": r"global trend strength, $A$",
                "F": r"badge influence probability, $\pi$",
                "TF": r"fraction of new users"}


def plot_horizontal_level(level=None, level_label=None):
    xmin, xmax = pyplot.gca().get_xlim()
    if level is not None:
        pyplot.axhline(y=level, color="red", lw=4, ls="--")
        if  level_label:
            pyplot.text((xmin+xmax)*0.5, level, level_label, 
                    color="red", verticalalignment="bottom", horizontalalignment="center", fontsize=20)



def plot_data(data, trend_name="trend",
              ylabel="AUC", xlabel=r"trend",              
              columns=["2P_TR_AUC", "2SW_TR_AUC", "2SB_TR_AUC", "BT_TR_AUC", "WT_TR_AUC"], 
              labels=["2P_TR_AUC", "2SW_TR_AUC", "2SB_TR_AUC", "BT_TR_AUC", "WT_TR_AUC"],
              colors = ["dodgerblue", "salmon", "limegreen", "salmon", "limegreen"],
              markers = ["*", "o", "h", ".", "."],
              lws = [4,4,4,1,1],
              lss = ["-","-","-","--","--"],
              **kwargs):
    
    
    xs = list(map(float, data[trend_name]))
    logging.debug("[plot_data] xs = %s" % xs) 
    
    pyplot.tick_params(axis='x', which='major', labelsize=20)
    pyplot.tick_params(axis='y', which='major', labelsize=20)
    pyplot.ylabel(ylabel, fontsize=25)
    pyplot.xlabel(xlabel, fontsize=25)
    pyplot.grid(True)
    
    ax1 = pyplot.gca()
        
    for i in reversed(range(len(columns))):
        column = columns[i]
        label = labels[i]
        color = colors[i%len(colors)] 
        marker = markers[i%len(markers)]
        lw = lws[i%len(lws)]
        ls = lss[i%len(lss)]
        if column not in data.columns:
            logging.warn("WARNING: [plot_data] no column=%s in data" % column)
            continue

        ax1.plot(xs, data[column], label=label, color=color, lw=lw, ls=ls,
                     marker=marker, markeredgecolor="none", markersize=10)
        
    leg = ax1.legend(fontsize=17, fancybox=True, loc=1)
    leg.get_frame().set_alpha(0.75)
    ax1.tick_params(axis='y', which='major', labelsize=20)
    
    #ymin, ymax = pyplot.gca().get_ylim()
    #pyplot.gca().set_ylim((ymin, 1.00))
    pyplot.gca().set_ylim((0.5, 1.00))
    #plot_horizontal_level(0.5)

    xmin, xmax = min(xs)-0.005, max(xs)+0.005
    pyplot.gca().set_xlim((xmin, xmax))
        
    pyplot.gcf().subplots_adjust(bottom=0.17, left=0.15)# right=0.85, left=0.15)
    


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="""Plots output data from simulaiton.py""")
    
    parser.add_argument("-i", "--input", dest='input', 
                        help="input JSON file", nargs='+',
                        required=True, )

    parser.add_argument("-p", "--params", dest='params', 
                        help="comma-separated params: option=value", 
                        required=False, default="")
                           
    parser.add_argument("-d", "--debug", dest='debug', 
                        help="print debug information", 
                        action="store_true", default=False)
        
    parser.add_argument("-t", "--trend", dest='trend_name', 
                        help="trend column name", 
                        required=False, default="trend")
    
    
    args = parser.parse_args(sys.argv[1: ])        
    params = parsing.parse_dictionary(args.params)
    output = args.input[0]
    
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
                
    trend_name = args.trend_name
    
    ##############################################################################################
    
    df = retrieve_data(args.input)
    output = args.input[0]        
    save_summary_files(output, df)

    ##############################################################################################
    
    df.sort_values(by=trend_name, inplace=True)
    df = df.groupby(trend_name).mean().reset_index() #calc group means
    print(df.head())
    
    xlabel = TREND2XLABEL.get(trend_name, trend_name)
    
    plots.pyplot_reset()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True       
    plot_data(df,
              columns=["2P_TR_AUC", "2SW_TR_AUC", "2SB_TR_AUC", "WT_TR_AUC", "BT_TR_AUC"],#, "LR0_TR_AUC", "LR1_TR_AUC"], 
              labels=["Poisson clustering", "2-phase theoretic", "2-phase bootstrap", "NHST theoretic", "NHST bootstrap", "LR0_TR_AUC", "LR1_TR_AUC"],
              colors=["dodgerblue", "salmon", "limegreen", "salmon", "limegreen", "red", "blue"], 
              trend_name=trend_name,
              xlabel=xlabel)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True       
    pyplot.savefig(output+"_"+trend_name+"_TR.pdf")
    
    plots.pyplot_reset()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True       
    plot_data(df,
              columns=["2P_TS_AUC", "2SW_TS_AUC", "2SB_TS_AUC"],#, "LR0_TS_AUC", "LR1_TS_AUC"], 
              labels=["Poisson clustering", "2-phase theoretic", "2-phase bootstrap", "LR0_TS_AUC", "LR1_TS_AUC"],
              colors=["dodgerblue", "salmon", "limegreen", "red", "blue"],
              trend_name=trend_name,
              xlabel=xlabel)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True       
    pyplot.savefig(output+"_"+trend_name+"_TS.pdf")
    
    