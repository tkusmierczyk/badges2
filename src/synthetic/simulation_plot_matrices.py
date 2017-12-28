#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import logging
import matplotlib
from matplotlib import pyplot
import seaborn as sns; sns.set()

sys.path.append("../")

from aux import parsing
from analytics import plots
from synthetic.simulation_plot import *




CONTROL_COLS = ["trend", "F",  "ishift", "covmshift", "trialno", "ix"]
VALUE_COLS = ["2P_TR_AUC", "2P_TS_AUC", 
              "2SW_TR_AUC", "2SW_TS_AUC", 
              "2SB_TR_AUC", "2SB_TS_AUC",
              #"LR0_TR_AUC", "LR0_TS_AUC",
              #"LR1_TR_AUC", "LR1_TS_AUC",
              "BT_TR_AUC", #"BT_TS_AUC",
              "WT_TR_AUC"] #"WT_TS_AUC",
#VALUE_COLS = ["2SW_TS_AUC"]


def extract_matrix(df, column,  aggregate=np.mean,
                    row_column="ishift", col_column="covmshift"):
    
    rowvals = sorted(df[row_column].unique())
    rowval2rowix = dict((r, j) for j, r in enumerate(rowvals))

    colvals = sorted(df[col_column].unique())
    colval2colix = dict((r, i) for i, r in enumerate(colvals))
    
    N, M = len(rowval2rowix), len(colval2colix)
    m = np.zeros((N, M))
    for g, gd in df.groupby([row_column, col_column]):
        rowval, colval = g[0], g[1]
        
        if colval not in colval2colix: continue
        if rowval not in rowval2rowix: continue
        
        colix = colval2colix[colval]
        rowix = rowval2rowix[rowval]
        
        m[rowix, colix] = aggregate(gd[column])
        logging.debug("[extract_matrix] %s=%s %s=%s => %s" % 
                      (row_column, rowval, col_column, colval, m[rowix, colix]))

    collabels = colvals #list(map(lambda v: "%.2f" % v, colvals))
    rowlabels = rowvals #list(map(lambda v: "%.2f" % v, rowvals))
    return m, rowlabels, collabels


def plot_matrix_values(matrix):
    for y in range(matrix.shape[0]):
        r = matrix.shape[0] - y - 1    
        for x in range(matrix.shape[1]):        
            pyplot.text(x + 0.5 , r + 0.5, 
                        '.%s' % int(np.round(100*matrix[y, x])), 
                fontsize=24,
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='w')    


def plot_matrix(matrix, rowlabels, collabels, title="", xlabel="", ylabel="", clabel="", 
                cmin=None, cmax=None, cmap="Blues"):
    matrix = np.flipud(m)
    rowlabels = list(reversed(rowlabels))
    
    #rowlabels = ticks_fmt(rowlabels)
    #collabels = ticks_fmt(collabels)
    
    #matplotlib.rcParams.update({'font.size':25})
    #sns.set(font_scale=2.0)
    cbar_kws = {"label":clabel} if clabel.strip() != "" else {}
    #cbar_kws.update({"font_scale": 30})
    sns.set(font_scale=2.5)
    heatmap = sns.heatmap(matrix, ax=pyplot.axes(), 
                          yticklabels=rowlabels, xticklabels=collabels, 
                          cmap=pyplot.get_cmap(cmap), cbar_kws=cbar_kws, 
                          vmax=cmax, vmin=cmin)
    #annot=True, fmt="d", linewidths=.5)                      
    plot_matrix_values(matrix)                          
    
    
    pyplot.xlabel(xlabel, fontsize=30)
    pyplot.ylabel(ylabel, fontsize=30)
    pyplot.title(title, fontsize=0)
    pyplot.tick_params(axis='both', which='major', labelsize=24)
    pyplot.gcf().subplots_adjust(bottom=0.15, right=0.99)
    
    

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
        

    args = parser.parse_args(sys.argv[1: ])        
    params = parsing.parse_dictionary(args.params)
    
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
            
    ##############################################################################################
    
    df = retrieve_data(args.input)
    output = args.input[0]        
    save_summary_files(output, df)

    ##############################################################################################

    for trend, tdf in df.groupby("trend"):
        for F, sdf in tdf.groupby("F"):
            for col in VALUE_COLS:
                logging.info("PROCESSING: %s" % col)
                cmap = "Blues" if "TR" in col else "Reds"
                m, rl, cl = extract_matrix(sdf, column=col,  aggregate=np.mean,
                                                     row_column="ishift", col_column="covmshift")
                matplotlib.rcParams.update({'font.size': 24})
                matplotlib.rcParams['pdf.fonttype'] = 42
                matplotlib.rcParams['ps.fonttype'] = 42
                matplotlib.rcParams['text.usetex'] = True       

                plot_matrix(m, rl, cl, 
                            xlabel=r"covariates discrepancy, $\Delta_x$", 
                            ylabel=r"badge effect, $\Delta_\lambda$", 
                            clabel="AUC", 
                            cmin=0.5, cmax=1.0, cmap=cmap)
        
                matplotlib.rcParams['pdf.fonttype'] = 42
                matplotlib.rcParams['ps.fonttype'] = 42  
                matplotlib.rcParams['text.usetex'] = True       
                plots.savefig(output+"_t%s_F%s_%s.pdf" % (trend, F, col))
    
    
    
    
    