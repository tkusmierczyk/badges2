#!/usr/bin/python3
# -*- coding: utf-8 -*-

#data plot
import matplotlib
import matplotlib.cm as cm
from matplotlib import pyplot

from scipy.stats import gamma
from scipy.stats import wishart

import pandas as pd
import numpy as np
import logging
import sys
sys.path.append("../")

from synthetic.thinning import run as synthesis
from analytics import plots
from aux import sharing


def _plot_data_2d(x, z, **kwargs):    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.autumn)
    y = x[:,1] if len(x[0]>=2) else np.zeros(len(x))
    pyplot.scatter(x[:,0], y, color=[ cmap.to_rgba(i) for i in z[:, 0]], **kwargs)
    
    
def _mean_std_to_shape_scale(mean0, stddev0):
    shape0 = mean0 * mean0 / (stddev0 * stddev0)
    scale0 = stddev0 / np.sqrt(shape0)
    return shape0, scale0


def _plot_gamma_(rng, shape, scale, **kwargs):
    if logging.getLogger("output").getEffectiveLevel()>logging.DEBUG: return    
    mean0, stddev0 = shape * scale, np.sqrt(shape) * scale    
    kwargs["label"] = kwargs.get("label", "")+(" m=%.3f s=%.3f shp=%.3f scale=%.3f" % (mean0, stddev0, shape, scale))    
    rng = int(np.ceil(rng))
    rv = gamma(shape, scale=scale)
    xs = (np.array(range(rng*20)))/10
    ys = rv.pdf(xs)
    pyplot.plot(xs, ys, **kwargs)


def _report_features(features0, features1, features, NC):
    if logging.getLogger("output").getEffectiveLevel()>logging.DEBUG: return
    plots.pyplot_reset()
    logging.debug("[sample_data] Storing features to /tmp/sample_data_features.tsv")
    features.to_csv("/tmp/sample_data_features.tsv", sep="\t", header=True, index=False)
    if NC < 2: return
    x = np.asarray(features0[["cov0", "cov1"]])
    z = np.asarray(list(zip(1 - features0["group"], features0["group"])))
    _plot_data_2d(x, z, label="$i_u=0$")
    x = np.asarray(features1[["cov0", "cov1"]])
    z = np.asarray(list(zip(1 - features1["group"], features1["group"])))
    _plot_data_2d(x, z, label="$i_u=1$")
    pyplot.xlabel("covariate 0")
    pyplot.ylabel("covariate 1")
    pyplot.grid(True)
    pyplot.legend()
    plots.savefig("/tmp/sample_data_features.png")


def _report_sampling(shape0, scale0, shape0a, scale0a, shape1a, scale1a):
    if logging.getLogger("output").getEffectiveLevel()>logging.DEBUG: return
    plots.pyplot_reset()    
    _plot_gamma_(30, shape0, scale0, label=r"$i_u=0$:", color="dodgerblue", lw=2)
    _plot_gamma_(30, shape0a, scale0a, label=r"$i_u=1$ before:", color="salmon", lw=2)
    _plot_gamma_(30, shape1a, scale1a, label=r"$i_u=1$ after:", color="red", lw=2)
    pyplot.grid(True)
    pyplot.legend(fontsize=8)
    pyplot.savefig("/tmp/sample_data_intensities_priors.png")


def data_generator(params):
    N = int(params.get("N", 400)) #number of users
    
    TF  = float(params.get("TF", 0.5)) #fraction of test users
    NTS = int(np.round(N*TF))
    NTR = N-NTS
    
    F   = float(params.get("F", 0.5)) #fraction of influenced (chaning parameters) users
    FTR = float(params.get("FTR", F)) #fraction of influenced in training
    FTS = float(params.get("FTS", F)) #fraction of influenced in testing
    assert FTR>=0.0 and FTR<=1.0
    assert FTS>=0.0 and FTS<=1.0    
    
    NTS1 = int(np.round(NTS*FTS))
    NTS0 = NTS-NTS1
    
    NTR1 = int(np.round(NTR*FTR))
    NTR0 = NTR-NTR1

    N0 = NTR0+NTS0
    N1 = NTR1+NTS1
            
    ixs_tr0 = np.array(range(NTR0), dtype=int) + 0 
    ixs_ts0 = np.array(range(NTS0), dtype=int) + NTR0
    ixs_tr1 = np.array(range(NTR1), dtype=int) + N0
    ixs_ts1 = np.array(range(NTS1), dtype=int) + N0+NTR1

    train_ids = np.concatenate([ixs_tr0, ixs_tr1])
    test_ids  = np.concatenate([ixs_ts0, ixs_ts1])
    
    ixs0 = np.concatenate([ixs_tr0, ixs_ts0])
    ixs1 = np.concatenate([ixs_tr1, ixs_ts1]) 
    assert sum( np.concatenate([ixs0, ixs1])!=np.array(range(N), dtype=int) )==0
    
    logging.info("[sample_data] %i users: %i influenced: %i training, %i testing; %i not inf.: %i training, %i testing"
                  % (N,N1,NTR1,NTS1,N0,NTR0,NTS0))
    
    ###########################################################################
    #SAMPLES:
    
    m, s = params.get("imean", 10), params.get("istd", 5)
    shift = params.get("ishift", 1.0)
    trend = params.get("trend", 0.0)

    #users who do not care
    mean0, stddev0 = m, s     
    shape0, scale0 = _mean_std_to_shape_scale(mean0, stddev0)
    
    #users who care
    mean0a, stddev0a = m-shift, s 
    shape0a, scale0a = _mean_std_to_shape_scale(mean0a, stddev0a)
    mean1a, stddev1a = m+shift, s     
    shape1a, scale1a = _mean_std_to_shape_scale(mean1a, stddev1a)

    sharing.report_d({"N": N, "TF": TF, "F": F, "FTR": FTR, "FTS": FTS,  
                      "trend": trend,
                      "imean": m, "istd": s, "shift": shift,
                      "mean0": mean0, "stddev0": stddev0, "shape0": shape0, "scale0": scale0,
                      "mean0a": mean0a, "stddev0a": stddev0a, "shape0a": shape0a, "scale0a": scale0a,
                      "mean1a": mean1a, "stddev1a": stddev1a, "shape1a": shape1a, "scale1a": scale1a}, 
                     label="data_")
    _report_sampling(shape0, scale0, shape0a, scale0a, shape1a, scale1a)
    
    data0 = synthesis(("""-d ../processes/twopoisson_process_factory.py                 
                 -n %(n)s -sf 0.0
                 --randomize
                 --r1 %(r0)s --lambda1 %(lambda0)s 
                 --r2 %(r1)s --lambda2 %(lambda1)s 
                 -t %(trend)s
                 """ % #--seed %(seed)i                 
                        { "r0": shape0, "lambda0": scale0,
                         "r1": shape0, "lambda1": scale0,
                         "n": N0, "trend": trend}).split())
    samples0 = data0["samples"]
    pos2ix = dict(enumerate(list(ixs0)))
    samples0["id"] = samples0["id"].apply(lambda i: pos2ix[i])
    processes0  = data0["processes"]
    
    data1 = synthesis(("""-d ../processes/twopoisson_process_factory.py                 
                 -n %(n)s -sf 1.0
                 --randomize                                  
                 --r1 %(r0)s --lambda1 %(lambda0)s 
                 --r2 %(r1)s --lambda2 %(lambda1)s 
                 -t %(trend)s
                 """ % #--seed %(seed)i 
                       { "r0": shape0a, "lambda0": scale0a,
                         "r1": shape1a, "lambda1": scale1a,
                         "n": N1, "trend": trend}).split())
    samples1 = data1["samples"]
    pos2ix = dict(enumerate(list(ixs1)))
    samples1["id"] = samples1["id"].apply(lambda i: pos2ix[i])    
    processes1  = data1["processes"]     
    
    processes = processes0 + processes1
    samples = pd.concat([samples0, samples1])

    if params.get("debug", False):
        logging.debug("[sample_data] Storing samples to /tmp/sample_data_time_samples.tsv")
        samples1.to_csv("/tmp/sample_data_time_samples.tsv", index=False, header=True, sep="\t")    

    
    ###########################################################################
    #FEATURES:
        
    #covariates dimensionality
    NC = params.get("covdim", 2)    
    assert NC==2    
    
    #covariates clustering mode = how covariates of influenced/not-influenced are generated
    covclust = params.get("covclust", 0)  
    
    if covclust==2: #not-influenced users in the center, influenced users in two clusters on both sides 
        means0 = np.array([0, 0])
        w = wishart(10, [[2,1],[1,2]])
        cov2 = cov1 = cov0 = w.rvs()
        
        eigenvalues, eigenvectors = np.linalg.eig(cov0)
        means2 = means1 = means0
        s = params.get("covmshift", 1.0)
        if eigenvalues[0]>eigenvalues[1]:   
            means1 = means1 + s*np.sqrt(eigenvalues[0])*eigenvectors[:, 0]
            means2 = means2 - s*np.sqrt(eigenvalues[0])*eigenvectors[:, 0]
        else:                               
            means1 = means1 + s*np.sqrt(eigenvalues[1])*eigenvectors[:, 1]
            means2 = means2 - s*np.sqrt(eigenvalues[1])*eigenvectors[:, 1]
            
        logging.debug("[sample_data] m0=%s cov0=%s" % (means0, "; ".join(map(str, cov0))))
        samples0 = np.random.multivariate_normal(means0, cov0, size=N0)        
        logging.debug("[sample_data] m1=%s cov1=%s" % (means1, "; ".join(map(str, cov1))))
        samples1 = np.random.multivariate_normal(means1, cov1, size=N1//2)                
        logging.debug("[sample_data] m2=%s cov2=%s" % (means2, "; ".join(map(str, cov2))))
        samples2 = np.random.multivariate_normal(means2, cov2, size=N1-len(samples1))                
        
        samples1 = np.concatenate((samples1, samples2), axis=0)
    else: #one cluster per each class: not-influenced users in the center, influenced shifted 
        means0 = np.array([0, 0])
        w = wishart(10, [[2,1],[1,2]])
        cov1 = cov0 = w.rvs()
        eigenvalues, eigenvectors = np.linalg.eig(cov0)
        means1 = means0
        s = params.get("covmshift", 1.0)
        if eigenvalues[0]>eigenvalues[1]:   means1 = means1 + s*np.sqrt(eigenvalues[0])*eigenvectors[:, 0]
        else:                               means1 = means1 + s*np.sqrt(eigenvalues[1])*eigenvectors[:, 1]
            
        logging.debug("[sample_data] m0=%s cov0=%s" % (means0, "; ".join(map(str, cov0))))
        samples0 = np.random.multivariate_normal(means0, cov0, size=N0)        
        logging.debug("[sample_data] m1=%s cov1=%s" % (means1, "; ".join(map(str, cov1))))
        samples1 = np.random.multivariate_normal(means1, cov1, size=N1)        

    #####################################
    
    features0 = pd.DataFrame(samples0)
    features0["id"] = ixs0
    features0["group"] = 0
    features0.rename(columns=dict((i, "cov%i"%i) for i in range(NC)), inplace=True)

    features1 = pd.DataFrame(samples1)
    features1["id"] = ixs1
    features1["group"] = 1
    features1.rename(columns=dict((i, "cov%i"%i) for i in range(NC)), inplace=True)         

    features = features0.append(features1)
        
    sharing.report_d({"cmeans0": str(means0), "cov0": "; ".join(map(str, cov0)),
                      "cmeans1": str(means1), "cov1": "; ".join(map(str, cov1))}, "data_")        
    _report_features(features0, features1, features, NC)             

    assert set(features["id"].unique())==set(samples["id"].unique())
    assert len(features)==len(processes)
    
    ###########################################################################    
    #ids = np.array(features["id"])
    group = np.array(features["group"])
    return list(train_ids), list(test_ids), features, samples, processes, group 
    
    
if __name__=="__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("output").setLevel(logging.DEBUG)
    ids, features, samples, processes, group = data_generator({"debug": True, "covmshift": 1.0})
    print(sharing.report_retrieve())
    
    