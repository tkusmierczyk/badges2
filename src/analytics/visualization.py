#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib
from matplotlib import pyplot
import seaborn as sns; sns.set()
import logging


def pyplot_reset():
    pyplot.cla()
    pyplot.clf()
    pyplot.close()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42 
    matplotlib.rcParams['text.usetex'] = True
    
    
def pyplot_savefig(path):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42 
    matplotlib.rcParams['text.usetex'] = True       
    pyplot.savefig(path)
      
   
    
def plot_groups(features, feature_names,
                 train_ids, train_group,
                 test_ids = [], test_group = [],
                 train_marker = "+", test_marker="x",
                 xlabel=None, ylabel=None, 
                 plot_legend=False,
                 plot_densities=True,
                 **kwargs):
    train_ids, test_ids = list(train_ids), list(test_ids)
    train_group, test_group = np.asarray(train_group), np.asarray(test_group)
    
    assert len(train_ids)==len(train_group)
    assert len(test_group)==0 or len(test_ids)==len(test_group)
    assert len(feature_names)==2
    
    x_all = np.asarray(features[feature_names]) #remove id and group info from features
    id2pos = dict((uid, pos) for pos, uid in enumerate(features["id"]))
    cm = pyplot.cm.get_cmap('coolwarm')

    logging.debug("plot train")    
    x = x_all[[id2pos[i] for i in train_ids] ,:]
    y = x[:,1] if len(x[0])>=2 else np.zeros(len(x))
    sc = pyplot.scatter(x[:,0], y, c=train_group, 
                        cmap=cm, marker=train_marker, 
                        label="validation", **kwargs)

    logging.debug("plot test")
    if len(test_ids)>0:
        x = x_all[[id2pos[i] for i in test_ids] ,:]   
        y = x[:,1] if len(x[0])>=2 else np.zeros(len(x))
        if len(test_group)==0:
            pyplot.scatter(x[:,0], y, color="lightgrey", 
                           marker=test_marker, label="prediction", **kwargs)
        else:
            pyplot.scatter(x[:,0], y, c=test_group, 
                           cmap=cm, marker=test_marker, label="prediction", **kwargs)

    if plot_legend:
        leg = pyplot.gca().legend(fontsize=20, fancybox=True)
        leg.get_frame().set_alpha(0.75)

    if plot_densities:
        logging.debug("density plots")    
        ids, weights = train_ids+test_ids, np.asarray(list(train_group)+list(test_group))         
        ids = np.random.choice(ids, size=1000, replace=True, p=(weights)/sum(weights))
        x = x_all[[id2pos[i] for i in ids] ,:]
        y = x[:,1] if len(x[0])>=2 else np.zeros(len(x))
        sns.kdeplot(x[:,0], y, n_levels=4, cmap="Reds", lw=4)
    
        ids, weights = train_ids+test_ids, 1.0-np.asarray(list(train_group)+list(test_group))      
        ids = np.random.choice(ids, size=1000, replace=True, p=(weights)/sum(weights))
        x = x_all[[id2pos[i] for i in ids] ,:]
        y = x[:,1] if len(x[0])>=2 else np.zeros(len(x))
        sns.kdeplot(x[:,0], y, n_levels=4, cmap="Blues", lw=4)


    logging.debug("plot axis")
    pyplot.tick_params(axis='x', which='major', labelsize=20)
    pyplot.tick_params(axis='y', which='major', labelsize=20)
    if not xlabel: xlabel = feature_names[0]
    if not ylabel: ylabel = feature_names[1]
    pyplot.xlabel(xlabel, fontsize=25)
    pyplot.ylabel(ylabel, fontsize=25)
    cbar = pyplot.colorbar(sc, label=r"$\hat{P}[{i}_u=1]$")
    cbar.ax.tick_params(labelsize=20)
    text = cbar.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=25)
    text.set_font_properties(font) 
    pyplot.grid(True)
    
    
    
def plot_bivariate(mu, Sigma, limits=None, N=60, **kwargs):
        
    if limits:
        xmin, ymin, xmax, ymax = limits
    else:
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)                
        delta0 = 3*np.sqrt(eigenvalues[0])*eigenvectors[:, 0]
        delta1 = 3*np.sqrt(eigenvalues[1])*eigenvectors[:, 1]
        xmin0, ymin0 = mu-delta0
        xmax0, ymax0 = mu+delta0
        xmin1, ymin1 = mu-delta1
        xmax1, ymax1 = mu+delta1
        xmin, ymin = min(xmin0, xmin1), min(ymin0, ymin1)
        xmax, ymax = max(xmax0, xmax1), max(ymax0, ymax1)
    
    # Our 2-dimensional distribution will be over variables X and Y    
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)
    cset = pyplot.contour(X, Y, Z, **kwargs)

    