#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Clustering users in covariates' space."""

import numpy as np
import logging
from scipy.stats import multivariate_normal as gauss


#data plot
import matplotlib
import matplotlib.cm as cm
from matplotlib import pyplot


def expectation(x, pi, cluster, pdf):
    N = len(x)
    K = pi.shape[1]
    assert K==len(cluster)
    z = np.zeros((N, K))
    
    for n in range(N):
        xn = x[n]
        
        den = sum( (pi[n, j]*pdf(xn, cluster[j])) for j in range(K))
        for k in range(K):
            z[n, k] = pi[n, k]*pdf(xn, cluster[k]) / den
    
    return z


def normal_maximization(x, z, user_group_assignment=None, group_prior_pi=None, weights=None):
    """ 
        Args:
            user_group_assignment  for each user it selects 
            group_prior_pi  each row represents 
    """
    N, D, K = len(x), len(x[0]), len(z[0])
    
    if weights is None: 
        weights = np.ones(N)    
    if user_group_assignment is None or group_prior_pi is None:
        user_group_assignment = np.zeros(N, dtype=int)
        group_prior_pi = np.zeros((1, K))
    
    #update group priors:
    group_pi = np.copy(group_prior_pi)
    for n in range(N): 
        w, g = weights[n], user_group_assignment[n]
        for k in range(K):            
            group_pi[g, k] += w*z[n, k] #BUG FOUND: group_pi[g, k] +        
    #renormalize: K = group_pi.shape[1]
    norm = group_pi.sum(1)
    group_pi = np.divide(group_pi, np.array([norm for _ in range(K)]).T) #TODO rewrite with [:,None] broadcasting
    logging.debug("[em2][normal_em][normal_maximization] group_pi=%s" % "; ".join(map(str, group_pi)))

    #broadcast groups pi to users pi 
    pi = np.zeros((N, K))
    for n in range(N):
        g = user_group_assignment[n]
        pi[n, :] = group_pi[g, :] 

    Nk = (z*weights[:,None]).sum(0) #Nk = z.sum(0)
    mu = np.zeros((K,D))
    ss = np.zeros((K,D,D))
    for k in range(K):
        mu[k] = sum( (z[n,k] * weights[n] * x[n]) for n in range(N)) / Nk[k]
        ss[k] = sum( (z[n,k] * weights[n] * np.outer((x[n]-mu[k]), (x[n]-mu[k]))) for n in range(N)) / Nk[k]
        
    #logging.debug("[em2][normal_em][normal_maximization] N=%i D=%i K=%i Nk=%s mu=%s" % (N, D, K, Nk, mu))
    return pi, mu, ss
            

def ll(x, pi, cluster, pdf):
    N = len(x)
    K = pi.shape[1]
    return sum(np.log(sum(pi[n, k]*pdf(x[n], cluster[k]) 
               for k in range(K))) 
               for n in range(N))


def normal_data_generation_2d(N = 100):

    x1 = np.random.normal(3, 5, N//2)
    x2 = x1+np.random.normal(3, 1, N//2)
    c1 = np.array([x1, x2]).T
    
    x1 = np.random.normal(1, 3, N//2)
    x2 = -x1+np.random.normal(3, 2, N//2)
    c2 = np.array([x1, x2]).T
    
    x = np.concatenate([c1, c2])
    l = np.concatenate([np.zeros(len(c1)), np.ones((len(c2)))])
    z = np.array([1-l, l]).T
    return x, z


def normal_data_generation_1d(N = 100):

    c1 = np.random.normal(3, 5, N//2)
    c2 = np.random.normal(20, 5, N//2)
    
    x = np.concatenate([c1, c2])
    l = np.concatenate([np.zeros(len(c1)), np.ones((len(c2)))])
    z = np.array([1-l, l]).T
    return x, z


def plot_data_2d(x, z):    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.autumn)
    y = x[:,1] if len(x[0])>=2 else np.zeros(len(x))
    pyplot.scatter(x[:,0], y, color=[ cmap.to_rgba(z0) for z0 in z[:, 0]])
    pyplot.grid(True)
    
    x1, x2 = pyplot.xlim()
    y1, y2 = pyplot.ylim()
    pyplot.scatter([1000000], [1000000], s=10, color=cmap.to_rgba(1.0), label="z0")
    pyplot.scatter([1000000], [1000000], s=10, color=cmap.to_rgba(0.0), label="z1")
    pyplot.xlim(x1,x2)    
    pyplot.ylim(y1,y2)
    pyplot.legend() 
    

def pyplot_reset():
    pyplot.cla()
    pyplot.clf()
    pyplot.close()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    
def normal_em(x, user_group_assignment=None, group_prior_pi=None, weights=None,
              K=2, MAX_ITER=20, LL_PREC=10**(-5)):
    
    assert len(x)==len(user_group_assignment)
    
    N = len(x)
    pdf = lambda xn, cluster: gauss.pdf(xn, mean=cluster[0], cov=cluster[1])    
        
    #z = np.random.uniform(size=(len(x), K))
    #norm = z.sum(1)
    #z = np.divide(z, np.array([norm for _ in range(K)]).T)
    
    norm = group_prior_pi.sum(1)
    p = np.divide(group_prior_pi, np.array([norm for _ in range(K)]).T)
    z = np.zeros( (len(x), K) )
    for i in range(N):
        group = user_group_assignment[i]
        z[i, :] = p[group, :]
        #z[i, :] = np.random.multinomial(1, p[group, :], size=1)[0]
    init_z = z        
                
        
    #print("[em2][normal_em] x=",x.shape," z=",z.shape)    
    prev_ll = None
    for it in range(MAX_ITER):
        logging.debug("[em2][normal_em] > it=%i Nk=%s" % (it, z.sum(0)))
                                  
        pi, mu, ss = normal_maximization(x, z, user_group_assignment, group_prior_pi, weights)
        z = expectation(x, pi, list(zip(mu, ss)), pdf)
        clustering_params = {"init_z": init_z, "pi": pi, "mu": mu, "ss": ss} #save for reporting

        #check for convergence
        curr_ll = ll(x, pi, list(zip(mu, ss)), pdf)
        logging.info("[em2][normal_em] iteration=%i ll=%s prev_ll=%s" % (it, curr_ll, prev_ll))
        if prev_ll is not None:            
            if abs((curr_ll-prev_ll)/prev_ll)<LL_PREC:
                logging.info("[em2][normal_em] EM has converged in %i iterations: ll=%s prev_ll=%s!" % (it+1, curr_ll, prev_ll))
                if logging.getLogger("output").getEffectiveLevel()==logging.DEBUG:  pyplot.savefig("/tmp/em2_clustering.png")
                return z, clustering_params                            
        prev_ll = curr_ll

        #report & plot intermediate clustering results
        if logging.getLogger("output").getEffectiveLevel()==logging.DEBUG and (it<20 or it%20==0):
            pyplot_reset()
            plot_data_2d(x, z)
            logging.debug("[em2][normal_em] < saving 2d preview to /tmp/it%s.png" % str(it).zfill(2))
            pyplot.savefig("/tmp/em2_it%s.png" % str(it).zfill(2))
    
    logging.info("[em2][normal_em] << EM has not converged in %i iterations: ll=%s!" % (MAX_ITER, curr_ll))
    return z, clustering_params
    
    
if __name__=="__main__":    
    
    x, z = normal_data_generation_2d()    
    plot_data_2d(x, z)
    pyplot.show()
    
    fmt = '[%(process)4d][%(asctime)s][%(levelname)-5s][%(module)s:%(lineno)d/%(funcName)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    normal_em(x, K=2, MAX_ITER=100, LL_PREC=10**(-6))
    