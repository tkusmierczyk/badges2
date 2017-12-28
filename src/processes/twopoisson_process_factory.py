#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import logging
import sys
sys.path.append("../")

from processes.twopoisson_process import TwoPoissonProcess


LOG_DBGINFO = 15
EPS = np.nextafter(0,1)
INF = float("inf")

class TwoPoissonProcessFactory:
    """
    A factory that produces <TwoPoissonProcess> processes. 
    """
    
    def __init__(self, args=""):
        parser = argparse.ArgumentParser() 
        parser.add_argument("-n", "--num_processes", dest='n', 
                            help="number of processes to be created", 
                            type=int, metavar="n", required=False, default=1000)        
              
        parser.add_argument("-sf", "--switch_fraction", dest='switch_fraction', 
                            help="faction of users who 'switch' their intensities", 
                            type=float, metavar="switch_fraction", required=False, default=0.5)        
        
        parser.add_argument("-a1", "--alpha1", dest='a1', 
                            help="intensity until the switching moment", 
                            type=float, metavar="a1", required=False, default=None)        
        parser.add_argument("-a2", "--alpha2", dest='a2', 
                            help="intensity after the switching moment", 
                            type=float, metavar="a2", required=False, default=None)
        
        parser.add_argument("-rand", "--randomize", dest='randomize', 
                            help="if flag set then intensities will be drawn from gamma dist", 
                            required=False, default=False, action="store_true")                
        parser.add_argument("-r1", "--r1", dest='r1', 
                            help="if -rand set: intensity 1 ~ Gamma(r1, lambda1)", 
                            type=float, required=False, default=0.5)        
        parser.add_argument("-l1", "--lambda1", dest='lambda1', 
                            help="if -rand set: intensity 1 ~ Gamma(r1, lambda1)", 
                            type=float, required=False, default=1000)                
        parser.add_argument("-r2", "--r2", dest='r2', 
                            help="if -rand set: intensity 2 ~ Gamma(r2, lambda2)", 
                            type=float, required=False, default=50)        
        parser.add_argument("-l2", "--lambda2", dest='lambda2', 
                            help="if -rand set: intensity 2 ~ Gamma(r2, lambda2)", 
                            type=float, required=False, default=10000)   
        
        parser.add_argument("-u", "--user", dest='fixed_user', 
                            help="intensity change fixed for every user (default True)", 
                            required=False, default=True, action="store_false")           
        
        parser.add_argument("-t", "--trend", dest='trend', 
                            help="intensity trend value", 
                            type=float, required=False, default=0.0)   
                
        help_msg = parser.format_help()        
        logging.log(LOG_DBGINFO, ("\n======================[TwoPoissonProcessFactory arguments description]=======================\n%s" + 
                                  "\n======================[/TwoPoissonProcessFactory arguments description]======================") % help_msg)
        
        args = parser.parse_args(args.split())        
        logging.log(LOG_DBGINFO, "[TwoPoissonProcessFactory] Parsed args: %s" % args)
        self._params = args        
        
        assert args.randomize or args.a1 is not None, "a1 must be randomly drawn with -rand or given explicitly with -a1"
        assert args.randomize or args.a2 is not None, "a2 must be randomly drawn with -rand or given explicitly with -a2"
                
        
    def yield_processes(self):
        switching_count = 0
        nonswitching_count = 0
            
        for process_count in range(self._params.n):
            total_count = switching_count+nonswitching_count
            switching_fraction = 0 if total_count==0 else float(switching_count) / total_count
            
            if self._params.randomize:
                a1 = float(np.random.gamma(shape=self._params.r1, scale=self._params.lambda1, size=1))
                a2 = float(np.random.gamma(shape=self._params.r2, scale=self._params.lambda2, size=1))  
                
                if self._params.fixed_user:
                    m1 = self._params.r1*self._params.lambda1
                    m2 = self._params.r2*self._params.lambda2
                    a2 = a1 + (m2-m1)                              
            else:
                a1 = self._params.a1
                a2 = self._params.a2                
            if a1<=0:   a1 = EPS
            if a2<=0:   a2 = EPS
            
            
            #switch_time = np.random.uniform(low=10, high=100, size=1)[0]
            #max_time    = switch_time + np.random.uniform(low=0, high=20, size=1)[0]
            switch_time = 100/a1#np.random.uniform(low=10, high=100, size=1)[0]
            max_time    = switch_time + switch_time*np.random.uniform(low=0, high=1, size=1)[0]
                                
            if self._params.switch_fraction>0.0 and switching_fraction<=self._params.switch_fraction:
                switching_count += 1
                process = TwoPoissonProcess( max_time=max_time, 
                                             switch_time=switch_time, 
                                             intensity0=a1, intensity1=a2,
                                             trend=self._params.trend)
            else:
                nonswitching_count += 1
                process = TwoPoissonProcess( max_time=max_time, 
                                             switch_time=switch_time, 
                                             intensity0=a1, intensity1=a1,
                                             trend=self._params.trend)
                
            if process_count<10:
                logging.debug("[TwoPoissonProcessFactory] process no %i: %s" % (process_count, process))
            yield process
        
        total_count = switching_count+nonswitching_count
        switching_fraction = 0 if total_count==0 else float(switching_count) / total_count        
        logging.info("[TwoPoissonProcessFactory] %i processes. %.2f switching intensity" % 
                      (total_count, switching_fraction))
        
        
        
