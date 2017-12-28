#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""The simplest process (sample code)."""

import argparse



class PoissonProcess:
    
    def __init__(self, max_time, intensity):
        self._max_time = max_time
        self._intensity = intensity        
    
    def max_time(self):
        return self._max_time
    
    def max_intensity(self):
        return self._intensity  
    
    def intensity(self, t, params):
        return self._intensity    


class PoissonProcessFactory:
    
    def __init__(self, args=""):
        print("[PoissonProcessFactory] Constructing process object (args='%s')" % args)
        parser = argparse.ArgumentParser(description="[PoissonProcess] Poisson temporal point process")
        parser.add_argument("-n", "--num", dest='numsamples', type=int,
                        help="how many times sampling should be repeated",
                        metavar="numsamples", required=False, default=1)            
        parser.add_argument("-i", "--intensity", dest='intensity', type=float,
                            help="intensity value",
                            metavar="intensity_value", required=True)
        parser.add_argument("-t", "--time", dest='time', help="max time", type=float,
                            metavar="time", required=True)
        args = parser.parse_args(args.split())
        print("[PoissonProcessFactory] Args:", args)
        
        self._intensity = args.intensity
        self._max_time = args.time
        self._numsamples = args.numsamples
                
        print("[PoissonProcessFactory] intensity = %f" % self._intensity)
        print("[PoissonProcessFactory] max_time = %f" % self._max_time)                
        print("[PoissonProcessFactory] numsamples = %i" % self._numsamples)                
    
    def yield_processes(self):
        for _ in range(self._numsamples):
            yield PoissonProcess(self._max_time, self._intensity)
            
            