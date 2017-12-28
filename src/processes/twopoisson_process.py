#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np


class TwoPoissonProcess:
    """
    Piecewise-constant Poisson process:
    two pieces separated by a switch time.
    """
    
    def __init__(self, intensity0, intensity1, 
                 switch_time, max_time, start_time=0,
                 trend=0.0):
        self._start_time = start_time
        self._max_time = max_time
        self._intensity0 = intensity0
        self._intensity1 = intensity1
        self._switch_time = switch_time
        self._trend = trend
        self._history = []
        
    def start_time(self):
        return self._start_time
    
    def switch_time(self):
        return self._switch_time    
    
    def max_time(self):
        return self._max_time
    
    def max_intensity(self):
        return max(self._intensity0 * (1.0 + self._trend*(self._max_time-self._start_time)),  
                   self._intensity1 * (1.0 + self._trend*(self._max_time-self._start_time)))  
    
    def intensity(self, t, params=None):
        assert t>=self._start_time
        assert t<=self._max_time
        if t <= self._switch_time:
            return self._intensity0 * (1.0 + self._trend*(t-self._start_time))
        else:
            return self._intensity1 * (1.0 + self._trend*(t-self._start_time))
        
    def update(self, t):
        self._history.append(t)
                
    def __str__(self):
        return ("""start=%6s switch=%6s max=%6s maxl=%6s l0=%6s l1=%6s trnd=%6s""" % 
                    ("%.1f" % self._start_time, "%.1f" % self._switch_time, "%.1f" % self.max_time(), 
                     "%.3f" % self.max_intensity(), "%.3f" % self._intensity0, "%.3f" % self._intensity1,
                     "%.3f" % self._trend))
        
    def get_numactions(self):
        return len(self._history)
    
    def get_numactions_before_switch(self):
        return sum(np.asarray(self._history)<self._switch_time)    
    
    def get_numactions_after_switch(self):
        return sum(np.asarray(self._history)>self._switch_time)    

    def get_data(self):
        times, actions = self.get_events() 
        return pd.DataFrame({"time": times, "type": actions})    
    
    def get_events(self):
        times, actions = self._history, ["action" for _ in self._history]
        times = times + [self._start_time, self._switch_time, self._max_time]               
        actions = actions + ["start_time", "switch_time", "max_time"]
        return times, actions
    
    