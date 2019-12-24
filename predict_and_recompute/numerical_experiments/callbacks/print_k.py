#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse

def print_k(K):
    '''
    print iteration every K steps
    '''

    def pk(**kwargs):
        '''
        callback function to save the updated solution at each step
    
        Parameters
        ---------- 
        kwargs['k'] : inetger
                      current iteration
    
        '''
    
        name = kwargs['output']['name']
        k = kwargs['k']
        max_iter = kwargs['max_iter']

        if k%K==0:
            print(f"{name}: iteration {k} of {max_iter}",end="\r")

    return pk
