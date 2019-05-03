#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

'''
IMPORT VARIANTS
---------------

These implementations are written to (mostly) align with the presentation of algorithms in [Greenbaum, Liu, Chen 2019] and [Chen, ... , 2019]

Note that each of the varaints has a section of each iteration where we "update indexing".
This is not computationally efficient, but it allows us to use more consistent variable names which makes numerical tests easier to write.


    Parameters
    ----------           
    A : (n,n) array_like
        SPD matrix from system $Ax=b$
    b : (n,) array_like
        right hand side from system $Ax=b$
    x0 : (n,) array_like
         Initial guess to solution $Ax=b$
    max_iter : integer
               number of iterations
    callbacks : list of functions
                each callback function will be called on all local variables each iteration
    
    
    Variant specific parameters
    ---------------------------
    preconditioner : callable from (n,) to (n,)
                     (if preconditioned variant) compute preconditioned applied to a vector
    w_replace : function: int -> bool
                (if gvcg) should return True if w_k should be replaced with Ar_k and False otherwise
    
    
    
    Returns
    -------
    Bunch object with fields defined based on which callback functions were listed in callbacks
    
    Notes
    -----
    GVCG is also referred to as pipelined CG or pipelined Chronopoulos/Gear CG.

    
    

    
STYLE GUIDE
-----------

    _k for _{k}, _k1 for _{k-1}, _k2 for _{k-2}
    vectors : 1,2 letters + t for tilde
    scalars : 1-3 letters
    "=" sign 7 spaces from start of definition, with 2 spaces after
    6 spaces from start of vector before next operation

'''

from .hs_cg import hs_cg, hs_pcg
from .cg_cg import cg_cg, cg_pcg
from .gv_cg import gv_cg, gv_pcg
from .m_cg import m_cg, m_pcg

from .ch_cg import ch_cg, ch_pcg
from .pipe_ch_cg import pipe_ch_cg, pipe_ch_cg_b, pipe_ch_pcg, pipe_ch_pcg_b
from .pipe_m_cg import pipe_m_cg, pipe_m_cg_b, pipe_m_pcg, pipe_m_pcg_b

from .pipe_ch_cg_rr import pipe_ch_cg_rr, pipe_ch_cg_b_rr, pipe_ch_pcg_rr, pipe_ch_pcg_b_rr
from .pipe_m_cg_rr import pipe_m_cg_rr, pipe_m_cg_b_rr, pipe_m_pcg_rr, pipe_m_pcg_b_rr

from .exact_cg import exact_cg



