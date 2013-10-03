#!/usr/bin/env python
"""
This is a module with functions which implement various aspects of estimation of VES demand systems.
"""

import pandas as pd
from numpy import linalg

def proj(y,x):
    "Linear projection of a matrix y on a matrix x."

    b=linalg.lstsq(x,y)[0]
    return x*b

def order_by_expenditures(X,Z=None,method='ranks'):
    """
    Construct a wealth ordering of households based on an NxD pandas DataFrame of expenditures X.

    Each row of X corresponds to an observation of expenditures on D goods for a particular household.

    There are two methods for ordering.  The first is by 'total' expenditures;
    the second by the average 'rank' of the household in expenditures across goods.
    
    If a DataFrame Z is also provided, then expenditures are projected onto Z, and orderings
    are based on residual expenditures instead of expenditures.

    Ethan Ligon                                                                October 2013
    """

    if Z is None:  # Use expenditures
        E=X 
    else:  # Use residuals
        E=X-proj(X,Z)

    if method=='total':
        R=E.sum(axis=1).rank().argsort().argsort()
    elif method=='rank':
        R=E.rank().sum(axis=1).rank().argsort().argsort()

    return R
        
