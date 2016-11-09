
# [[file:~/Dropbox/0Fall2016/vesdemand/Empirics/neediness.org::*Rank%201%20SVD%20Approximation%20to%20Matrix%20with%20Missing%20Data][svd_missing]]

# Tangled on Mon Nov  7 19:02:06 2016
import numpy as np
from oct2py import Oct2Py
octave=Oct2Py()
octave.addpath('../utils/IncPACK/')

def mysvd(X):
    """Wrap np.linalg.svd so that output is "thin" and X=usv.T.
    """
    u,s,vt = np.linalg.svd(X,full_matrices=False)
    s=np.diag(s)
    v = vt.T
    return u,s,v

def svd_missing(X):
    [u,s,v]=octave.svd_missing(X)
    s=np.matrix(s)
    u=np.matrix(u)
    v=np.matrix(v)
    return u,s,v

# svd_missing ends here
