
# [[file:~/Dropbox/0Fall2016/vesdemand/Empirics/neediness.org::*Test%20of%20Rank%201%20SVD%20Approximation%20to%20Matrix%20with%20Missing%20Data][svd_rank1_approximation_with_missing_data_test]]

# Tangled on Mon Nov  7 19:02:06 2016
percent_missing=0.2
import numpy as np
import pandas as pd
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

def svd_rank1_approximation_with_missing_data(x,return_usv=False,VERBOSE=True): 
    """
    Return rank 1 approximation to a pd.DataFrame x, where x may have
    elements which are missing.
    """
    m,n=x.shape

    if n<m: 
        x=x.dropna(how='all')
        x=x.T
        TRANSPOSE=True
    else:
        x=x.dropna(how='all',axis=1)
        TRANSPOSE=False

    u,s,v=svd_missing(x.as_matrix())
    if VERBOSE:
        print("Estimated singular values: ",)
        print(s)

    xhat=pd.DataFrame(v[:,0]*s[0]*u[:,0].T,columns=x.index,index=x.columns)

    if not TRANSPOSE: xhat=xhat.T

    if return_usv:
        return xhat,u,s,v
    else: return xhat

(n,m)=(50,5000)
a=np.random.normal(size=(n,1))
b=np.random.normal(size=(1,m))
e=np.random.normal(size=(n,m))*1e-1

X0=np.outer(a,b)+e

X=X0.copy()
X[np.random.random_sample(X.shape)<percent_missing]=np.nan

X0=pd.DataFrame(X0).T
X=pd.DataFrame(X).T

Xhat=svd_rank1_approximation_with_missing_data(X,VERBOSE=False)

print("Proportion missing %g and correlation %5.4f" % (percent_missing, pd.concat([X.stack(dropna=False),Xhat.stack()],axis=1).corr().iloc[0,1]))

# svd_rank1_approximation_with_missing_data_test ends here
