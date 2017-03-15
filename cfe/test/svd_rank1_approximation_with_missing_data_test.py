
# [[file:~/Dropbox/0Fall2016/cfedemands/Empirics/cfe_estimation.org::*Test%20of%20Rank%201%20SVD%20Approximation%20to%20Matrix%20with%20Missing%20Data][svd_rank1_approximation_with_missing_data_test]]

# Tangled on Wed Mar 15 11:31:45 2017
n=12
m=2000
percent_missing=0.5
SEED=0
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np

def missing_inner_product(X,min_obs=None):
  n,m=X.shape

  if n<m: 
      axis=1
      N=m
  else: 
      axis=0
      N=n

  xbar=X.mean(axis=axis)

  if axis:
      C=(N-1)*X.T.cov(min_periods=min_obs)
  else:
      C=(N-1)*X.cov(min_periods=min_obs)

  return C + N*np.outer(xbar,xbar)

def drop_columns_wo_covariance(X,min_obs=None,VERBOSE=False):
    """Drop columns from pd.DataFrame that lead to missing elements of covariance matrix."""

    m,n=X.shape
    assert(m>n)

    HasMiss=True
    while HasMiss:
        foo = X.cov(min_periods=min_obs).count()
        if np.sum(foo<X.shape[1]):
            badcol=foo.argmin()
            del X[badcol] # Drop  good with  most missing covariances
            if VERBOSE: print("Dropping %s." % badcol)
        else:
            HasMiss=False

    return X

def svd_missing(A,max_rank=None,min_obs=None):

    P=missing_inner_product(A,min_obs=min_obs)

    sigmas,u=np.linalg.eig(P)

    order=np.argsort(-sigmas)
    sigmas=sigmas[order]

    # Truncate rank of representation using Kaiser criterion (positive eigenvalues)
    u=u[:,order]
    u=u[:,sigmas>0]
    s=np.sqrt(sigmas[sigmas>0])

    if max_rank is not None and len(s) > max_rank:
        u=u[:,:max_rank]
        s=s[:max_rank]

    r=len(s)
    us=np.matrix(u)*np.diag(s)

    v=np.zeros((len(s),A.shape[1]))
    for j in range(A.shape[1]):
        a=A.iloc[:,j].as_matrix().reshape((-1,1))
        x=np.nonzero(~np.isnan(a))[0] # non-missing elements of vector a
        if len(x)>=r:
            v[:,j]=(np.linalg.pinv(us[x,:])*a[x]).reshape(-1)
        else:
            v[:,j]=np.nan

    return np.matrix(u),s,np.matrix(v).T

def svd_rank1_approximation_with_missing_data(x,return_usv=False,max_rank=None,min_obs=None,VERBOSE=True):
    """
    Return rank 1 approximation to a pd.DataFrame x, where x may have
    elements which are missing.
    """
    x=x.copy()
    m,n=x.shape

    if n<m:  # If matrix 'thin', make it 'short'
        x=x.T
        TRANSPOSE=True
    else:
        TRANSPOSE=False

    x=x.dropna(how='all',axis=1) # Drop any column which is /all/ missing.
    x=x.dropna(how='all',axis=0) # Drop any row which is /all/ missing.

    x=drop_columns_wo_covariance(x.T,min_obs=min_obs).T
    u,s,v = svd_missing(x,max_rank=max_rank,min_obs=min_obs)
    if VERBOSE:
        print("Estimated singular values: ",)
        print(s)

    xhat=pd.DataFrame(v[:,0]*s[0]*u[:,0].T,columns=x.index,index=x.columns).T

    if TRANSPOSE: xhat=xhat.T

    if return_usv:
        return xhat,u,s,v
    else: return xhat

if SEED:
    np.random.seed(SEED)

a=np.random.normal(size=(n,1))
b=np.random.normal(size=(1,m))
e=np.random.normal(size=(n,m))*5e-1

X0=np.outer(a,b) + e
X0=X0-X0.mean(axis=0)

X=X0.copy()
X[np.random.random_sample(X.shape)<percent_missing]=np.nan

X0=pd.DataFrame(X0).T
X=pd.DataFrame(X).T

Xhat,u,s,v=svd_rank1_approximation_with_missing_data(X,VERBOSE=False,return_usv=True)

rho_a=np.corrcoef(np.c_[a,u[:,0]],rowvar=0)[0,1]
rho_b=pd.DataFrame({'b':b.reshape(-1),'v':v[:,0].A.reshape(-1)}).corr().iloc[0,1]
missing=np.isnan(X.as_matrix()).reshape(-1,1).mean()
print "Proportion missing %g and correlations are %5.4f and %5.4f." % (missing, rho_a,rho_b),
print "Singular value=%g" % s[0],
if SEED: print "Seed=%g" % SEED
else: print

# svd_rank1_approximation_with_missing_data_test ends here
