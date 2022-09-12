# [[file:../../Empirics/cfe_estimation.org::svd_rank1_approximation_with_missing_data_example][svd_rank1_approximation_with_missing_data_example]]
# Tangled on Mon Sep 12 15:51:17 2022
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import warnings

def missing_inner_product(X,min_obs=None):
    """Compute inner product X.T@X, allowing for possibility of missing data."""
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
    assert m>n, "Fewer rows than columns.  Consider passing the transpose."

    # If good has fewer total observations than min_obs, can't possibly
    # have more cross-products.  Dropping here is faster than iterative procedure below.
    X = X.loc[:,X.count()>=min_obs]

    HasMiss=True
    while HasMiss:
        foo = X.cov(min_periods=min_obs).count()
        if np.sum(foo<X.shape[1]):
            badcol=foo.idxmin()
            del X[badcol] # Drop  good with  most missing covariances
            if VERBOSE: print("Dropping %s, with only %d covariances." % (badcol,foo[badcol]))
        else:
            HasMiss=False

    return X

def svd_missing(A,max_rank=None,min_obs=None,heteroskedastic=False):
    """Singular Value Decomposition with missing values

    Returns matrices U,S,V.T, where A~=U*S*V.T.

    Inputs: 
        - A :: matrix or pd.DataFrame, with NaNs for missing data.

        - max_rank :: Truncates the rank of the representation.  Note
                      that this impacts which rows of V will be
                      computed; each row must have at least max_rank
                      non-missing values.  If not supplied rank may be
                      truncated using the Kaiser criterion.

        - min_obs :: Smallest number of non-missing observations for a 
                     row of U to be computed.

        - heteroskedastic :: If true, use the "heteroPCA" algorithm
                       developed by Zhang-Cai-Wu (2018) which offers a
                       correction to the svd in the case of
                       heteroskedastic errors.  If supplied as a pair,
                       heteroskedastic[0] gives a maximum number of
                       iterations, while heteroskedastic[1] gives a
                       tolerance for convergence of the algorithm.

    Ethan Ligon                                        September 2021

    """
    max_its=50
    tol = 1e-3

    P=missing_inner_product(A,min_obs=min_obs) # P = A.T@A

    def heteropca(C,r=1,max_its=max_its,tol=tol):
        """Estimate r factors and factor weights of covariance matrix C."""

        N = C - np.diag(np.diag(C))

        NLast = 1
        t = 0
        while np.linalg.norm(N-NLast)>tol and t<max_its:
            NLast = N

            u,s,vt = np.linalg.svd(N,full_matrices=False)

            Ntilde = u[:,:r]@np.diag(s[:r])@vt[:r,:]

            N = N - np.diag(np.diag(N)) + np.diag(np.diag(Ntilde))

            t += 1

        if t==max_its:
            warnings.warn("Exceeded maximum iterations (%d)" % max_its)

        s = np.sqrt(s[:r])
        
        u = u[:,:r]

        return u,s

    sigmas,u=np.linalg.eigh(P)

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

    if heteroskedastic:
        try:
            max_its,tol = heteroskedastic
        except TypeError:
            pass
            
        u,s = heteropca(P,r=r,max_its=max_its,tol=tol)
    
    us=u@np.diag(s)

    v=np.zeros((len(s),A.shape[1]))
    for j in range(A.shape[1]):
        a=A.iloc[:,j].values.reshape((-1,1))
        x=np.nonzero(~np.isnan(a))[0] # non-missing elements of vector a
        if len(x)>=r:
            v[:,j]=(np.linalg.pinv(us[x,:])@a[x]).reshape(-1)
        else:
            v[:,j]=np.nan

    return u,s,v.T

def svd_rank1_approximation_with_missing_data(x,return_usv=False,max_rank=1,
                                              min_obs=None,VERBOSE=True):
    """
    Return rank 1 approximation to a pd.DataFrame x, where x may have
    elements which are missing.
    """
    x=x.copy()
    m,n=x.shape

    if min_obs is None: min_obs = 1

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

    xhat=pd.DataFrame(s*v@u.T,columns=x.index,index=x.columns).T

    if TRANSPOSE: 
        out = xhat.T
    else:
        out = xhat

    if return_usv:
        u = u.squeeze()
        if u.shape[0] == xhat.shape[1]:
            u = pd.Series(u.squeeze(),index=xhat.columns)
            v = pd.Series(v.squeeze(),index=xhat.index)
        elif u.shape[0] == xhat.shape[0]:
            u = pd.Series(u.squeeze(),index=xhat.index)
            v = pd.Series(v.squeeze(),index=xhat.columns)
        return xhat,u,s,v
    else: return xhat

(n,m)=(1000,500)
a=np.random.normal(size=(n,1))
b=np.random.normal(size=(1,m))
e=np.random.normal(size=(n,m))*1e-5*0

X0=np.array([[-0.22,  0.32, -0.43],
             [0.01, 0.00,  0.00],
             [-0.22,  0.31, -0.42],
             [0.01, -0.03,  0.04],
             [-0.21, 0.31, -0.38]])
X0 = np.outer(a,b) + e

X0=X0-X0.mean(axis=1).reshape((-1,1))

X=X0.copy()
X[0,0]=np.nan
X[0,1]=np.nan

X0=pd.DataFrame(X0).T
X=pd.DataFrame(X).T

def test_symmetry_of_svd_rank1_approximation():
    Xhat=svd_rank1_approximation_with_missing_data(X0,VERBOSE=False)
    XhatT=svd_rank1_approximation_with_missing_data(X0.T,VERBOSE=False)
    assert np.all(Xhat.T == XhatT)

def test_accuracy_of_svd_rank1_approximation():
    Xhat=svd_rank1_approximation_with_missing_data(X,VERBOSE=False)
    error = X0 - Xhat
    assert np.max(np.max(error)<1e-2)
  
Xhat=svd_rank1_approximation_with_missing_data(X,VERBOSE=False)
XhatT=svd_rank1_approximation_with_missing_data(X0.T,VERBOSE=False)

print(X0)
print(X)
print(Xhat)
print((X0-Xhat)/X0)

assert np.linalg.norm((X0-Xhat)/X0,ord=np.inf)//np.sqrt(np.prod(X0.shape)) < 1e-2
# svd_rank1_approximation_with_missing_data_example ends here
