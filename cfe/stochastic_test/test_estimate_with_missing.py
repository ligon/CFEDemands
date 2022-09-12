# [[file:../../Empirics/cfe_estimation.org::test_estimate_with_missing][test_estimate_with_missing]]
# Tangled on Mon Sep 12 15:51:17 2022
import numpy as np
from cfe.result import to_dataframe


import pandas as pd
import warnings
import sys
from collections import OrderedDict
from cfe.df_utils import drop_missing, ols, arellano_robust_cov, broadcast_binary_op, use_indices, df_norm

def estimate_reduced_form(y,z,return_v=False,return_se=False,VERBOSE=False):
  """Estimate reduced-form Frisch expenditure/demand system.

  Inputs:
      - y : pd.DataFrame of log expenditures or log quantities, indexed by (j,t,m), 
            where j indexes the household, t the period, and m the market.  
            Columns are different expenditure items, indexed by i.

      - z : pd.DataFrame of household characteristics; index should match that of y.  
            Columns are different characteristics, indexed by l.

  Outputs:
      - a : Estimated good-time-market fixed effects.

      - ce : Residuals (can be provided as an input to get_log_lambdas()).

      - d : Estimated coefficients associated with characteristics z.

      - sed : (Optional, if return_se) Estimated standard errors for coefficients d.

      - sea : (Optional, if return_se) Estimated standard errors for coefficients a.

      - V : (Optional, if return_v) Estimated covariance matrix of coefficients d.

  Ethan Ligon                                            February 2017
  """
  try: # Be a little forgiving if t or m index is missing.
      assert y.index.names==['j','t','m'], "Indices should be (j,t,m)?"
      assert y.columns.name == 'i', "Name of column index should be i?"
  except AssertionError:
      y = y.reset_index()
      if not 'm' in y.columns: y['m']=1
      if not 't' in y.columns: y['t']=1
      y = y.set_index(['j','t','m'])
      y.columns.set_names('i',inplace=True)

  try:
      assert z.index.names==['j','t','m'], "Indices should be (j,t,m)?"
      assert z.columns.name == 'k', "Name of column index should be k?"
  except AssertionError:
      z = z.reset_index()
      if not 'm' in z.columns: z['m']=1
      if not 't' in z.columns: z['t']=1
      z = z.set_index(['j','t','m'])

      z.columns.set_names('k',inplace=True)

  assert len(z.index.intersection(y.index))>0, "Indices of z & y don't match."

  periods = list(set(y.index.get_level_values('t')))
  ms = list(set(y.index.get_level_values('m')))

  # Time-market dummies
  DateLocD = use_indices(y,['t','m'])
  DateLocD = pd.get_dummies(list(zip(DateLocD['t'],DateLocD['m'])))
  DateLocD.index = y.index

  sed = pd.DataFrame(columns=y.columns)
  sea = pd.DataFrame(columns=y.columns)
  a = pd.Series(index=y.columns,dtype=float)
  b = OrderedDict() 
  d = OrderedDict() 
  ce = pd.DataFrame(index=y.index,columns=y.columns)
  V = OrderedDict()

  for i,Item in enumerate(y.columns):
      if VERBOSE: print(Item)

      lhs,rhs=drop_missing([y.iloc[:,[i]],pd.concat([z,DateLocD],axis=1)])
      stdev = rhs.std()
      for constant in stdev[stdev==0].index.tolist():
          warnings.warn("No variation in: %s" % str(constant))
      rhs=rhs.loc[:,rhs.std()>0] # Drop  any X cols with no variation
      useDateLocs=list(set(DateLocD.columns.tolist()).intersection(rhs.columns.tolist()))

      # Calculate deviations
      lhsbar=lhs.mean(axis=0)
      assert ~np.any(np.isnan(lhsbar)), "Missing data in lhs for item %s." % Item
      assert np.all(lhs.std()>0), "No variation in non-missing data for item %s." % Item
      lhs=lhs-lhsbar
      lhs=lhs-lhs.mean(axis=0)

      rhsbar=rhs.mean(axis=0)
      assert ~np.any(np.isnan(rhsbar)), "Missing data in rhs?"
      rhs=rhs-rhsbar
      rhs=rhs-rhs.mean(axis=0)

      # Need to make sure time-market effects sum to zero; add
      # constraints to estimate restricted least squares
      ynil=pd.DataFrame([0],index=[(-1,0,0)],columns=lhs.columns)
      znil=pd.DataFrame([[0]*z.shape[1]],index=[(-1,0,0)],columns=z.columns)
      timednil=pd.DataFrame([[1]*DateLocD.shape[1]],index=[(-1,0,0)],columns=DateLocD.columns)

      X=rhs.append(znil.join(timednil))
      X=X.loc[:,X.std()>0] # Drop  any X cols with no variation

      # Estimate d & b
      myb,mye=ols(X,lhs.append(ynil),return_se=False,return_v=False,return_e=True) 
      ce[Item]=mye.iloc[:-1,:] # Drop constraint that sums time-effects to zero

      if return_v or return_se:
          if z.shape[1]:
              V[Item]=arellano_robust_cov(z,ce[Item])
              sed[Item]=pd.Series(np.sqrt(np.diag(V[Item])), index=z.columns) # reduced form se on characteristics
              
              stderrs = (mye.groupby(['t','m']).std()/np.sqrt(mye.groupby(['t','m']).count()))
              if len(useDateLocs) > 0:
                  sea[Item] = stderrs.squeeze()
              else:
                  sea[Item] = stderrs[Item]

      zvars = z.columns.intersection(myb.index) 
      d[Item]= myb.loc[zvars].squeeze() # reduced form coefficients on characteristics

      b[Item] = myb.loc[useDateLocs].squeeze()  # Terms involving prices
      a[Item] = lhsbar.mean() - d[Item].squeeze().dot(rhsbar[zvars]) - np.array(b[Item]).dot(rhsbar[useDateLocs])

  b = pd.DataFrame(b,index=y.groupby(level=['t','m']).mean().index)
  b = b.T
  sed = sed.T
  sea = sea.T

  if b.shape[1]==1: # Only a single time-market
    assert np.all(np.isnan(b)), "Only one good-time effect should mean b not identified"
    b[:]=0

  d = pd.DataFrame(d).T
  d.index.name = 'i'

  out = [b.add(a,axis=0),ce,d]
  if return_se:
      out += [sed,sea]
  if return_v:
      V = xr.Dataset(V).to_array(dim='i')
      out += [V]
  return out
import pandas as pd
from scipy.stats.distributions import lognorm
import numpy as np

import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=UserWarning)
    import xarray as xr


def geometric_brownian(sigma=1.):
    return lognorm(s=sigma,scale=np.exp(-(sigma**2)/2))

def lambdabar(T,M,Fbar):
    return xr.DataArray(np.cumprod(Fbar.rvs(size=(T,M)),axis=0),
                        dims=('t','m'),
                        coords={'t':range(T),'m':range(M)})

def lambdas(N,T,M=1,G0=lognorm(.5),Fbar=geometric_brownian(.1),F=geometric_brownian(.2)):

    L0 = xr.DataArray(G0.rvs(size=(N,1,M)),dims=('j','t','m'),
                    coords={'j':range(N),'t':range(1),'m':range(M)})  # Initial lambdas
    innov = xr.DataArray(F.rvs(size=(N,T-1,M)),dims=('j','t','m'),
                             coords={'j':range(N),'t':range(1,T),'m':range(M)})

    L = xr.concat((L0,innov),dim='t').transpose('j','t','m')
  
    # Add aggregate shocks Lbar:
    return L*lambdabar(T,M,Fbar=Fbar) #lambdas
def prices(T,M,n,G0=lognorm(.5),Fbar=geometric_brownian(.05),F=geometric_brownian(.2)):

    P0 = xr.DataArray(G0.rvs(size=(n,1,M)),dims=('i','t','m'),
                      coords={'i':range(n),'t':range(1),'m':range(M)})  # Initial lambdas
    innov = xr.DataArray(F.rvs(size=(n,T-1,M)),dims=('i','t','m'),
                               coords={'i':range(n),'t':range(1,T),'m':range(M)})

    P = xr.concat((P0,innov),dim='t').transpose('t','m','i')
    
    # Add aggregate shocks L0:
    return P*lambdabar(T,M,Fbar=Fbar) # prices
def characteristics(N,T,M=1): 
    z = lambdas(N,T,M,Fbar=geometric_brownian(.05),F=geometric_brownian(0.1))
    return z # characteristics

import pandas as pd
from scipy.stats import distributions
import numpy as np

def measurement_error(N,T,M,n,mu_phi=0.,sigma_phi=0.,mu_eps=0.,sigma_eps=1.):
    """Return samples from two measurement error processes; one additive, the other  multiplicative.
  
    - The additive error (phi) is a normal distribution with mean
      =mu_phi= and standard deviation =sigma_phi=.
    
    - The multiplicative error (eps) is a log-normal distribution with mean
      =mu_eps= and standard deviation =sigma_eps=.
    """

    def additive_error(N=N,T=T,M=M,n=n,sigma=sigma_phi):
        return xr.DataArray(distributions.norm.rvs(scale=sigma,size=(N,T,M,n)) + mu_phi,dims=('j','t','m','i'))

    def multiplicative_error(N=N,T=T,M=M,n=n,sigma=sigma_eps):
        return xr.DataArray(np.exp(distributions.norm.rvs(loc=-sigma/2.,scale=sigma,size=(N,T,M,n)) + mu_eps),dims=('j','t','m','i'))

    phi=additive_error(N,T,M,n,sigma=sigma_phi)
    eps=multiplicative_error(N,T,M,n,sigma=sigma_eps)

    return phi,eps

def expenditures(N,T,M,n,k,beta,mu_phi=0,sigma_phi=0.,mu_eps=0,sigma_eps=0.,Fbar=geometric_brownian(.001),p=None,rho_lz=0):
    """Generate artificial expenditures for $N$ households in $M$ markets
    over $T$ periods on $n$ items.  Return dataframe of expenditures
    and a dictionary of "true" underlying variables, the latter as
    type =xarray.DataArray=.

    Households are distinguished by a $k$-vector of characteristics,
    but common Frisch elasticities expressed as an $n$-vector beta.

    If supplied, optional arguments
    (mu_phi,sigma_phi,mu_eps,sigma_eps) describe the parameters of two
    different measurement error processes.  The first is a normally
    distributed additive measurement error process, with mean =mu_phi=
    and standard deviation =sigma_phi=.  The second is a
    multiplicative log-normal error process, with (log) mean =mu_eps=
    and (log) standard deviation =sigma_eps=.

    An optional xarray of prices =p= can also be provided.

    A parameter rho_lz (with default of zero) can be specified to
    induce a correlation between characteristics z and log lambdas.

    Ethan Ligon                                                     January 2018
    """

    if len(beta.shape)<2:
        Theta=xr.DataArray(np.diag(beta),dims=('i','ip'))
    else:
        Theta=xr.DataArray(beta,dims=('i','ip'))

    beta=Theta.sum('ip') # Row sum of elasticity matrix

    l = lambdas(N,T,M,Fbar=Fbar)
    
    foo = xr.DataArray(data=[chr(i) for i in range(ord('a'),ord('a')+k)],name='k',dims='k')

    z = xr.concat([characteristics(N,T,M) for i in range(k)],dim=foo)

    l = l*(z**rho_lz).prod('k') 

    L = np.reshape(l,(N,T,M)) 
    
    if p is None:
        p = prices(T,M,n)

    # Build x in steps
    #x = np.kron(np.log(L),-beta)
    x = np.log(L)*(-beta)
    x = x + np.log(p) - (Theta*np.log(p)).sum('ip') 
    x = x + np.log(z).sum('k')

    x = np.exp(x)

    phi,e=measurement_error(N,T,M,n,mu_phi=mu_phi,sigma_phi=sigma_phi,mu_eps=mu_eps,sigma_eps=sigma_eps)

    truth = xr.Dataset({'beta':beta,'lambdas':l,'characteristics':z,'prices':p,'x0':x})

    x = (x + p*phi) # Additive error
    x = x*e # Multiplicative error

    x = x*(x>0) # Truncation

    return x,truth

def artificial_data(T=2,N=120,M=1,k=2,n=4,sigma_e=0.001,sigma_phi=0.1):

    x,truth=expenditures(N,T,M,n,k,beta=np.linspace(1,3,n),sigma_phi=sigma_phi,sigma_eps=sigma_e)

    y=np.log(x)

    return y,truth

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
import pandas as pd

try: 
    from joblib import Parallel, delayed
    #import timeit
    PARALLEL=True
except ImportError:
    PARALLEL=False
    #warnings.warn("Install joblib for parallel bootstrap.")

PARALLEL = False # Not yet working.

def get_loglambdas(e,TEST=False,time_index='t',max_rank=1,min_obs=None,VERBOSE=False):
    """
    Use singular-value decomposition to compute loglambdas and price elasticities,
    up to an unknown factor of proportionality phi.

    Input e is the residual from a regression of log expenditures purged
    of the effects of prices and household characteristics.   The residuals
    should be arranged as a matrix, with columns corresponding to goods. 
    """ 

    assert e.shape[0]>e.shape[1], "More goods than observations."

    chat = svd_rank1_approximation_with_missing_data(e,VERBOSE=VERBOSE,max_rank=max_rank,min_obs=min_obs).T

    R2 = chat.var()/e.var()

    # Possible that initial elasticity b_i is negative, if inferior goods permitted.
    # But they must be positive on average.
    if chat.iloc[0,:].mean()>0:
        b=chat.iloc[0,:]
    else:
        b=-chat.iloc[0,:]

    loglambdas=(-chat.iloc[:,0]/b.iloc[0])

    # Find phi that normalizes first round loglambdas
    phi=loglambdas.groupby(level=time_index).std().iloc[0]
    loglambdas=loglambdas/phi

    loglambdas=pd.Series(loglambdas,name='loglambda')
    bphi=pd.Series(b*phi,index=e.columns,name=r'\phi\beta')

    if TEST:
        foo=pd.DataFrame(-np.outer(bphi,loglambdas).T,index=loglambdas.index,columns=bphi.index)
        assert df_norm(foo-chat)<1e-4
        #print("blogL norm: %f" % np.linalg.norm(foo-chat))

    return bphi,loglambdas

def iqr(x):
    """The interquartile range of a pd.Series of observations x."""
    q=x.quantile([0.25,0.75])

    try:
        return q.diff().iloc[1]
    except AttributeError:
        return np.nan

def bootstrap_elasticity_stderrs(e,clusterby=['t','m'],tol=1e-2,minits=30,return_v=False,return_samples=False,VERBOSE=False,outfn=None,TRIM=True):
    """Bootstrap estimates of standard errors for \\phi\\beta.

    Takes pd.DataFrame of residuals as input.

    Default is to `cluster' by (t,m) via a block bootstrap.

    If optional parameter TRIM is True, then calculations are
    performed using the interquartile range (IQR) instead of the
    standard deviation, with the standard deviation computed as
    IQR*0.7416 (which is a good approximation provided the
    distribution is normal).

    Ethan Ligon                              January 2017
    """

    def resample(e):
        #e = e.iloc[np.random.random_integers(0,e.shape[0]-1,size=e.shape[0]),:]
        e = e.iloc[np.random.randint(0,e.shape[0],size=e.shape[0]),:]
        e = e - e.mean()
        return e

    def new_draw(e,clusterby):      
        if clusterby:
            S=e.reset_index().groupby(clusterby,as_index=True)[e.columns].apply(resample)
        else:
            S=resample(e)

        bs,ls=get_loglambdas(S)

        return bs

    if outfn: outf=open(outfn,'a')

    delta=1.
    old = pd.Series([1]*e.shape[1])
    new = pd.Series([0]*e.shape[1])
    i=0
    chunksize=2

    assert chunksize>=2, "chunksize must be 2 or more."
    while delta>tol or i < minits:
        delta=np.nanmax(np.abs(old.values.reshape(-1)-new.values.reshape(-1)))
        if VERBOSE and i>chunksize: 
            stat = np.nanmax(np.abs((std0.values.reshape(-1)-std1.values.reshape(-1))/std0.values.reshape(-1)))
            print("Draws %d, delta=%5.4f.  Measure of non-normality %6.5f." % (i, delta, stat))
        old=new

        if PARALLEL:
            #start=timeit.timeit()
            bees = Parallel(n_jobs=chunksize)(delayed(new_draw)(e,clusterby) for chunk in range(chunksize))
            #print(timeit.timeit() - start)
        else:
            #start=timeit.timeit()
            bees = [new_draw(e,clusterby) for chunk in range(chunksize)]
            #print(timeit.timeit() - start)

        if outfn: 
            for bs in bees:
                if np.any(np.isnan(bs)):
                    warnings.warn("Resampling draw with no data?")
                outf.write(','.join(['%6.5f' % b for b in bs])+'\n')

        try:
            B=B.append(bees,ignore_index=True)
        except NameError:
            B=pd.DataFrame(bees,index=range(chunksize)) # Create B

        i+=chunksize

        std0=B.std()
        std1=B.apply(iqr)*0.7416 # Estimate of standard deviation, with trimming
        if TRIM:
            new=std1
        else:
            new=std0

    if outfn: outf.close()

    out = [new]
    if return_samples:
        B.dropna(how='all',axis=1,inplace=True) # Drop any goods always missing estimate
        out += [B]

    if return_v:
        B.dropna(how='all',axis=1,inplace=True) # Drop any goods always missing estimate
        out += [B.cov()]

    if len(out)==1:
        return out[0]
    else:
        return out
import numpy as np
from scipy import sparse
import pandas as pd
from warnings import warn

def df_norm(a,b=None,ignore_nan=True,ord=None):
    """
    Provides a norm for numeric pd.DataFrames, which may have missing data.

    If a single pd.DataFrame is provided, then any missing values are replaced with zeros, 
    the norm of the resulting matrix is returned.

    If an optional second dataframe is provided, then missing values are similarly replaced, 
    and the norm of the difference is replaced.

    Other optional arguments:

     - ignore_nan :: If False, missing values are *not* replaced.
     - ord :: Order of the matrix norm; see documentation for numpy.linalg.norm.  
              Default is the Froebenius norm.
    """
    a=a.copy()
    if not b is None:
      b=b.copy()
    else:
      b=pd.DataFrame(np.zeros(a.shape),columns=a.columns,index=a.index)

    if ignore_nan:
        missing=(a.isnull()+0.).replace([1],[np.NaN]) +  (b.isnull()+0.).replace([1],[np.NaN]) 
        a=a+missing
        b=b+missing
    return np.linalg.norm(a.fillna(0).values - b.fillna(0).values)

def df_to_orgtbl(df,tdf=None,sedf=None,conf_ints=None,float_fmt='\\(%5.3f\\)',bonus_stats=None):
    """
    Returns a pd.DataFrame in format which forms an org-table in an emacs buffer.
    Note that headers for code block should include ":results table raw".

    Optional inputs include conf_ints, a pair (lowerdf,upperdf).  If supplied, 
    confidence intervals will be printed in brackets below the point estimate.

    If conf_ints is /not/ supplied but sedf is, then standard errors will be 
    in parentheses below the point estimate.

    If tdf is False and sedf is supplied then stars will decorate significant point estimates.
    If tdf is a df of t-statistics stars will decorate significant point estimates.

    if sedf is supplied, this creates some space for =bonus_stats= to be reported on each row.

    BUGS: Dataframes that have multiindex columns can't be nicely represented as orgmode tables, 
    but we do our best.
    """
    if len(df.shape)==1: # We have a series?
        df = pd.DataFrame(df) 

    # Test for duplicates in index
    if df.index.duplicated().sum()>0:
        warn('Dataframe index contains duplicates.')

    # Test for duplicates in columns
    if df.columns.duplicated().sum()>0:
        warn('Dataframe columns contain duplicates.')

    try: # Look for a multiindex
        levels = len(df.index.levels)
        names = ['' if v is None else v for v in df.index.names]
    except AttributeError: # Single index
        levels = 1
        names = [df.index.name if (df.index.name is not None) else '']

    def column_heading(df):
        try: # Look for multiindex columns
            collevels = len(df.columns.levels)
            colnames = ['' if v is None else v for v in df.columns.names]
        except AttributeError: # Single index
            collevels = 1
            colnames = [df.columns.name if (df.columns.name is not None) else '']

        if collevels == 1:
            s = '| ' + ' | '.join(names) + ' | ' + '|   '.join([str(s) for s in df.columns])+'  |\n|-\n'
        else:
            colhead = np.array(df.columns.tolist()).T
            lastcol = ['']*collevels
            for l,j in enumerate(colhead.T.copy()):
                for k in range(collevels):
                    if lastcol[k] == j[k]: colhead[k,l] = ''
                lastcol = j

            colhead = colhead.tolist()
            s = ''
            for k in range(collevels):
                if k < collevels - 1:
                    s += '| '*levels + ' | '
                else:
                    s += '| ' + ' | '.join(names) + ' | '
                s += ' | '.join(colhead[k]) + '  |\n'
            s += '|-\n'

        return s

    def se_linestart(stats,i):
        if stats is None: 
            return '|'*levels
        else:
            stats = stats.loc[i]
            assert levels >= len(stats), "Too many columns of bonus stats"
            line = ['']*(levels-len(stats)+1)
            line += stats.tolist()
            return ' | '.join(line) 

    s = column_heading(df)

    if (tdf is None) and (sedf is None) and (conf_ints is None):
        lastidx = ['']*levels
        for i in df.index:
            if levels == 1: # Normal index
                s += '| %s  ' % i
            else:
                for k in range(levels):
                    if lastidx[k] != i[k]:
                        s += '| %s ' % i[k]
                    else:
                        s += '| '
            lastidx =i 
    
            for j in df.columns: # Point estimates
                try:
                    entry='| '+float_fmt+' '
                    if np.isnan(df[j][i]):
                        s+='| --- '
                    else:
                        s+=entry % df[j][i]
                except TypeError:
                    s += '| %s ' % str(df[j][i])
            s+='|\n'
        return s
    elif not (tdf is None) and (sedf is None) and (conf_ints is None):
        lastidx = ['']*levels
        for i in df.index:
            if levels == 1: # Normal index
                s += '| %s  ' % i
            else:
                for k in range(levels):
                    if lastidx[k] != i[k]:
                        s += '| %s ' % i[k]
                    else:
                        s += '| '
            lastidx = i 

            for j in df.columns:
                try:
                    stars=(np.abs(tdf[j][i])>1.65) + 0.
                    stars+=(np.abs(tdf[j][i])>1.96) + 0.
                    stars+=(np.abs(tdf[j][i])>2.577) + 0.
                    stars = int(stars)
                    if stars>0:
                        stars='^{'+'*'*stars + '}'
                    else: stars=''
                except KeyError: stars=''
                entry='| '+float_fmt+stars+' '
                if np.isnan(df[j][i]):
                    s+='| --- '
                else:
                    s+=entry % df[j][i]
            s+='|\n'

        return s
    elif not (sedf is None) and (conf_ints is None): # Print standard errors on alternate rows
        if tdf is not False:
            try: # Passed in dataframe?
                tdf.shape
            except AttributeError:  
                tdf=df[sedf.columns]/sedf

        lastidx = ['']*levels
        for i in df.index:
            if levels == 1: # Normal index
                s += '| %s  ' % i
            else:
                for k in range(levels):
                    if lastidx[k] != i[k]:
                        s += '| %s ' % i[k]
                    else:
                        s += '| '
            lastidx = i 

            for j in df.columns: # Point estimates
                if tdf is not False:
                    try:
                        stars=(np.abs(tdf[j][i])>1.65) + 0.
                        stars+=(np.abs(tdf[j][i])>1.96) + 0.
                        stars+=(np.abs(tdf[j][i])>2.577) + 0.
                        stars = int(stars)
                        if stars>0:
                            stars='^{'+'*'*stars + '}'
                        else: stars=''
                    except KeyError: stars=''
                else: stars=''
                entry='| '+float_fmt+stars+'  '
                if np.isnan(df[j][i]):
                    s+='| --- '
                else:
                    s+=entry % df[j][i]
            s+='|\n' + se_linestart(bonus_stats,i)
            for j in df.columns: # Now standard errors
                s+='  '
                try:
                    if np.isnan(df[j][i]): # Pt estimate miss
                        se=''
                    elif np.isnan(sedf[j][i]):
                        se='(---)'
                    else:
                        se='(' + float_fmt % sedf[j][i] + ')' 
                except KeyError: se=''
                entry='| '+se+'  '
                s+=entry 
            s+='|\n'
        return s
    elif not (conf_ints is None): # Print confidence intervals on alternate rows
        if tdf is not False and sedf is not None:
            try: # Passed in dataframe?
                tdf.shape
            except AttributeError:  
                tdf=df[sedf.columns]/sedf
        lastidx = ['']*levels
        for i in df.index:
            if levels == 1: # Normal index
                s += '| %s  ' % i
            else:
                for k in range(levels):
                    if lastidx[k] != i[k]:
                        s += '| %s ' % i[k]
                    else:
                        s += '| ' 
            lastidx = i 

            for j in df.columns: # Point estimates
                if tdf is not False and tdf is not None:
                    try:
                        stars=(np.abs(tdf[j][i])>1.65) + 0.
                        stars+=(np.abs(tdf[j][i])>1.96) + 0.
                        stars+=(np.abs(tdf[j][i])>2.577) + 0.
                        stars = int(stars)
                        if stars>0:
                            stars='^{'+'*'*stars + '}'
                        else: stars=''
                    except KeyError: stars=''
                else: stars=''
                entry='| '+float_fmt+stars+' '
                if type(df[j][i]) is not str and np.isnan(df[j][i]):
                    s+='| --- '
                else:
                    s+=entry % df[j][i]
            s+='|\n' + se_linestart(bonus_stats,i)

            for j in df.columns: # Now confidence intervals
                s+='  '
                try:
                    ci='[' + float_fmt +','+ float_fmt + ']'
                    ci= ci % (conf_ints[0][j][i],conf_ints[1][j][i])
                except KeyError: ci=''
                entry='| '+ci+'  '
                s+=entry 
            s+='|\n'
        return s

def orgtbl_to_df(table, col_name_size=1, format_string=None, index=None, dtype=None):
  """
  Returns a pandas dataframe.
  Requires the use of the header `:colnames no` for preservation of original column names.

  - `table` is an org table which is just a list of lists in python.
  - `col_name_size` is the number of rows that make up the column names.
  - `format_string` is a format string to make the desired column names.
  - `index` is a column label or a list of column labels to be set as the index of the dataframe.
  - `dtype` is type of data to return in DataFrame.  Only one type allowed.
  """
  import pandas as pd

  if col_name_size==0:
    return pd.DataFrame(table)

  colnames = table[:col_name_size]

  if col_name_size==1:
    if format_string:
      new_colnames = [format_string % x for x in colnames[0]]
    else:
      new_colnames = colnames[0]
  else:
    new_colnames = []
    for colnum in range(len(colnames[0])):
      curr_tuple = tuple([x[colnum] for x in colnames])
      if format_string:
        new_colnames.append(format_string % curr_tuple)
      else:
        new_colnames.append(str(curr_tuple))

  df = pd.DataFrame(table[col_name_size:], columns=new_colnames)

  if index:
    df.set_index(index, inplace=True)

  return df

def drop_missing(X,infinities=False):
    """
    Return tuple of pd.DataFrames in X with any 
    missing observations dropped.  Assumes common index.

    If infinities is false values of plus or minus infinity are 
    treated as missing values.
    """

    foo=pd.concat(X,axis=1)
    if not infinities:
        foo.replace(np.inf,np.nan)
        foo.replace(-np.inf,np.nan)

    foo = foo.dropna(how='any')

    assert len(set(foo.columns))==len(foo.columns) # Column names must be unique!

    Y=[]
    for x in X:
        Y.append(foo.loc[:,pd.DataFrame(x).columns]) 

    return tuple(Y)

def use_indices(df,idxnames):
    return df.reset_index()[idxnames].set_index(df.index)

y,truth=artificial_data(T=2,N=5000,k=2,n=10,sigma_e=1e-10)

y = to_dataframe(y,['j','t','m']).T
#y = y.reset_index().set_index(['j','t','m'])

#beta,L,dz,p=truth
dz = to_dataframe(truth['characteristics'],['j','t','m']).T

#dz=dz.reset_index().set_index(['j','t','m'])
dz=np.log(dz)

numeraire=None #'x0'

# Try with missing data for contrast
y.values[np.random.random_sample(y.shape)<0.0]=np.nan

y.replace(-np.inf,np.nan,inplace=True)

#b,ce,d,V=estimate_bdce_with_missing_values(y,dz,return_v=True)
b,ce,d = estimate_reduced_form(y,dz,return_v=False)

bphi,logL=get_loglambdas(ce,TEST=True)
cehat=np.outer(pd.DataFrame(bphi),pd.DataFrame(-logL).T).T
cehat=pd.DataFrame(cehat,columns=bphi.index,index=logL.index)

print("Norm of error in approximation of CE: %f" % df_norm(cehat,ce))

# Some naive standard errors

#yhat=b.T.add(cehat + (dz.dot(d.T)),axis=0,level='t')
yhat = broadcast_binary_op(cehat + dz.dot(d.T),lambda x,y: x+y,b.T)

e=y.sub(yhat)

C = pd.DataFrame({"L0":to_dataframe(np.log(truth['lambdas'])),"Lhat":logL.squeeze()}).corr()
print("Correlation of log lambda with estimate (before normalization): %f" % C.values[0][-1])
assert C.values[0][-1]>0.97, "loglambda correlation with truth too low."

if not numeraire is None:
    logL=broadcast_binary_op(logL,lambda x,y: x+y,b.loc[numeraire]) # Add term associated with numeraire good
    b=b-b.loc[numeraire]
else:
    logL=broadcast_binary_op(logL,lambda x,y: x+y,b.mean()) # Add term associated with numeraire good
    b=b-b.mean()

# Evaluate estimate of beta:
print("Norm of (bphi,beta): %f" % np.var(bphi/truth['beta'])) # Funny norm deals with fact that b only identified up to a scalar

C = pd.DataFrame({"L0":to_dataframe(np.log(truth['lambdas'])),"Lhat":logL.squeeze()}).corr()
print("Correlation of log lambda with estimate (after normalization): %f" % C.values[0][-1])
assert C.values[0][-1]>0.95, "loglambda correlation with truth too low."

print("Mean of errors:")
print(e.mean(axis=0))

def test_mean():
    assert np.abs(e.stack().mean())/e.stack().std() < 1e-2    

test_mean()
# test_estimate_with_missing ends here
