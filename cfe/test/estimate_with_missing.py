
# [[file:~/Dropbox/0Fall2016/cfedemands/Empirics/cfe_estimation.org::*Tests%20of%20estimation%20with%20missing%20data][test_estimate_with_missing]]

# Tangled on Wed Mar 15 11:31:51 2017
import numpy as np
import pandas as pd
import warnings
import sys
from collections import OrderedDict
import numpy as np
from scipy import sparse

def df_norm(a,b=None,ignore_nan=True):
    a=a.copy()
    if not b is None:
      b=b.copy()
    else:
      b=pd.DataFrame(np.zeros(a.shape),columns=a.columns,index=a.index)

    if ignore_nan:
        missing=(a.isnull()+0.).replace([1],[np.NaN]) +  (b.isnull()+0.).replace([1],[np.NaN]) 
        a=a+missing
        b=b+missing
    return np.linalg.norm(a.fillna(0).as_matrix() - b.fillna(0).as_matrix())

def df_to_orgtbl(df,tdf=None,sedf=None,conf_ints=None,float_fmt='%5.3f'):
    """
    Returns a pd.DataFrame in format which forms an org-table in an emacs buffer.
    Note that headers for code block should include ":results table raw".

    Optional inputs include conf_ints, a pair (lowerdf,upperdf).  If supplied, 
    confidence intervals will be printed in brackets below the point estimate.

    If conf_ints is /not/ supplied but sedf is, then standard errors will be 
    in parentheses below the point estimate.

    If tdf is False and sedf is supplied then stars will decorate significant point estimates.
    If tdf is a df of t-statistics stars will decorate significant point estimates.
    """
    if len(df.shape)==1: # We have a series?
       df=pd.DataFrame(df)

    if (tdf is None) and (sedf is None) and (conf_ints is None):
        return '|'+df.to_csv(sep='|',float_format=float_fmt,line_terminator='|\n|')
    elif not (tdf is None) and (sedf is None) and (conf_ints is None):
        s = '|  |'+'|   '.join([str(s) for s in df.columns])+'\t|\n|-\n'
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns:
                try:
                    stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                    if stars>0:
                        stars='^{'+'*'*stars + '}'
                    else: stars=''
                except KeyError: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n'

        return s
    elif not (sedf is None) and (conf_ints is None): # Print standard errors on alternate rows
        if tdf is not False:
            try: # Passed in dataframe?
                tdf.shape
            except AttributeError:  
                tdf=df[sedf.columns]/sedf
        s = '|  |'+'|   '.join([str(s) for s in df.columns])+'  |\n|-\n'
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns: # Point estimates
                if tdf is not False:
                    try:
                        stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                        stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                        stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                        if stars>0:
                            stars='^{'+'*'*stars + '}'
                        else: stars=''
                    except KeyError: stars=''
                else: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n|'
            for j in df.columns: # Now standard errors
                s+='  '
                try:
                    se='$(' + float_fmt % sedf.loc[i,j] + ')$' 
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
        s = '|  |'+'|   '.join([str(s) for s in df.columns])+'  |\n|-\n'
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns: # Point estimates
                if tdf is not False and tdf is not None:
                    try:
                        stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                        stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                        stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                        if stars>0:
                            stars='^{'+'*'*stars + '}'
                        else: stars=''
                    except KeyError: stars=''
                else: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n|'
            for j in df.columns: # Now confidence intervals
                s+='  '
                try:
                    ci='$[' + float_fmt +','+ float_fmt + ']$'
                    ci= ci % (conf_ints[0].loc[i,j],conf_ints[1].loc[i,j])
                except KeyError: ci=''
                entry='| '+ci+'  '
                s+=entry 
            s+='|\n'
        return s

def orgtbl_to_df(table, col_name_size=1, format_string=None, index=None):
  """
  Returns a pandas dataframe.
  Requires the use of the header `:colnames no` for preservation of original column names.
  `table` is an org table which is just a list of lists in python.
  `col_name_size` is the number of rows that make up the column names.
  `format_string` is a format string to make the desired column names.
  `index` is a column label or a list of column labels to be set as the index of the dataframe.
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

def balance_panel(df):
    """Drop households that aren't observed in all rounds."""
    pnl=df.to_panel()
    keep=pnl.loc[list(pnl.items)[0],:,:].dropna(how='any',axis=1).iloc[0,:]
    df=pnl.loc[:,:,keep.index].to_frame(filter_observations=False)
    df.index.names=pd.core.base.FrozenList(['Year','HH'])

    return df

def drop_missing(X):
    """
    Return tuple of pd.DataFrames in X with any 
    missing observations dropped.  Assumes common index.
    """

    foo=pd.concat(X,axis=1).dropna(how='any')
    assert len(set(foo.columns))==len(foo.columns) # Column names must be unique!

    Y=[]
    for x in X:
        Y.append(foo.loc[:,x.columns])

    return tuple(Y)

def use_indices(df,idxnames):
    return df.reset_index()[idxnames].set_index(df.index)

def broadcast_binary_op(x, op, y):
    """Perform x op y, allowing for broadcasting over a multiindex.

    Example usage: broadcast_binary_op(x,lambda x,y: x*y ,y)
    """
    x = pd.DataFrame(x.copy())
    y = pd.DataFrame(y.copy())
    if y.shape[1]==1:
        y=pd.DataFrame([y.iloc[:,0]]*x.shape[1],index=x.columns).T

    cols = list(x.columns)
    xindex = list(x.index.names)
    yindex = list(y.index.names)

    dif = list(set(xindex)-set(yindex))
    x.reset_index(dif, inplace=True)

    x=x.sortlevel()

    newdf = x.copy()

    for col in cols:
        newdf[col] = op(x[col],y[col])

    newdf = newdf.reset_index().set_index(xindex).sortlevel()
    return newdf

def arellano_robust_cov(X,u,clusterby=['t','mkt']):
    X,u = drop_missing([X,u])
    clusters = set(zip(*tuple(use_indices(u,clusterby)[i] for i in clusterby)))
    if  len(clusters)>1:
        # Take out time averages
        u=broadcast_binary_op(u,lambda x,y:x-y, u.groupby(level=clusterby).mean()).squeeze()
        X=broadcast_binary_op(X,lambda x,y:x-y, X.groupby(level=clusterby).mean()) 
        Xu=X.mul(u,axis=0)
        if len(X.shape)==1:
            XXinv=np.array([1./(X.T.dot(X))])
        else:
            XXinv=np.linalg.inv(X.T.dot(X))
        Vhat = XXinv.dot(Xu.T.dot(Xu)).dot(XXinv)
    else:
        u=u-u.mean()
        X=X-X.mean()

        Xu=X.mul(u,axis=0)
        if len(X.shape)==1:
            XXinv=np.array([1./(X.T.dot(X))])
        else:
            XXinv=np.linalg.inv(X.T.dot(X))
        Vhat = XXinv.dot(Xu.T.dot(Xu)).dot(XXinv)

    try:
        return pd.DataFrame(Vhat,index=X.columns,columns=X.columns)
    except AttributeError:
        return Vhat


def ols(x,y,return_se=True,return_v=False,return_e=False):

    x=pd.DataFrame(x) # Deal with possibility that x & y are series.
    y=pd.DataFrame(y)
    N,n=y.shape
    k=x.shape[1]

    # Drop any observations that have missing data in *either* x or y.
    x,y = drop_missing([x,y]) 

    b=np.linalg.lstsq(x,y)[0]

    b=pd.DataFrame(b,index=x.columns,columns=y.columns)

    out=[b.T]
    if return_se or return_v or return_e:

        u=y-x.dot(b)

        # Use SUR structure if multiple equations; otherwise OLS.
        # Only using diagonal of this, for reasons related to memory.  
        S=sparse.dia_matrix((sparse.kron(u.T.dot(u),sparse.eye(N)).diagonal(),[0]),shape=(N*n,)*2) 

        if return_se or return_v:

            # This will be a very large matrix!  Use sparse types
            V=sparse.kron(sparse.eye(n),(x.T.dot(x).dot(x.T)).as_matrix().view(type=np.matrix).I).T
            V=V.dot(S).dot(V.T)

        if return_se:
            se=np.sqrt(V.diagonal()).reshape((x.shape[1],y.shape[1]))
            se=pd.DataFrame(se,index=x.columns,columns=y.columns)

            out.append(se)
        if return_v:
            # Extract blocks along diagonal; return an Nxkxn array
            V={y.columns[i]:pd.DataFrame(V[i*k:(i+1)*k,i*k:(i+1)*k],index=x.columns,columns=x.columns) for i in range(n)} 
            out.append(V)
        if return_e:
            out.append(u)
    return tuple(out)

def estimate_reduced_form(y,z,return_v=False,return_se=False,VERBOSE=False):
    """Estimate reduced-form Frisch expenditure/demand system.

    Inputs:
       - y : pd.DataFrame of log expenditures or log quantities, indexed by (j,t,mkt), 
             where j indexes the household, t the period, and mkt the market.  
             Columns are different expenditure items.

       - z : pd.DataFrame of household characteristics; index should match that of y.

    Ethan Ligon                                            February 2017
    """
    assert(y.index.names==['j','t','mkt'])
    assert(z.index.names==['j','t','mkt'])

    periods = list(set(y.index.get_level_values('t')))
    mkts = list(set(y.index.get_level_values('mkt')))

    # Time-market dummies
    DateLocD = use_indices(y,['t','mkt'])
    DateLocD = pd.get_dummies(zip(DateLocD['t'],DateLocD['mkt']))
    DateLocD.index = y.index

    sed = pd.DataFrame(columns=y.columns)
    a = pd.Series(index=y.columns)
    b = OrderedDict() #pd.DataFrame(index=y.columns)
    d = OrderedDict() #pd.DataFrame(index=y.columns,columns=z.columns).T
    ce = pd.DataFrame(index=y.index,columns=y.columns)
    V = pd.Panel(items=y.columns,major_axis=z.columns,minor_axis=z.columns)

    for i,Item in enumerate(y.columns):
        if VERBOSE: print(Item)

        lhs,rhs=drop_missing([y.iloc[:,[i]],pd.concat([z,DateLocD],axis=1)])

        # Calculate deviations
        lhsbar=lhs.mean(axis=0)
        assert ~np.any(np.isnan(lhsbar)), "Missing data in lhs?"
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

        # Estimate d & b
        myb,mye=ols(X,lhs.append(ynil),return_se=False,return_v=False,return_e=True) # Need version of pandas >0.14.0 (?) for this use of join
        ce[Item]=mye.iloc[:-1,:] # Drop constraint that sums time-effects to zero

        if return_v or return_se:
            V[Item]=arellano_robust_cov(z,mye)
            sed[Item]=pd.Series(np.sqrt(np.diag(V[Item])), index=z.columns) # reduced form se on characteristics

        #d[Item]=myb.iloc[:,:z.shape[1]].as_matrix()[0] # reduced form coefficients on characteristics
        d[Item]=myb[z.columns] # reduced form coefficients on characteristics

        b[Item] = myb[DateLocD.columns].squeeze()  # Terms involving prices
        a[Item] = y[Item].mean() - d[Item].dot(z.mean(axis=0)) - b[Item].dot(DateLocD.mean().values)

    b = pd.DataFrame(b)
    b.index=pd.MultiIndex.from_tuples(b.index,names=['t','mkt'])
    b = b.T

    d = pd.concat(d.values())

    out = [b.add(a,axis=0),ce,d]
    if return_se:
        out += [sed]
    if return_v:
        out += [V]
    return out
import pandas as pd
from scipy.stats.distributions import lognorm
import numpy as np

def geometric_brownian(sigma=1.):
    return lognorm(s=sigma,scale=np.exp(-(sigma**2)/2))

def lambdabar(T,Fbar):
    return np.cumprod(Fbar.rvs(size=(T,1)),axis=0)

def lambdas(T,N,G0=lognorm(.5),Fbar=geometric_brownian(.1),F=geometric_brownian(.2)):

    L0=G0.rvs(size=(1,N))  # Initial lambdas
    innov=F.rvs(size=(T-1,N))
    L=np.cumprod(np.r_[L0,innov],axis=0)
    
    # Add aggregate shocks L0:
    return L*lambdabar(T,Fbar=Fbar) #lambdas
prices = lambda T,n : lambdas(T,n,Fbar=geometric_brownian(.05),F=geometric_brownian(0.2)) # prices
characteristics = lambda T,N : lambdas(T,N,Fbar=geometric_brownian(.05),F=geometric_brownian(0.1)) # characteristics

import pandas as pd
from scipy.stats import distributions
import numpy as np

def measurement_error(T,N,n,mu_phi=0.,sigma_phi=0.1,mu_eps=0.,sigma_eps=1.):

    def additive_error(T=T,N=N,n=n,sigma=sigma_phi):
        return distributions.norm.rvs(scale=sigma,size=(T,N,n)) + mu_phi

    def multiplicative_error(T=T,N=N,n=n,sigma=sigma_eps):
        return np.exp(distributions.norm.rvs(loc=-sigma/2.,scale=sigma,size=(T,N,n)) + mu_eps)

    phi=additive_error(T,N,n,sigma=sigma_phi)
    eps=multiplicative_error(T,N,n,sigma=sigma_eps)

    return phi,eps

def expenditures(T,N,M,n,beta,mu_phi=0,sigma_phi=0.1,mu_eps=0,sigma_eps=1.):

    if len(beta.shape)<2:
        Theta=np.matrix(np.diag(beta))
    else:
        Theta=np.matrix(beta)
        beta=Theta.sum(axis=0).A # Row sum of elasticity matrix

    l=lambdas(T,N)
    dz=np.c_[characteristics(T,N), characteristics(T,N)]
    L=np.reshape(l,(T,N,1)) 
    p=prices(T,n)

    x=np.exp(np.kron(np.log(L),-beta) + (np.log(p)*(np.eye(n)-Theta)).A.reshape((T,1,n)) + np.tile(np.log(dz).sum(axis=0).reshape((T,N,1)),(1,1,n)))

    phi,e=measurement_error(T,N,n,mu_phi=mu_phi,sigma_phi=sigma_phi,mu_eps=mu_eps,sigma_eps=sigma_eps)
    
    x=(x+p.reshape(T,1,n)*phi) # Additive error
    x=x*e # Multiplicative error

    x=x*(x>0) # Truncation

    x=pd.Panel(x.T,items=['x%d' % i for i in range(n)]).to_frame()
    x.index.set_names(['j','t'],inplace=True)

    dz=pd.DataFrame(pd.DataFrame(dz).T.stack(),index=x.index,columns=['z%d' % i for i in range(dz.shape[0])])
    l=pd.DataFrame(pd.DataFrame(l).T.stack(),index=x.index)
    p=pd.DataFrame(p,columns=x.columns,index=x.index.levels[1])

    return x,{'beta':beta,'lambdas':l,'characteristics':dz,'prices':p}

def artificial_data(T=2,N=120,M=1,n=4,sigma_e=0.001):

    # truth=(beta,lambdas,characteristics,prices)
    x,truth=expenditures(T,N,M,n,beta=np.linspace(1,3,n),sigma_eps=sigma_e)

    y=np.log(x)

    return y,truth

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
import pandas as pd

def get_loglambdas(e,TEST=False,time_index='t',max_rank=1,min_obs=None):
    """
    Use singular-value decomposition to compute loglambdas and price elasticities,
    up to an unknown factor of proportionality phi.

    Input e is the residual from a regression of log expenditures purged
    of the effects of prices and household characteristics.   The residuals
    should be arranged as a matrix, with columns corresponding to goods. 
    """ 
    assert(e.shape[0]>e.shape[1]) # Fewer goods than observations

    chat = svd_rank1_approximation_with_missing_data(e,VERBOSE=False,max_rank=max_rank,min_obs=min_obs)

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
        #print "blogL norm: %f" % np.linalg.norm(foo-chat)

    return bphi,loglambdas

def iqr(x):
    """The interquartile range of a pd.Series of observations x."""
    import numpy as np
    return x.quantile([0.25,0.75]).diff().iloc[1]

def bootstrap_elasticity_stderrs(e,tol=1e-3,minits=30,return_samples=False,VERBOSE=False,outfn=None,TRIM=True):
    """Bootstrap estimates of standard errors for \phi\beta.

    Takes pd.DataFrame of residuals as input.

    If optional parameter TRIM is True, then calculations are
    performed using the interquartile range (IQR) instead of the
    standard deviation, with the standard deviation computed as
    IQR*0.7416 (which is a good approximation provided the
    distribution is normal).

    Ethan Ligon                              January 2017
    """
    bhat,Lhat=get_loglambdas(e)

    if outfn: outf=open(outfn,'a')

    delta=1.
    old=np.array(1)
    new=np.array(0)
    i=1
    L=[]
    while delta>tol or i < minits:
        delta=np.nanmax(np.abs(old.reshape(-1)-new.reshape(-1)))
        if VERBOSE and (i % 2)==0 and i>2: 
            print "Iteration %d, delta=%5.4f.  Measure of non-normality %6.5f." % (i, delta,np.nanmax(np.abs(std0.reshape(-1)-std1.reshape(-1))))
        old=new
        S=e.iloc[np.random.random_integers(0,e.shape[0]-1,size=e.shape[0]),:]
        S=S-S.mean() 

        bs,ls=get_loglambdas(S)
        assert(~np.any(np.isnan(bs)))
        try:
            B=B.append(bs,ignore_index=True)
        except NameError:
            B=pd.DataFrame(bs).T # Create B

        L.append(ls)

        std0=B.std()
        std1=B.apply(iqr)*0.7416 # Estimate of standard deviation, with trimming
        if TRIM:
            new=std1
        else:
            new=std0

        if outfn: outf.write(','.join(['%6.5f' % b for b in bs])+'\n')
        i+=1

    if outfn: outf.close()
    if return_samples:
        return new,B
    else:
        return new
import numpy as np
from scipy import sparse

def df_norm(a,b=None,ignore_nan=True):
    a=a.copy()
    if not b is None:
      b=b.copy()
    else:
      b=pd.DataFrame(np.zeros(a.shape),columns=a.columns,index=a.index)

    if ignore_nan:
        missing=(a.isnull()+0.).replace([1],[np.NaN]) +  (b.isnull()+0.).replace([1],[np.NaN]) 
        a=a+missing
        b=b+missing
    return np.linalg.norm(a.fillna(0).as_matrix() - b.fillna(0).as_matrix())

def df_to_orgtbl(df,tdf=None,sedf=None,conf_ints=None,float_fmt='%5.3f'):
    """
    Returns a pd.DataFrame in format which forms an org-table in an emacs buffer.
    Note that headers for code block should include ":results table raw".

    Optional inputs include conf_ints, a pair (lowerdf,upperdf).  If supplied, 
    confidence intervals will be printed in brackets below the point estimate.

    If conf_ints is /not/ supplied but sedf is, then standard errors will be 
    in parentheses below the point estimate.

    If tdf is False and sedf is supplied then stars will decorate significant point estimates.
    If tdf is a df of t-statistics stars will decorate significant point estimates.
    """
    if len(df.shape)==1: # We have a series?
       df=pd.DataFrame(df)

    if (tdf is None) and (sedf is None) and (conf_ints is None):
        return '|'+df.to_csv(sep='|',float_format=float_fmt,line_terminator='|\n|')
    elif not (tdf is None) and (sedf is None) and (conf_ints is None):
        s = '|  |'+'|   '.join([str(s) for s in df.columns])+'\t|\n|-\n'
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns:
                try:
                    stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                    stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                    if stars>0:
                        stars='^{'+'*'*stars + '}'
                    else: stars=''
                except KeyError: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n'

        return s
    elif not (sedf is None) and (conf_ints is None): # Print standard errors on alternate rows
        if tdf is not False:
            try: # Passed in dataframe?
                tdf.shape
            except AttributeError:  
                tdf=df[sedf.columns]/sedf
        s = '|  |'+'|   '.join([str(s) for s in df.columns])+'  |\n|-\n'
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns: # Point estimates
                if tdf is not False:
                    try:
                        stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                        stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                        stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                        if stars>0:
                            stars='^{'+'*'*stars + '}'
                        else: stars=''
                    except KeyError: stars=''
                else: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n|'
            for j in df.columns: # Now standard errors
                s+='  '
                try:
                    se='$(' + float_fmt % sedf.loc[i,j] + ')$' 
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
        s = '|  |'+'|   '.join([str(s) for s in df.columns])+'  |\n|-\n'
        for i in df.index:
            s+='| %s  ' % i
            for j in df.columns: # Point estimates
                if tdf is not False and tdf is not None:
                    try:
                        stars=(np.abs(tdf.loc[i,j])>1.65) + 0.
                        stars+=(np.abs(tdf.loc[i,j])>1.96) + 0.
                        stars+=(np.abs(tdf.loc[i,j])>2.577) + 0.
                        if stars>0:
                            stars='^{'+'*'*stars + '}'
                        else: stars=''
                    except KeyError: stars=''
                else: stars=''
                entry='| $'+float_fmt+stars+'$  '
                s+=entry % df.loc[i,j]
            s+='|\n|'
            for j in df.columns: # Now confidence intervals
                s+='  '
                try:
                    ci='$[' + float_fmt +','+ float_fmt + ']$'
                    ci= ci % (conf_ints[0].loc[i,j],conf_ints[1].loc[i,j])
                except KeyError: ci=''
                entry='| '+ci+'  '
                s+=entry 
            s+='|\n'
        return s

def orgtbl_to_df(table, col_name_size=1, format_string=None, index=None):
  """
  Returns a pandas dataframe.
  Requires the use of the header `:colnames no` for preservation of original column names.
  `table` is an org table which is just a list of lists in python.
  `col_name_size` is the number of rows that make up the column names.
  `format_string` is a format string to make the desired column names.
  `index` is a column label or a list of column labels to be set as the index of the dataframe.
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

def balance_panel(df):
    """Drop households that aren't observed in all rounds."""
    pnl=df.to_panel()
    keep=pnl.loc[list(pnl.items)[0],:,:].dropna(how='any',axis=1).iloc[0,:]
    df=pnl.loc[:,:,keep.index].to_frame(filter_observations=False)
    df.index.names=pd.core.base.FrozenList(['Year','HH'])

    return df

def drop_missing(X):
    """
    Return tuple of pd.DataFrames in X with any 
    missing observations dropped.  Assumes common index.
    """

    foo=pd.concat(X,axis=1).dropna(how='any')
    assert len(set(foo.columns))==len(foo.columns) # Column names must be unique!

    Y=[]
    for x in X:
        Y.append(foo.loc[:,x.columns])

    return tuple(Y)

def use_indices(df,idxnames):
    return df.reset_index()[idxnames].set_index(df.index)

def broadcast_binary_op(x, op, y):
    """Perform x op y, allowing for broadcasting over a multiindex.

    Example usage: broadcast_binary_op(x,lambda x,y: x*y ,y)
    """
    x = pd.DataFrame(x.copy())
    y = pd.DataFrame(y.copy())
    if y.shape[1]==1:
        y=pd.DataFrame([y.iloc[:,0]]*x.shape[1],index=x.columns).T

    cols = list(x.columns)
    xindex = list(x.index.names)
    yindex = list(y.index.names)

    dif = list(set(xindex)-set(yindex))
    x.reset_index(dif, inplace=True)

    x=x.sortlevel()

    newdf = x.copy()

    for col in cols:
        newdf[col] = op(x[col],y[col])

    newdf = newdf.reset_index().set_index(xindex).sortlevel()
    return newdf

def arellano_robust_cov(X,u,clusterby=['t','mkt']):
    X,u = drop_missing([X,u])
    clusters = set(zip(*tuple(use_indices(u,clusterby)[i] for i in clusterby)))
    if  len(clusters)>1:
        # Take out time averages
        u=broadcast_binary_op(u,lambda x,y:x-y, u.groupby(level=clusterby).mean()).squeeze()
        X=broadcast_binary_op(X,lambda x,y:x-y, X.groupby(level=clusterby).mean()) 
        Xu=X.mul(u,axis=0)
        if len(X.shape)==1:
            XXinv=np.array([1./(X.T.dot(X))])
        else:
            XXinv=np.linalg.inv(X.T.dot(X))
        Vhat = XXinv.dot(Xu.T.dot(Xu)).dot(XXinv)
    else:
        u=u-u.mean()
        X=X-X.mean()

        Xu=X.mul(u,axis=0)
        if len(X.shape)==1:
            XXinv=np.array([1./(X.T.dot(X))])
        else:
            XXinv=np.linalg.inv(X.T.dot(X))
        Vhat = XXinv.dot(Xu.T.dot(Xu)).dot(XXinv)

    try:
        return pd.DataFrame(Vhat,index=X.columns,columns=X.columns)
    except AttributeError:
        return Vhat


def ols(x,y,return_se=True,return_v=False,return_e=False):

    x=pd.DataFrame(x) # Deal with possibility that x & y are series.
    y=pd.DataFrame(y)
    N,n=y.shape
    k=x.shape[1]

    # Drop any observations that have missing data in *either* x or y.
    x,y = drop_missing([x,y]) 

    b=np.linalg.lstsq(x,y)[0]

    b=pd.DataFrame(b,index=x.columns,columns=y.columns)

    out=[b.T]
    if return_se or return_v or return_e:

        u=y-x.dot(b)

        # Use SUR structure if multiple equations; otherwise OLS.
        # Only using diagonal of this, for reasons related to memory.  
        S=sparse.dia_matrix((sparse.kron(u.T.dot(u),sparse.eye(N)).diagonal(),[0]),shape=(N*n,)*2) 

        if return_se or return_v:

            # This will be a very large matrix!  Use sparse types
            V=sparse.kron(sparse.eye(n),(x.T.dot(x).dot(x.T)).as_matrix().view(type=np.matrix).I).T
            V=V.dot(S).dot(V.T)

        if return_se:
            se=np.sqrt(V.diagonal()).reshape((x.shape[1],y.shape[1]))
            se=pd.DataFrame(se,index=x.columns,columns=y.columns)

            out.append(se)
        if return_v:
            # Extract blocks along diagonal; return an Nxkxn array
            V={y.columns[i]:pd.DataFrame(V[i*k:(i+1)*k,i*k:(i+1)*k],index=x.columns,columns=x.columns) for i in range(n)} 
            out.append(V)
        if return_e:
            out.append(u)
    return tuple(out)

y,truth=artificial_data(T=2,N=50,M=2,n=5,sigma_e=1e-8)

y['mkt']=1
y=y.reset_index().set_index(['j','t','mkt'])

#beta,L,dz,p=truth
dz=truth['characteristics']
dz['mkt']=1
dz=dz.reset_index().set_index(['j','t','mkt'])
dz=np.log(dz)

numeraire=None #'x0'

# Try with missing data for contrast
y.as_matrix()[np.random.random_sample(y.shape)<0.0]=np.nan

y.replace(-np.inf,np.nan,inplace=True)

#b,ce,d,V=estimate_bdce_with_missing_values(y,dz,return_v=True)
b,ce,d=estimate_reduced_form(y,dz,return_v=False)

bphi,logL=get_loglambdas(ce,TEST=True)
cehat=np.outer(pd.DataFrame(bphi),pd.DataFrame(-logL).T).T
cehat=pd.DataFrame(cehat,columns=bphi.index,index=logL.index)

print "Norm of error in approximation of CE: %f" % df_norm(cehat,ce)

# Some naive standard errors

#yhat=b.T.add(cehat + (dz.dot(d.T)),axis=0,level='t')
yhat = broadcast_binary_op(cehat + dz.dot(d.T),lambda x,y: x+y,b.T)

e=y.sub(yhat)

print "Correlation of log lambda with estimate (before normalization): %f" % pd.DataFrame({"L0":np.log(truth['lambdas'][0]),"Lhat":logL}).corr().iloc[0,1]

if not numeraire is None:
    logL=broadcast_binary_op(logL,lambda x,y: x+y,b.loc[numeraire]) # Add term associated with numeraire good
    b=b-b.loc[numeraire]
else:
    logL=broadcast_binary_op(logL,lambda x,y: x+y,b.mean()) # Add term associated with numeraire good
    b=b-b.mean()

# Evaluate estimate of beta:
print "Norm of (bphi,beta): %f" % np.var(bphi/truth['beta']) # Funny norm deals with fact that b only identified up to a scalar

foo=logL.reset_index('mkt')
foo['loglambda0']=np.log(truth['lambdas'][0])
foo=foo.reset_index().set_index(['j','t','mkt'])
print "Correlation of log lambda with estimate (after normalization):"
print foo.groupby(level=['t','mkt']).corr()

print "Mean of errors:"
print e.mean(axis=0)

# test_estimate_with_missing ends here
