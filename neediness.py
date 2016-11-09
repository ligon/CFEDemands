
# [[file:~/Dropbox/0Fall2016/vesdemand/Empirics/neediness.org::*Expenditure%20Shares][agg_shares_and_mean_shares]]

# Tangled on Mon Nov  7 19:02:06 2016
import pylab as pl 
def expenditure_shares(df):

    aggshares=df.groupby(level='t').sum()
    aggshares=aggshares.div(aggshares.sum(axis=1),axis=0).T
    meanshares=df.div(df.sum(axis=1),level='j',axis=0).groupby(level='t').mean().T

    mratio=(np.log(meanshares)-np.log(aggshares))
    sharesdf=pd.Panel({'Mean shares':meanshares,'Agg. shares':aggshares})

    return sharesdf,mratio

def agg_shares_and_mean_shares(df,figname=None,ConfidenceIntervals=False,ax=None):
    """Figure of log mean shares - log agg shares.

    Input df is a pd.DataFrame of expenditures, ordered by (t,j).

    ConfidenceIntervalues is an optional argument.  
    If True, the returned figure will have 95% confidence intervals.  
    If a float in (0,1) that will be used for the size of the confidence 
    interval instead.
    """

    shares,mratio=expenditure_shares(df)
    meanshares=shares['Mean shares']

    tab=shares.to_frame().unstack()
    tab.sort_values(by=('Agg. shares',meanshares.columns[0]),ascending=False,inplace=True)

    if ax is None:
        fig, ax = pl.subplots()

    mratio.sort_values(by=mratio.columns[0],inplace=True)
    ax.plot(range(mratio.shape[0]),mratio.as_matrix(), 'o')
    ax.legend(mratio.columns,loc=2)
    ax.set_ylabel('Log Mean shares divided by Aggregate shares')

    v=ax.axis()
    i=0
    for i in range(len(mratio)):
        name=mratio.ix[i].name # label of expenditure item

        if mratio.iloc[i,0]>0.2:
            #pl.text(i,mratio.T.iloc[0][name],name,fontsize='xx-small',ha='right')

            # The key option here is `bbox`. 
            ax.annotate(name, xy=(i,mratio.T.iloc[0][name]), xytext=(-20,10), 
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.25', 
                        color='red'),fontsize='xx-small')

        if mratio.iloc[i,0]<-0.2:
            #pl.text(i,mratio.T.iloc[0][name],name,fontsize='xx-small')
            ax.annotate(name, xy=(i,mratio.T.iloc[0][name]), xytext=(20,-10), 
                        textcoords='offset points', ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.25', 
                        color='red'),fontsize='xx-small')


    if ConfidenceIntervals>0: # Bootstrap some confidence intervals
        if ConfidenceIntervals==1: ConfidenceIntervals=0.95
        current=0
        last=1
        M=np.array([],ndmin=3).reshape((mratio.shape[0],mratio.shape[1],0))
        i=0
        mydf=df.loc[:,mratio.index]
        while np.max(np.abs(current-last))>0.001 or i < 1000:
            last=current
            # Sample households in each  round with replacement
            bootdf=mydf.iloc[np.random.random_integers(0,df.shape[0]-1,df.shape[0]),:]
            bootdf.reset_index(inplace=True)
            bootdf['HH']=range(bootdf.shape[0])
            bootdf.set_index(['Year','HH'],inplace=True)
            shares,mr=expenditure_shares(bootdf)
            M=np.dstack((M,mr.as_matrix()))
            M.sort(axis=2)
            a=(1-ConfidenceIntervals)/2.
            lb= mratio.as_matrix() - M[:,:,np.floor(M.shape[-1]*a)]
            ub=M[:,:,np.floor(M.shape[-1]*(ConfidenceIntervals+a))] - mratio.as_matrix()
            current=np.c_[lb,ub]
            i+=1
        T=mratio.shape[1]
        for t in range(T):
            ax.errorbar(np.arange(mratio.shape[0]),mratio.as_matrix()[:,t],yerr=current[:,[t,t-T]].T.tolist())
            tab[(df.index.levels[0][t],'Upper Int')]=current[:,t-T]
            tab[(df.index.levels[0][t],'Lower Int')]=current[:,t]

    ax.axhline()

    if figname:
        pl.savefig(figname)

    return tab,ax

# agg_shares_and_mean_shares ends here

# [[file:~/Dropbox/0Fall2016/vesdemand/Empirics/neediness.org::*Expenditure%20Shares][group_expenditures]]

# Tangled on Mon Nov  7 19:02:06 2016
def group_expenditures(df,groups):
    myX=pd.DataFrame(index=df.index)
    for k,v in groups.iteritems():
        myX[k]=df[['$x_{%d}$' % i for i in v]].sum(axis=1)
            
    return myX

# group_expenditures ends here

# [[file:~/Dropbox/0Fall2016/vesdemand/Empirics/neediness.org::*Brand%20(2006)%20algorithm%20for%20iterative%20rank-1%20SVD%20updates][brand2006]]

# Tangled on Mon Nov  7 19:02:06 2016
"""
Adapted from code in version 0.7.4 of gensim
(https://pypi.python.org/pypi/gensim/0.7.4).  That code is licensed
under the LGPL (http://www.gnu.org/licenses/lgpl.html).  I assert that
this notice satisfies the requirements imposed on on this work
enumerated in Section 5 ("Combined Libraries") of the LGPL.
""" 

import logging 
import numpy
np=numpy

logger = logging.getLogger('lsimodel')
logger.setLevel(logging.INFO)



def svdUpdate(U, S, V, a, b):
    """
    Update SVD of an (m x n) matrix `X = U * S * V^T` so that
    `[X + a * b^T] = U' * S' * V'^T`
    and return `U'`, `S'`, `V'`.

    The original matrix X is not needed at all, so this function implements one-pass
    streaming rank-1 updates to an existing decomposition. 

    `a` and `b` are (m, 1) and (n, 1) matrices.

    You can set V to None if you're not interested in the right singular
    vectors. In that case, the returned V' will also be None (saves memory).

      This is the rank-1 update as described in
    **Brand, 2006: Fast low-rank modifications of the thin singular value decomposition**,
    but without separating the basis from rotations.
    """

    def fixmiss(c,S,U):
        """Interpolate to deal with missing values in vector c."""

        miss=np.isnan(c).nonzero()[0]
        nonmiss=(~np.isnan(c)).nonzero()[0]

        B=S*np.linalg.pinv(U[nonmiss,:]*S)*c[nonmiss]
        p = c[nonmiss] - U[nonmiss,:]*B
        Ra = np.linalg.norm(p)
        if len(miss)>0:
          chat=U[miss,:]*S*B
        else:
          chat = c

    return chat,p,Ra

    # convert input to matrices (no copies of data made if already numpy.ndarray or numpy.matrix)
    S = numpy.asmatrix(S)
    U = numpy.asmatrix(U)
    if V is not None:
        V = numpy.asmatrix(V)

  
    b = numpy.asmatrix(b).reshape(b.size, 1)

    rank = S.shape[0]

    # eq (6)
    a,p,Ra = fixmiss(numpy.asmatrix(a).reshape(a.size, 1),S,U)
    #m = U.T * a   # These are for the non-missing case
    #p = a - U * m
    #Ra = numpy.sqrt(p.T * p)
    if float(Ra) < 1e-10:
        logger.debug("input already contained in a subspace of U; skipping update")
        return U, S, V
    P = (1.0 / float(Ra)) * p

    if V is not None:
        # eq (7)
        n = V.T * b
        q = b - V * n
        Rb = numpy.sqrt(q.T * q)
        if float(Rb) < 1e-10:
            logger.debug("input already contained in a subspace of V; skipping update")
            return U, S, V
        Q = (1.0 / float(Rb)) * q
    else:
        n = numpy.matrix(numpy.zeros((rank, 1)))
        Rb = numpy.matrix([[1.0]])    

    if float(Ra) > 1.0 or float(Rb) > 1.0:
        logger.debug("insufficient target rank (Ra=%.3f, Rb=%.3f); this update will result in major loss of information"
                      % (float(Ra), float(Rb)))

    # eq (8)
    K = numpy.matrix(numpy.diag(list(numpy.diag(S)) + [0.0])) + numpy.bmat('m ; Ra') * numpy.bmat('n ; Rb').T

    # eq (5)
    u, s, vt = numpy.linalg.svd(K, full_matrices = False)
    tUp = numpy.matrix(u[:, :rank])
    tVp = numpy.matrix(vt.T[:, :rank])
    tSp = numpy.matrix(numpy.diag(s[: rank]))
    Up = numpy.bmat('U P') * tUp
    if V is not None:
        Vp = numpy.bmat('V Q') * tVp
    else:
        Vp = None
    Sp = tSp

    return Up, Sp, Vp

# brand2006 ends here

# [[file:~/Dropbox/0Fall2016/vesdemand/Empirics/neediness.org::*Rank%201%20SVD%20Approximation%20to%20Matrix%20with%20Missing%20Data][svd_rank1_approximation_with_missing_data]]

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

# svd_rank1_approximation_with_missing_data ends here

# [[file:~/Dropbox/0Fall2016/vesdemand/Empirics/neediness.org::*Estimation%20of%20reduced%20form][estimate_reduced_form]]

# Tangled on Mon Nov  7 19:02:09 2016
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
    Print(pd.DataFrame in format which forms an org-table.)
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
        if tdf is not False: tdf=df[sedf.columns]/sedf
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
        if tdf is not False and sedf is not None: tdf=df[sedf.columns]/sedf
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

def arellano_robust_cov(X,u):
    rounds=u.index.get_level_values(1).unique() # Periods to cluster by
    if  len(rounds)>1:
        u=u.sub(u.groupby(level='t').mean()) # Take out time averages
        X.sub(X.groupby(level='t').mean())
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
def estimate_bdce_with_missing_values(y,z,market=None,prices=None,return_v=False,return_se=False, time_index=1):
    """Estimate reduced form objects b, d, and ce.  

    Inputs are log expenditures and household characteristics (both in
    logs).  Both must be pd.DataFrames.

    The optional variable market is a series which identifies locations
    (e.g, rural/urban)  which may be thought to have different prices.
    In this case different latent price variables are estimated for
    different regions. 

    The optional variable prices is a df of prices for (possibly
    selected) goods.  Where supplied these (logged) price data will be
    used in lieu of a latent variable approach.

    Ethan Ligon                                            April,  2016
    ELLIOIT (2016-11-07): time_index indicates order in multi-index of time axis. Also used to get index name.
    """
    n,N,T=y.to_panel().shape

    b=OrderedDict()
    d=OrderedDict()
    a=OrderedDict()
    myE=OrderedDict()
    sed=OrderedDict()
    V=OrderedDict()
    t = y.index.names[time_index]

    years=[year for year in y.index.levels[time_index]]

    Timed=pd.get_dummies(use_indices(z,[t])[t])

    for i in range(n):
        myy,myz=drop_missing([y.iloc[:,[i]],z])
        # Calculate a within transformation
        Wy=myy-myy.mean()
        Wy=Wy-Wy.mean()

        Wz=myz-myz.mean(axis=0)
        Wz=Wz-Wz.mean(axis=0)

        #~ ELLIOTT: (2016-11-09) moved USE_PRICE to avoid making year/market dummies when prices are provided instead
        USE_PRICE=(prices is not None) and (y.columns[i] in prices.index)
        if USE_PRICE:
            timed=np.log(prices.iloc[i,:])
        elif not market is None:
            #~ ELLIOTT (2016-11-07): original function used tuples for market-time labels. Now using strings.
            foo=pd.Series(["-".join(map(str,tuple(x))) for x in pd.concat([use_indices(Wz,[t])[t],market],axis=1,join='inner').as_matrix().tolist()],index=Wz.index,name=t+"-"+market.name)
            timed=pd.get_dummies(foo)
        else:
            timed=pd.get_dummies(use_indices(Wz,[t])[t])

        years = [x for x in timed.columns]

        Wtimed=timed-timed.mean() # Don't forget within transformation of time dummies! 
        Wtimed=Wtimed-Wtimed.mean()  # First de-meaning can be improved upon

        print(y.columns[i])
        if not USE_PRICE:
            # Need to make sure time-market effects sum to zero
            ynil=pd.DataFrame([0],index=[(-1,0)],columns=Wy.columns)
            znil=pd.DataFrame([[0]*Wz.shape[1]],index=[(-1,0)],columns=Wz.columns)
            timednil=pd.DataFrame([[1]*timed.shape[1]],index=[(-1,0)],columns=timed.columns)

            X=Wz.append(znil).join(Wtimed.append(timednil))
            # Estimate d & b
            myb,mye=ols(X,Wy.append(ynil),return_se=False,return_v=False,return_e=True) # Need version of pandas >0.14.0 (?) for this use of join
        else:
            X=Wz.join(Wtimed)
            myb,mye=ols(X,Wy,return_se=False,return_v=False,return_e=True) # Need version of pandas >0.14.0 (?) for this use of join

        #mye=mye.iloc[:-1,:] # Drop constraint
        if return_v or return_se:
            myV=arellano_robust_cov(X,mye.iloc[:,0])
            #~ ELLIOTT (2016-11-07) Changed to series to allow for different lengths (comes up with market FE's).
            myse= pd.Series(np.sqrt(np.diag(myV)), index=X.columns)


        for year in years:
            if year not in myb.columns:
                myb[year]=np.NaN 

        myb=myb[z.columns.tolist()+years] #~ Numpy returns error if years list contains tuples. Switch to strings, then back at end.

        d[y.columns[i]]=myb.iloc[:,:Wz.shape[1]].as_matrix()[0] # reduced form coefficients on characteristics
        if return_se: # Get std. errs for characteristics
            sed[y.columns[i]]=myse # reduced form se on characteristics

        #~ Get b as series instead of matrix to allow for different lengths (markets with no HH's consuming item are dropped from regression)
        b[y.columns[i]]=myb.iloc[:,Wz.shape[1]:].T[y.columns[i]] #.as_matrix()[0] # Terms involving prices
        a[y.columns[i]] = (myy.mean() - d[y.columns[i]].dot(myz.mean(axis=0)) - b[y.columns[i]].dot(timed.mean().as_matrix())).as_matrix()[0]

        #myce[y.columns[i]] = pd.Series((myy - a[y.columns[i]]).as_matrix().reshape(-1) - myz.as_matrix().dot(d[y.columns[i]]) - timed.as_matrix().dot(b[y.columns[i]]),index=myy.index)
        myE[y.columns[i]] = mye.iloc[:-1,:]  # Drop constraint
        V[y.columns[i]] = myV


    d=pd.DataFrame(d,index=z.columns).T

    if return_se: # Get std. errs for characteristics
        sed=pd.DataFrame(sed,index=X.columns).T[z.columns]

    if not market is None: #~ ELLIOTT (2016-11-07): Split stringified time-market index into multi-index.
        b=pd.DataFrame(b) #~ ELLIOTT (2016-11-07): "years" is a local list in a loop, which varies in length if consuming markets vary in number ,index=years) 
        b.index.name=t
        b = b.reset_index()
        b[market.name] = b[t].apply(lambda x: x.split("-")[time_index-1])
        try: b[market.name] =b[market.name].apply(int) 
        except ValueError: pass #~ some market names are strings.
        b[t] = b[t].apply(lambda x: x.split("-")[time_index])
        try: b[t] =b[t].apply(int) 
        except ValueError: pass #~ some period names are strings.
        b = b.set_index([t,market.name]).T
    else:
        b=pd.DataFrame(b,index=years) 
        b.index.name=t
        b.T

    a=pd.DataFrame(a,columns=y.columns,index=['Constant']).T['Constant']

    #ce0 = y - a - z.dot(d.T) - Timed.dot(b.T) #  Should be equal to ce if no prices
    ce=pd.concat(myE.values(),axis=1)

    assert np.abs(ce.unstack(t).mean()).sum() < 1e-10 #~ ELLIOTT (2016-11-07): Sure to unstack by year, as indicated in `year_index'

    out = [b.add(a,axis=0),ce,d]
    if return_se:
        out += [sed]
    if return_v:
        V = pd.Panel(V,major_axis=X.columns,minor_axis=X.columns)
        out += [V]
    return out

# estimate_reduced_form ends here

# [[file:~/Dropbox/0Fall2016/vesdemand/Empirics/neediness.org::*Extraction%20of%20Elasticities%20and%20Neediness][get_loglambdas]]

# Tangled on Mon Nov  7 19:02:09 2016
import pandas as pd

def get_loglambdas(e,TEST=False):
    """
    Use singular-value decomposition to compute loglambdas and price elasticities,
    up to an unknown factor of proportionality phi.

    Input e is the residual from a regression of log expenditures purged
    of the effects of prices and household characteristics.   The residuals
    should be arranged as a matrix, with columns corresponding to goods. 
    """ 

    assert(e.shape[0]>e.shape[1]) # Fewer goods than observations

    chat = svd_rank1_approximation_with_missing_data(e,VERBOSE=False)

    R2 = chat.var()/e.var()

    # Possible that initial elasticity b_i is negative, if inferior goods permitted.
    # But they must be positive on average.
    if chat.iloc[0,:].mean()>0:
        b=chat.iloc[0,:]
    else:
        b=-chat.iloc[0,:]

    loglambdas=(-chat.iloc[:,0]/b.iloc[0])

    # Find phi that normalizes first round loglambdas
    phi=loglambdas.groupby(level=0).std().iloc[0]
    loglambdas=loglambdas/phi

    bphi=pd.Series(b*phi,index=e.columns)

    if TEST:
        foo=-np.outer(bphi,loglambdas).T
        assert np.linalg.norm(foo-chat)<1e-4
        print("blogL norm: %f" % np.linalg.norm(foo-chat))

    return bphi,loglambdas

def bootstrap_elasticity_stderrs(e,tol=1e-4,minits=30,return_samples=False,VERBOSE=False):
    """
    Bootstrap estimates of standard errors for \phi\beta.

    Takes pd.DataFrame of residuals as input.
    """
    B=[]
    old=np.array(0)
    new=np.array(1)
    i=1
    while np.nanmean(np.abs(old.reshape(-1)-new.reshape(-1)))>tol or i < minits:
        if VERBOSE and (i % 2)==0: 
            print(i, np.nanmean(np.abs(old.reshape(-1)-new.reshape(-1))))
        old=new
        S=e.iloc[np.random.random_integers(0,e.shape[0]-1,size=e.shape[0]),:]
        B.append(get_loglambdas(S)[0])
        new=pd.concat(B,axis=1).T.std()
        i+=1

    if return_samples:
        return new,B
    else:
        return new


def alt_loglambdas(e,returnSE=False):
    """
    Use averaging to compute loglambdas and price elasticities.  

    Input e is the residual from a regression of log expenditures purged
    of the effects of prices and household characteristics.   The residuals
    should be arranged as a matrix, with columns corresponding to goods. 
    """ 

    assert(e.shape[0]>e.shape[1]) # Fewer goods than observations

    logL=-e.mean(axis=1)
    logLse=e.std(axis=1)

    logLbar=(e*0).add(logL,axis=0).mean(axis=0) # Depends on missing lambdas

    bphi=e.mean(axis=0)/logLbar

    # Recenter, and make sure sign correct:
    bbar=(e*0).add(bphi).mean(axis=1) # Depends on missing lambdas

    logL=logL*np.abs(bbar)
    #bphi=bphi/bbar.mean()
    bphi=bphi/bphi.mean()

    # Possible that initial elasticity b_i is negative, if inferior goods permitted.
    # But they must be positive on average.
    assert(np.abs(bphi.mean()-1)<1e-2)

    bphi=pd.Series(bphi,index=e.columns)

    if returnSE:
        return bphi,logL,logLse
    else:
        return bphi,logL

# get_loglambdas ends here
