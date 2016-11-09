
# [[file:~/Dropbox/0Fall2016/vesdemand/Empirics/neediness.org::*Utility%20Functions][df_utils]]

# Tangled on Mon Nov  7 19:02:06 2016
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

# df_utils ends here
