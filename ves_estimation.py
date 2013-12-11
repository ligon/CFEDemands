#!/usr/bin/env python
"""
This is a module with functions which implement various aspects of estimation of VES demand systems.
"""

import pandas as pd
from numpy import linalg
import numpy as np
import pylab as pl
import sys
sys.path.append('../Computation')
import variable_elasticity_utility as ves
import engel_curves as engel

def pandas2vic(*dfs):
    """
    Transform a pandas series or dataframe to a (values,index,columns) tuple; drop any rows with NaNs.
    """
    all=pd.concat(dfs,join='inner',axis=1).dropna()

    assert len(all)>0

    return tuple([(df.loc[all.index].values,all.index,df.columns) for df in dfs])

def vic2pandas(vic):
    """Transform (value,columns,index) to pandas.DataFrame."""
    return pd.DataFrame(**vic)

def proj(y,x,returnb=False):
    """
    Linear projection of a matrix y on a matrix x.

    If y and x are pandas.DataFrames or Series, then return similar objects.
    """

    vicx,vicy=pandas2vic(x,y)

    b=linalg.lstsq(vicx[0],vicy[0])[0]

    if returnb:
        return pd.DataFrame(np.dot(vicx[0],b),index=vicx[1],columns=vicy[2]),b
    else:
        return pd.DataFrame(np.dot(vicx[0],b),index=vicx[1],columns=vicy[2])

def group_expenditures(df,groups):
    myX=pd.DataFrame(index=df.index)
    for k,v in groups.iteritems():
        try:
            myX[k]=df[list(v)].sum(axis=1)
        except IndexError:
            myX[k]=df[['$x_{%d}$' % i for i in v]].sum(axis=1)
            
    return myX
            
def order_by_expenditures(X,groups=None,Z=None,method='rank'):
    """
    Construct a wealth ordering of households based on an NxD pandas DataFrame of expenditures X.

    Each row of X corresponds to an observation of expenditures on D goods for a particular household.

    There are two methods for ordering.  The first is by 'total' expenditures;
    the second by the average 'rank' of the household in expenditures across goods.

    If the optional list of lists "groups" is provided, then expenditures are first summed across groups
    before ranking.
    
    If a DataFrame Z is also provided, then expenditures are projected onto Z, and orderings
    are based on residual expenditures instead of expenditures.

    Ethan Ligon                                                                October 2013
    """
    myX=pd.DataFrame(index=X.index)
    if groups:
        for k in range(len(groups)):
            myX['group%d'% k]=X[list(groups[k])].sum(axis=1)
    else:
        myX=X

    if Z is None:  # Use expenditures
        E=myX 
    else:  # Use residuals
        E=myX-proj(myX,Z)

    if method=='total':
        R=E.sum(axis=1).rank().argsort().argsort()
    elif method=='rank':
        R=E.rank().sum(axis=1).rank().argsort().argsort()
    else:
        raise ValueError, "Unknown ranking method"

    return (R+1.)/len(R),myX


def engel_curves(rslt,ybounds=[0,10],fname=None):
    
    return engel.plot(rslt['Price'],rslt['alpha'],rslt['gamma'],rslt['phi'],
                                 labels=rslt.index,ybounds=ybounds,fname=fname)    
    

def fake_hhsize(N,T,p0=1./3,p1=.9):
    """
    Generate time-varying household sizes for N households over T periods.

    The parameter p0 governs the initial (geometric) size distribution (smaller p0 means bigger sizes),
    while the parameter p1 governs the rate at which these household sizes evolve over time.
    """
    hhsize=np.random.geometric(p0,size=N).reshape((N,1)) # Initial household size
    X=np.c_[np.zeros((N,1)),np.arange(N).reshape((-1,1)),hhsize]
    for t in range(1,T):
        xt=hhsize + np.random.geometric(p1,size=(N,1))-np.random.geometric(p1,size=(N,1))
        hhsize=xt
        xt=np.choose(xt>0,[1,xt]) # Minimum hhsize should be 1
        xt=np.c_[t*np.ones((N,1)),np.arange(N).reshape((-1,1)),xt]
        X=np.r_[X,xt]

    return X

def fake_prices(K,T,sigma=1./4):
    """
    Generate time-varying household sizes for K goods over T periods.
    Martingale process with multiplicative log-normal innovations.
    """
    p=np.random.random(size=K).reshape((1,-1)) # Initial prices
    X=np.c_[np.zeros((1,1)),p]
    for t in range(1,T):
        pt=p*np.exp(np.random.normal(-(sigma**2/2),sigma,size=(1,K)))
        p=pt
        pt=np.c_[t*np.ones((1,1)),pt]
        X=np.r_[X,pt]

    return X

def fake_data(size=(2,100,4),alphasigma=0.1):
    """
    Generate a fake dataset for for K goods for N households over T periods.
    """
    T,N,K=size

    d={}
    x=fake_hhsize(N,T)
    d['HHSize']=x[:,-1]
    d['Year']=x[:,0]
    d['HH']=x[:,1]

    # Let alphas be a function of hhsize
    alphabar=np.random.random(size=K).reshape((1,K))*x[:,-1].reshape((-1,1))
    alpha=alphabar*np.exp(np.random.normal(-(alphasigma**2)/2.,alphasigma,size=(alphabar.shape)))

    # Generate phis from a double geometric; make proportional to household size
    phi=0.25*(np.random.geometric(2./3,size=(1,K))-np.random.geometric(2./K,size=(1,K)))*x[:,-1].reshape((-1,1))

    gamma=np.arange(1.,K+1)/K*3

    prices=fake_prices(K,T)

    ystar=np.exp(np.random.normal(10,3,size=N))
    i=0
    X=[]
    Y=[]
    for t in range(T):
        for j in range(N):
            y=ystar[j]-sum(prices[t,1:]*phi[i,:])
            Y.append(y)
            X.append(ves.marshalliandemands(y,prices[t,1:],alpha[i,:],gamma,phi[i,:]))
            i+=1
            print (t,j)

    X=np.array(X)

    d['y']=Y
    for k in range(K):
        d['x%d' % k]=X[:,k]

    df=pd.DataFrame(d)
    df.set_index(['Year','HH'],inplace=True,drop=False)

    return df
                     
        
if __name__=='__main__':
    df=fake_data()
    #test=pd.DataFrame({'x1':[1,2,3,4],'x2':[2,3,4,6],'hhsize':[1,1,2,2]})
    #Ex=proj(test[['x1','x2']],test[['hhsize']])
