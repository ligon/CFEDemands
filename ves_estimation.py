#!/usr/bin/env python
"""
This is a module with functions which implement various aspects of estimation of VES demand systems.
"""

from warnings import warn
import pandas as pd
from numpy import linalg
from numpy.linalg import norm
import numpy as np
import pylab as pl
import sys
sys.path.append('../Computation')
sys.path.append('../Data/Uganda')
import uganda
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
    b=pd.DataFrame(b.T,index=y.columns,columns=x.columns)

    if returnb:
        return pd.DataFrame(np.dot(vicx[0],b.T),index=vicx[1],columns=vicy[2]),b.T
    else:
        return pd.DataFrame(np.dot(vicx[0],b.T),index=vicx[1],columns=vicy[2])

def ols(x,y):
    b=linalg.lstsq(x,y)[0]

    u=y-np.dot(x,b)
    v=np.cov(u)*np.mat(np.dot(x.T,x)).I

    return b,np.sqrt(np.diag(v))

def group_expenditures(df,groups):
    myX=pd.DataFrame(index=df.index)
    for k,v in groups.iteritems():
        try:
            myX[k]=df[list(v)].sum(axis=1)
        except IndexError:
            myX[k]=df[['$x_{%d}$' % i for i in v]].sum(axis=1)
            
    return myX

def difference_over_time(df):
    mydf=df.copy()
    mydf.sortlevel(inplace=True)
    
    Rounds=df.index.levels[0]
    T=len(Rounds)
    N=len(df.index.levels[1])

    mydf.reset_index(level='Year',inplace=True)

    for t in range(len(Rounds[1:])):
        ddf=mydf[mydf['Year']==Rounds[t]] - mydf[mydf['Year']==Rounds[t-1]]

        ddf['Year']=Rounds[t]

    ddf.reset_index(inplace=True)
    ddf.set_index(['Year','HH'],inplace=True)

    return ddf

def estimate(x,z,p=None,phi=1e-4):
    """
    Given a pd.Dataframe x of expenditure data, and a pd.DataFrame z
    of household characteristics, estimate parameters gamma, alphabar,
    and lambdas.

    Dataframes are indexed by (Year,hh), with columns in x
    corresponding to expenditure types (no other variables should
    appear), and columns in z corresponding to characteristics.
    """

    logx = np.log(x+phi)
    if p:
        logp=np.log(p)

    # Step 1: Demean over households j and periods t.

    barlogx = logx.mean(axis=0)
    lhs1 = logx - barlogx
    barz =  z.mean(axis=0)
    rhsz1 = z - barz
    if p:
        rhsp1 = logp - logp.mean(axis=0) 
        

    # Step 2: assume normalizations:
    #   * p_i0=1 for i=1,...,n;
    #   * mean(log\lambda^j_t)=0
    #   * mean(\epsilon_i)=0 for i=1,...,n.
    #   * mean(1/gamma_i)=1

    def logalphabar_hat(gammas, deltas, barlogx=barlogx, p=p):
        # Zero out elements of delta which aren't finite:
        deltas.where(np.isfinite(deltas),0,inplace=True)
        
        if p:
            candidate=barlogx*gammas - barz.dot(deltas) + (1-gamma)*logp.mean(axis=0)
        else:
            candidate=barlogx*gammas - barz.dot(deltas) + (1-gammas)*logp(gammas,deltas).mean(axis=0)

        return candidate


    # Step 3: Demean over periods t.
    barlogxij = logx.groupby(level='HH').mean()
    barzj =  z.groupby(level='HH').mean()
    rhsz3 = (z.reset_index(level='Year') - barzj).reset_index().set_index(['Year','HH'])
    if p:
        rhsp3 = logp - logp.mean(axis=0)  # Assumes common prices across households

    # Expected average \log\lambda^j
    def barloglambdaj_hat(gammas,deltas,p=p,barlogx=barlogx,barlogxij=barlogxij):
        # Zero out elements of delta which aren't finite:
        deltas.where(np.isfinite(deltas),0,inplace=True)

        gl=(barlogx - barlogxij) - barzj.dot(deltas/gammas)

        if p:
            gl = gl - logp.mean(axis=0).dot(1-1/gammas)

        gl=gl.dropna()
        
        u,s,v=linalg.svd(gl)
        S=np.zeros((u.shape[1],v.shape[0]))

        # Check on decomposition
        if True:
            Sfull=S + 0.
            for i in range(len(s)): Sfull[i,i]=s[i]
            glhat=np.dot(np.dot(u,Sfull),v)
            assert(norm(gl-glhat)<1e-3)
        
        S[0,0]=s[0]
        gwT = np.dot(np.dot(u,S),v).T

        # Temporary normalization (if prices unobserved): first weight equal to 1 (w[0,0]=1).
        barloglambdas=pd.Series(gwT[0,:]/abs(gwT[0,0]),index=gl.index)
        gs=pd.Series(abs(gwT[:,0]),index=gl.columns)

        # Use value of gamma passed, instead of from local svd.
        #barloglambdas=pd.Series(gwT[0,:]*gammas[0],index=gl.index)
        #gs=1./gammas


        if not p: # Free to normalize gammas
            gs.where(np.abs(np.log(gs))<10,np.nan,inplace=True)  # Set extreme outliers to NaN
            gsbar = np.mean(gs)
            gs = gs/gsbar # Normalization
            barloglambdas = barloglambdas*gsbar # Normalization

        try:
            assert(max(abs(gs-1/gammas))<1e-3)
        except AssertionError:
            warn('Two estimates of 1/gamma are not in close agreement.')

        return barloglambdas,gs


    # Step 4:
    barlogxit = logx.groupby(level='Year').mean()
    barzt =  z.groupby(level='Year').mean()
    if not p:
        def logp(gammas,deltas):
            idx=gammas.index
            lnp=(barlogxit.loc[:,idx] - barlogx.loc[idx])*(gammas/(gammas-1.)) - barzt.dot(deltas/(gammas-1.))
            # Deal with case where gamma=1
            lnp.where(np.isfinite(lnp),0,inplace=True)

            return lnp
    

    # Step 5: Difference out fixed stuff and effect of changes in z
    
    logx_deviations=(logx.reset_index(level='HH').loc[:,barlogxit.columns] - barlogxit).reset_index()
    logx_deviations['HH']=np.tile(logx.index.levels[1].values,(2,))
    logx_deviations.set_index(['Year','HH'],inplace=True)
    Y = difference_over_time(logx_deviations)

    Yhat,d = proj(Y,difference_over_time(z),returnb=True)
    Y = (Y - Yhat).dropna()

    # Step 6: Use svd to obtain gammas, dloglambdas
    u,s,v=linalg.svd(Y)
    print "Singular values of gwT"
    print s

    S=np.zeros((u.shape[1],v.shape[0]))
    S[0,0]=s[0]
    gwT = np.dot(np.dot(u,S),v).T

    G=pd.Series(abs(gwT[:,0]),index=Y.columns)

    # Eliminate goods if they deliver non-positive gammas
    
    G.where(np.abs(np.log(G))<10,np.nan,inplace=True)  # Set extreme outliers to NaN

    dloglambdas=pd.Series((gwT[0,:])/abs(gwT[0,0]),index=Y.index).unstack(level='Year') # Temporary normalization

    if not p:
        gbar=np.mean(G)
        G=G/gbar  # Normalization of gamma (mean(1/gamma)=1)
        dloglambdas = dloglambdas*gbar

    gammas=pd.Series(1/G,index=Y.columns)
    deltas = d/G

    try:
        assert((abs(deltas.as_matrix())<np.inf).all(axis=1))
    except AssertionError:
        warn('delta has some non-finite elements.')

    #dloglambdas=pd.Series((gwT[0,:]-np.mean(gwT[0,:]))*np.mean(G),index=Y.index).unstack(level='Year') #Normalization
             
    #assert(abs(np.mean(barloglambdaj_hat(gammas,deltas))) < 1e-10) # Check normalization of lambdas

    barloglambdaj,gs=barloglambdaj_hat(gammas,deltas)


    householdd={2005:(barloglambdaj - dloglambdas.T/2).T.dropna()}
    householdd[2010]=(barloglambdaj + dloglambdas.rename(columns={2005:2010}).T/2).T.dropna()

    householddf=pd.DataFrame(householdd[2005])
    householddf[2010]=householdd[2010]
    
    goodsdf=deltas.T.to_dict()
    logalphabars=logalphabar_hat(gammas,deltas)

    logalphabars.where(np.abs(logalphabars)<3*len(logalphabars),np.nan,inplace=True)
    logalphabars=logalphabars-logalphabars.mean()  # Normalization
    
    goodsdf.update({'1/gamma':gs,'gamma':gammas, 'alphabar':np.exp(logalphabars),'phi':phi})

    prices=np.exp(logp(gammas,deltas))

    return householddf,pd.DataFrame(goodsdf,index=deltas.columns), prices

def predicted_expenditures(goodsdf,hhdf,prices):
    """Yields predicted expenditures.  The pd.DataFrame goodsdf should
    have columns 'gamma', 'delta', and 'alphabar'.  The hhdf should
    have a column 'lambdas', and other columns corresponding to
    columns in goodsdf['deltas'].  The priceseries should have
    prices by good, period.
    """
    N,T=(len(hhdf),len(hhdf.columns))
    alpha=goodsdf['alphabar']
    
    for v in set(goodsdf.columns).intersection(set(hhdf.columns)): # Names of household characteristics in z
        alpha += hhdf[v].dot(goodsdf[v])

    X=[]
    idx=[]
    for t in range(T):
        for j in range(N):
            idx+=[(hhdf.columns[t],hhdf.index[j])]
            hhlambda=np.exp(hhdf.iloc[j,t])
            hhalpha = goodsdf['alphabar'] # plus zdelta!
            p=prices.iloc[t,:]
            c=np.array(ves.frischdemands(hhlambda,p,hhalpha,goodsdf['gamma'].as_matrix(),goodsdf['phi'].as_matrix()))
            X.append(c*p)

    return pd.DataFrame(X,index=pd.MultiIndex.from_tuples(idx),columns=goodsdf.index).sort()

    
def estimate_gamma_alpha(expdf,rhsdf,phi=1e-4):
    """
    Given a pd.Dataframe df of expenditure data, estimate parameters gamma and alpha.

    Dataframe is indexed by (Year,hh), with columns corresponding to
    expenditure types (no other variables should appear).
    """
    expdf.sortlevel(inplace=True)
    Rounds=expdf.index.levels[0]
    T=len(Rounds)
    N=len(expdf.index.levels[1])
    n=len(expdf.columns)
    ANOVA={}
    ANOVA['Total_var*2']=2*np.log(expdf+phi).var()
    # First difference across years
    try:
        expdf=expdf.reset_index(level=0)
    except ValueError:
        expdf=expdf.reset_index(level=0,drop=True)  # Avoid collision

    rhsdf['Constant']=1

    try:
        z=rhsdf.copy()
        rhsdf=rhsdf.reset_index(level=0)
    except ValueError:
        rhsdf=rhsdf.reset_index(level=0,drop=True)
    
    for t in range(len(Rounds[1:])):
        dz=rhsdf[rhsdf['Year']==Rounds[t]] - rhsdf[rhsdf['Year']==Rounds[t-1]]

        # Here we add candidate phi before taking logs
        dy=np.log(expdf[expdf['Year']==Rounds[t]]+phi)-np.log(expdf[expdf['Year']==Rounds[t-1]]+phi)

        dy['Year']=Rounds[t]
        dz['Year']=Rounds[t]

    dy.reset_index(inplace=True)
    dz.reset_index(inplace=True)
    dy.set_index(['Year','HH'],inplace=True)
    dz.set_index(['Year','HH'],inplace=True)

    ANOVA['dy_var']=dy.var(numeric_only=True)
    ANOVA['alpha_var']=ANOVA['Total_var*2']-ANOVA['dy_var']

    # Reduced form is y=a_it + d_i*z_j; first, difference out the a_i=ybar-d_i*zbar
    ybar=dy.mean(axis=0,numeric_only=True)
    diffs=dy-ybar
    dz['Constant']=1
    dzbar=dz.mean(axis=0,numeric_only=True)
    dz=dz-dzbar
    dz['Constant']=1 # Put back differenced-out constant term
    dzbar['Constant']=1

    ANOVA['Price_var']=ybar**2
    ANOVA['ddy_var']=diffs.var(numeric_only=True)

    diffshat,d=proj(diffs,dz,returnb=True)

    ANOVA['dddy_var']=diffs.var()-diffshat.var()
    ANOVA['lambda_var']=diffshat.var()
    ANOVA['error_var']=(diffs-diffshat).var()

    # Ratios of gammas can be obtained from ratios of dz's, but to
    # extract individual gammas we need some normalization.

    diffshatbar=np.abs(np.dot(dzbar.values,d))
    myG=np.mean(np.outer(diffshatbar,1/diffshatbar),axis=0) 

    Y=(diffs-diffshat).dropna()
    
    u,s,v=linalg.svd(Y)
    print s
        
    S=np.zeros((u.shape[1],v.shape[0]))
    S[0,0]=s[0]
    gwT = np.dot(np.dot(u,S),v.T).T

    gwT=gwT/abs(gwT[0,0])

    G=abs(gwT[:,0])
    gammas=pd.Series(1/G,index=Y.columns)

    # Set crazy values to NaN
    gammas.where(gammas<100,inplace=True)
    G=1/gammas
    
    # Construct the alphabar
    alphabar=np.log(expdf[diffs.columns]+phi).mean(axis=0)

    # Get rid of 1/gamma_i and take antilog to get the alphabars we want
    alphabar=(alphabar/G)

    # Now back out price change; normalize period 1 prices to 1.
    ait=expdf.groupby(level=0).mean()-np.tile(expdf.mean(axis=0),(N,1)) - z.groupby(level=0).mean().dot(d)
    dlogpt=ait/(1-G)

    n,alphabar,gamma,phi = ves.check_args(np.exp(dlogpt),alphabar,1/G,phi)
    alphabar=pd.Series(np.exp(alphabar),index=Y.columns)

    ANOVA.update({'Price':np.exp(dlogpt),'alpha':alphabar,'gamma':gammas,'phi':phi})

    dloglambdas=pd.Series(gwT[0,:]-np.mean(gwT[0,:]),index=Y.index)

    deltas=pd.DataFrame({z.columns[i]:d.ix[i]/G for i in range(len(z.columns))})
    
    return pd.DataFrame(ANOVA,index=diffs.columns),dloglambdas,deltas

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

def balance_panel(df):
    """Drop households that aren't observed in all rounds."""
    pnl=df.to_panel()
    keep=pnl.loc[list(pnl.items)[0],:,:].dropna(how='any',axis=1).iloc[0,:]
    df=pnl.loc[:,:,keep.index].to_frame(filter_observations=False)
    df.index.names=pd.core.base.FrozenList(['Year','HH'])
    
    return df

def engel_curves(rslt,ybounds=[0,10],fname=None):
    
    return engel.plot(rslt['Price'],rslt['alpha'],rslt['gamma'],rslt['phi'],
                                 labels=rslt.index,ybounds=ybounds,fname=fname)

def CRRA_adjustment(X,g):
    """
    Given an NTxn dataframe of expenditures X on n goods and a vector g=[1/gamma_i]_i,
    calculate the factor by which relative risk aversion differs from
    what it would if gamma_i=gamma for each household-period.
    """
    RRA=(X*g).dropna(axis=1,how='all').dropna(axis=0,how='any').sum(axis=1)
    ell=g/g
    RRA=g.mean()*(X*ell).dropna(axis=1,how='all').dropna(axis=0,how='any').sum(axis=1)/RRA

    return RRA

def bootstrap(df,lhsvar,rhsvar,reps=100):
    """
    Bootstrap reps draws of estimate using df.
    """
    Gammas=[]
    Rounds=list(set(df.index.levels[0]))
    T=len(Rounds)
    hhs=list(set(df.index.levels[1]))
    N=len(hhs) # Number of households
    for i in range(reps):
        print "Draw %d" % i
        use=[]
        bootdf=pd.DataFrame(columns=df.columns,dtype='float64')
        k=0
        while k<N:
            hh=hhs[np.random.random_integers(0,N-1,1)] # Draw single household
            try:  
                obs=pd.DataFrame([df.loc[(t,hh)] for t in Rounds],dtype='float64')
                obs.rename(index={(t,hh):(t,k) for t in Rounds},inplace=True)
                k+=1
            except KeyError: # Not a full set of rounds for household?
                pass
            use.append(obs)
        bootdf=bootdf.append(use)
        bootdf.index = pd.MultiIndex.from_tuples(bootdf.index,names=['Year','HH'])
        g=estimate(bootdf.loc[:,lhsvar],bootdf.loc[:,rhsvar],phi=1e-14)[1]['gammas']
        g=g/np.mean(g)
        Gammas.append(g)

    return pd.DataFrame(Gammas,index=range(reps))

        

def fake_hhsize(N,T,p0=1./3,p1=.9):
    """
    Generate time-varying household sizes for N households over T periods.

    The parameter p0 governs the initial (geometric) size distribution (smaller p0 means bigger sizes),
    while the parameter p1 governs the rate at which these household sizes evolve over time.
    """
    hhsize = np.random.geometric(p0,size=N).reshape((N,1)) # Initial household size
    X=np.c_[np.zeros((N,1)),np.arange(N).reshape((-1,1)),hhsize]
    for t in range(1,T):
        xt=hhsize + np.random.geometric(p1,size=(N,1))-np.random.geometric(p1,size=(N,1))
        hhsize=xt # xt
        xt=np.choose(xt>0,[1,xt]) # Minimum hhsize should be 1
        xt=np.c_[t*np.ones((N,1)),np.arange(N).reshape((-1,1)),xt]
        X=np.r_[X,xt]

    return X.astype(np.int32)

def fake_prices(K,T,sigma=1./4):
    """
    Generate time-varying prices for K goods over T periods.
    Martingale process with multiplicative log-normal innovations.

    First column is an indicator of the round.
    """
    p=np.ones(K).reshape((1,-1)) # Initial prices; normalize to 1
    X=np.c_[np.zeros((1,1)),p]
    for t in range(1,T):
        pt=p*np.exp(np.random.normal(-(sigma**2/2),sigma,size=(1,K)))
        p=pt
        pt=np.c_[t*np.ones((1,1)),pt]
        X=np.r_[X,pt]

    return X

def fake_data(size=(2,100,4),delta=1.,alphasigma=0.1):
    """
    Generate a fake dataset for for K goods for N households over T periods.
    """
    T,N,n=size

    d={}
    x = fake_hhsize(N,T)
    d['HHSize']=x[:,-1]
    d['Year']=x[:,0]
    d['HH']=x[:,1]

    # Let alphas be a function of hhsize
    alphabar=np.random.random(size=n)
    alphabar=np.log(alphabar) - np.mean(np.log(alphabar)) # Normalization
    alphabar=alphabar.reshape((1,n)) 
    alpha=alphabar + delta*x[:,-1].reshape((-1,1)) + np.random.normal(-(alphasigma**2)/2.,alphasigma,size=(alphabar.shape))
    alpha=np.exp(alpha)
    
    # Generate phis from a double geometric; make proportional to household size
    #phi=0.25*(np.random.geometric(2./3,size=(1,K))-np.random.geometric(2./K,size=(1,K)))*x[:,-1].reshape((-1,1))

    # Eliminate dependence of phi on household size.
    phi=1e-14*np.ones((n,))

    gammainv=1./np.arange(1.,n+1)/n*3
    gammainv=gammainv/np.mean(gammainv)  # Normalization
    gamma=1/gammainv
    

    prices=fake_prices(n,T,sigma=1e-15)

    #ystar=np.exp(np.random.normal(10,3,size=(N,T)))
    ystar=np.exp(np.random.normal(size=(N,T)))
    #ystar=1+np.arange(N*T).reshape(N,T) # Use this to eliminate randomness in total expenditures
    X=[]
    Y=[]
    L=[]
    for t in range(T):
        for j in range(N):
            y=ystar[j,t]-sum(prices[t,1:]*phi)
            Y.append(y)
            L.append(ves.lambdavalue(y,prices[t,1:],alpha[j,:],gamma,phi))
        print "Period %d" % t

    
    L=np.array(L).reshape((N,T),order='F')
    L=np.exp(np.log(L) - np.log(L).mean(axis=0))
    
    for t in range(T):
        for j in range(N):
            X.append(ves.frischdemands(L[j,t],prices[t,1:],alpha[j,:],gamma,phi))


    X=np.array(X)

    d['y']=Y
    for k in range(n):
        d['x%d' % k]=X[:,k]

    df=pd.DataFrame(d)
    df.set_index(['Year','HH'],inplace=True,drop=True)
    
    truegoodsdf=pd.DataFrame({'gamma':gamma,'phi':phi,'alphabar':np.exp(alphabar[0])},index=['x%d' %k for k in range(n)])
    hhdf={'lambda':np.log(L).reshape(-1,order='F')}
    hhdf.update({'alpha_%d' % i:alpha[:,i] for i in range(n)})
    truehhdf=pd.DataFrame(hhdf,index=df.index).unstack('Year')
    prices=pd.DataFrame(prices[:,1:],columns=truegoodsdf.index,index=range(T))

    return df,{'goods':truegoodsdf,'lambda':truehhdf,'prices':prices}

def test0(n=2,N=4,T=2):
    """
    (Almost) deterministic test.  No household characteristics to affect alpha.
    """
    
    df,truevalues=fake_data(size=(T,N,n),delta=0.,alphasigma=1e-16)

    # Note zeroing out of household characteristics:
    hhdf,goodsdf,prices=estimate(df.loc[:,["x%d" % i for i in range(n)]],0*df.loc[:,['HHSize']],phi=1e-14)

    exphat=predicted_expenditures(goodsdf,hhdf,prices)
    mse=((df-exphat)**2).mean().dropna()
    try:
        print goodsdf['alphabar']
        assert(mse.max()<1e-4) # bound on mse
    except AssertionError:
        print "true expenditures",
        print df
        print "MSE"
        print mse

        dg=norm((1./truevalues['goods']['gamma'].as_matrix())-goodsdf['1/gamma'].as_matrix(),np.inf)
        print "Difference in 1/gammas: %g" % dg
        if dg>1e-2:
            print "1/gamma (true, estimated)"
            print 1./truevalues['goods']['gamma'], goodsdf['1/gamma']

        print "Difference in lambdas: %g" % norm(truevalues['lambda']['lambda'].as_matrix()-hhdf.as_matrix(),np.inf)
        print "Difference in alphabars: %g" % norm(truevalues['goods']['alphabar'].as_matrix()-goodsdf['alphabar'].as_matrix(),np.inf)
        raise AssertionError
            

def test1(n=2,N=4,T=2):
    """
    Small, (almost) deterministic test.  Add household characteristics
    """
    df,truevalues=fake_data(size=(T,N,n),delta=1.,alphasigma=1e-16)
    #test=pd.DataFrame({'x1':[1,2,3,4],'x2':[2,3,4,6],'hhsize':[1,1,2,2]})
    #Ex=proj(test[['x1','x2']],test[['hhsize']])
    hhdf,goodsdf,prices=estimate(df.loc[:,["x%d" % i for i in range(n)]],df.loc[:,['HHSize']],phi=1e-14)
    #Gammas=bootstrap(df,["x%d" % i for i in range(12)],['HHSize'],reps=3)
    exphat=predicted_expenditures(goodsdf,hhdf,prices)
    mse=((df-exphat)**2).mean().dropna()
    try:
        assert(mse.max()<1e-4) # bound on mse
    except AssertionError:
        print "true expenditures",
        print df
        print "MSE"
        print mse
        print "Difference in lambdas: %g" % norm(truevalues['lambda']['lambda'].as_matrix()-hhdf.as_matrix(),np.inf)
        print "Difference in alphabars: %g" % norm(truevalues['goods']['alphabar'].as_matrix()-goodsdf['alphabar'].as_matrix(),np.inf)
        raise AssertionError

        
if __name__=='__main__':
    #test0()
    #test1() # Fails (test of adding hh characteristics)
    #test0(N=1500) # Passes (reasonable # of households)
    #test0(N=100,n=25) # Passes
    #test0(N=1800,n=25)
    uganda.main(datadir='../Data/Uganda/')
