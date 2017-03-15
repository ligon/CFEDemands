
# [[file:~/Dropbox/0Fall2016/cfedemands/Empirics/cfe_estimation.org::*Artificial%20data][artificial_data]]

# Tangled on Wed Mar 15 11:31:50 2017
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

# artificial_data ends here
