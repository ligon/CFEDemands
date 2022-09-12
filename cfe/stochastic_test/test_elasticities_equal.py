# [[file:../../Empirics/cfe_estimation.org::test_elasticities_equal][test_elasticities_equal]]
# Tangled on Mon Sep 12 15:51:17 2022
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats.distributions import f as F

def elasticities_equal(b1,b2,v1,v2,N,N2,pvalue=False,criterion=False):

    assert N2<N, "N2 should be size of sub-sample of pooled sample."
    b1 = b1.reshape((-1,1))
    b2 = b2.reshape((-1,1))

    n=len(b1)

    assert n==len(b2), "Length of vectors must be equal"

    def Fcriterion(psi):
        try:
            psi=psi[0,0]
        except (TypeError, IndexError):
            pass

        d = psi*b1 - b2
        if d.shape[0]<d.shape[1]: d = d.T

        W = np.linalg.inv((psi**2)*v1 + v2) # Independent case

        F = N2*(N-n-1)/((N-1)*(n-1)) * d.T@W@d

        if ~np.isscalar(F):
            F=F[0,0]

        return F

    #result = minimize_scalar(Fcriterion,method='bounded',bounds=[0,10])
    Fcriterion(1.)
    result = minimize_scalar(Fcriterion)
    psi=np.abs(result['x'])
    Fstat=result['fun']

    assert result['success'], "Minimization failed?"

    outputs = [psi,Fstat]

    if pvalue:
        p = 1 - F.cdf(Fstat,n-1,N-n-1)
        outputs.append(p)

    if criterion:
        outputs.append(Fcriterion)
    
    return tuple(outputs)

N = 10000
N2 = 5000
b0=np.array([1,2,3])
v0=np.array([[1,0.5,0.25],[0.5,1,.5],[.25,.5,1]])
B=np.random.multivariate_normal(b0,v0,size=N)

b1=np.mean(B,axis=0)
v1=np.cov(B,rowvar=False)

b2=2*np.mean(B[:N2,:],axis=0) # So true value of psi=2
v2=4*np.cov(B[:N2,:],rowvar=False)

def covb1b2(psi=1.,tol=1e-2):
    last=1
    next=0
    b1bar=0
    b2bar=0
    i=0
    while np.linalg.norm(next-last)>tol:
        i+=1
        last=next
        B1=B[np.random.randint(N,size=N),:]
        newb1=psi*np.mean(B1,axis=0)
        newb2=2*np.mean(B1[np.random.randint(N,size=N2),:],axis=0)
        next = next*(1-1./i) + np.outer(newb1,newb2)/i
        b1bar = b1bar*(1-1./i) + newb1/i
        b2bar = b2bar*(1-1./i) + newb2/i
        if i>100: continue

    C = next - np.outer(b1bar,b2bar)
    return (C + C.T)/2.

def Vmom(psi=1.,tol=1e-2):
    last=1
    next=0
    dbar=0
    i=0
    while np.linalg.norm(next-last)>tol:
        i+=1
        last=next
        newb1=psi*np.mean(B[np.random.randint(N,size=N),:],axis=0)
        newb2=2*np.mean(B[np.random.randint(N,size=N2),:],axis=0)
        d = newb1 - newb2
        next = next*(1-1./i) + np.outer(d,d)/i
        dbar = dbar*(1-1./i) + d/i
        if i>100: continue

    return next - np.outer(dbar,dbar)

psi,F,p,crit = elasticities_equal(b1,b2,v1,v2,N,N2,pvalue=True,criterion=True)
#C=covb1b2()

assert np.abs(psi-2)<0.05, "Value of psi should be about 2"
assert p>0.01, "Should seldrom reject equality of elasticities."
# test_elasticities_equal ends here
