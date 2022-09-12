# [[file:../../Empirics/result.org::test_drop_useless_expenditures][test_drop_useless_expenditures]]
# Tangled on Mon Sep 12 15:51:09 2022
from scipy.stats.distributions import chi2
import cfe
import numpy as np

# Tangling may not include :vars from header
try: 
    N
except NameError: # :var inputs not set?
    N=5000
    T=1
    n=6

x,parts = cfe.dgp.expenditures(N,T,1,n,2,np.array([0.5,1.,1.5,2.,2.5,3.]),sigma_phi=0.0,sigma_eps=0.01)
x = x.where(x>0,np.nan)  # Zeros to missing

x = x.where(np.random.rand(*x.shape)>0.9,np.nan) # drop most observations


z = parts['characteristics']

R = cfe.Result(y=np.log(x),z=np.log(z),min_xproducts=50)

assert len(R.coords['i']<n), "Failed to drop missing items?"
# test_drop_useless_expenditures ends here
