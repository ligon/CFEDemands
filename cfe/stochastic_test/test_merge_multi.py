# [[file:../../Empirics/cfe_estimation.org::test_merge_multi][test_merge_multi]]
# Tangled on Mon Sep 12 15:51:17 2022

import numpy as np
import pandas as pd
from cfe.df_utils import merge_multi
from cfe.dgp import expenditures #, lambdas, characteristics, measurement_error  
from cfe.result import to_dataframe
import pytest

def artificial_data(T=2,N=120,M=1,k=2,n=4,sigma_e=0.001,sigma_phi=0.1):
    x,truth=expenditures(N,T,M,n,k,beta=np.linspace(1,3,n),sigma_phi=sigma_phi,sigma_eps=sigma_e)
    y=np.log(x+0.001)
    return y,truth

N = 5000
tol = 1e-4

a,truth=artificial_data(T=2,N=N,k=2,n=10,sigma_e=1e-10)
a = to_dataframe(a,['j','t','m']).T

# make a dataframe with a subset of the index of the previous one 
b = pd.DataFrame(np.random.normal(size=(N,1)), columns = ['b'])
b.index.names = ['j']

# make one that has some missing info
b_subset = b.iloc[1:500, :].copy()


@pytest.mark.parametrize("b", [
    b,
    b_subset,
])
def test_merge_multi(b):
    mymerge = a.join(b)
    merge_multi_test = merge_multi(a, b, on='j')

    assert np.sum(np.abs(mymerge - merge_multi_test)).sum() <  tol
# test_merge_multi ends here
