# [[file:../../Empirics/cfe_estimation.org::test_df_norm][test_df_norm]]
# Tangled on Mon Sep 12 15:51:17 2022


import numpy as np
import pandas as pd
from cfe.df_utils import df_norm
from cfe.dgp import expenditures #, lambdas, characteristics, measurement_error  
from cfe.result import to_dataframe
import pytest

def artificial_data(T=2,N=120,M=1,k=2,n=4,sigma_e=0.001,sigma_phi=0.1):
    x,truth=expenditures(N,T,M,n,k,beta=np.linspace(1,3,n),sigma_phi=sigma_phi,sigma_eps=sigma_e)
    y=np.log(x+0.001)
    return y,truth

a,truth=artificial_data(T=2,N=5000,k=2,n=10,sigma_e=1e-10)
a = to_dataframe(a,['j','t','m']).T
# create another random matrix for subtracting 
b = pd.DataFrame(np.random.normal(size = a.shape), index = a.index, columns = a.columns)

# introduce NAs for a test case
a_missing = a.copy()
b_missing = b.copy()
for i in range(5000):
    row_a = np.random.randint(low=0, high = a.shape[0])
    col_a = np.random.randint(low=0, high = a.shape[1])
    a_missing.iloc[row_a, col_a] = np.NaN

    row_b = np.random.randint(low=0, high = b.shape[0])
    col_b = np.random.randint(low=0, high = b.shape[1])
    b_missing.iloc[row_b, col_b] = np.NaN

tol = 1e-4

@pytest.mark.parametrize("m", [
    a,
    a_missing,
])
def test_norm_onemat(m):
    mynorm = (m**2).sum().sum()**.5
    df_norm_norm = df_norm(m)
    assert np.abs(mynorm - df_norm_norm) < tol

@pytest.mark.parametrize("m, n", [
    (a, b),
    (a_missing, b_missing),
    (a, b_missing),
    (a_missing, b)
])
def test_norm_twomat(m, n):
    net = m - n
    mynorm = (net**2).sum().sum()**.5
    df_norm_norm = df_norm(m, n)
    assert np.abs(mynorm - df_norm_norm) < tol
# test_df_norm ends here
