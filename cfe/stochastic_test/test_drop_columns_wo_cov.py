# [[file:../../Empirics/cfe_estimation.org::test_drop_columns_wo_cov][test_drop_columns_wo_cov]]
# Tangled on Mon Sep 12 15:51:17 2022
from cfe.estimation import drop_columns_wo_covariance
import pandas as pd
import numpy as np
import pytest 

# generate random data
N = 5
M = 3
squaredf = pd.DataFrame(np.random.normal(size=(N, N)))
rectandf = pd.DataFrame(np.random.normal(size=(N, M)))

# min_obs values to test: 1, ..., M plus None
min_obs_list = [i for i in range(M)]
min_obs_list.append(None)

# create test cases by introducing na values
line = rectandf.copy()
line.iloc[1, :] = np.NaN

rand = rectandf.copy()
rand.iloc[1, 1] = np.NaN
rand.iloc[3, 2] = np.NaN

single = rectandf.copy()
single.iloc[2,2] = np.NaN

another = rectandf.copy()
another.iloc[0,1] = np.NaN 
another.iloc[1,1] = np.NaN 
another.iloc[2,1] = np.NaN 

@pytest.mark.parametrize("df", [
    line,
    rand,
    single,
    another
])
def test_equal_to_drop_cols(df):
    results = 0
    for m in min_obs_list:
        foo = drop_columns_wo_covariance(df, min_obs=m)
        baz = df.loc[:,df.count()>=m]

        print(foo)

        # get difference 
        diff = foo - baz
        diff = diff.fillna(0)

        if ~((foo.shape == baz.shape) & ((diff < 1e-5).all(axis=None))):
            results += 1

    assert results == 0

# end rest_drop_columns_wo_cov.py
# test_drop_columns_wo_cov ends here
