# [[file:../../Empirics/cfe_estimation.org::*Tests][Tests:1]]
# Tangled on Mon Sep 12 15:51:17 2022
import cfe
import numpy as np
import pandas as pd

N = 10000

u = pd.DataFrame({'Constant':1,'u':np.random.normal(size=(N,))})

# Just regress u on constant
b,se,e = cfe.df_utils.ols(u[['Constant']],u[['u']],return_se=True,return_e=True)

np.testing.assert_allclose(0,b,atol=2/np.sqrt(N),err_msg='Point estimate wrong') # Outside two se
np.testing.assert_allclose(1/np.sqrt(N),se,atol=2*np.sqrt(3)/np.sqrt(N),err_msg='SE estimate wrong')
np.testing.assert_allclose(u[['u']] - u['u'].mean(),e,atol=1/np.sqrt(N),err_msg='Residuals wrong')
# Tests:1 ends here

# [[file:../../Empirics/cfe_estimation.org::*Tests][Tests:2]]
# Tangled on Mon Sep 12 15:51:17 2022
import cfe
import pylab as pl
import numpy as np
import pandas as pd

from cfe.dgp import geometric_brownian
from cfe.result import to_dataframe

J=100
n=12
T=4
x,parts = cfe.dgp.expenditures(J,T,1,n,2,np.linspace(0.25,3,n),sigma_phi=0.,sigma_eps=0.01,Fbar=geometric_brownian(1.))
y = to_dataframe(np.log(x.where(x>0,np.nan)),'i')
z = to_dataframe(np.log(parts['characteristics']),'k')

X = z.copy()
X['Constant'] = 1

b,v = cfe.df_utils.ols(X,y,return_v=True,return_se=False)
# Tests:2 ends here
