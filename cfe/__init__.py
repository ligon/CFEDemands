import demands
from . import estimation
from . import df_utils
from . import dgp
from . import input_files
from .result import Result, from_dataset
#from demands import engel_curves
from .regression import Regression, read_sql, read_pickle

with open('VERSION') as f: __version__ = f.readline()
