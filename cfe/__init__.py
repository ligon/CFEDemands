import consumerdemands as demands
from . import estimation
from . import df_utils
from . import dgp
from . import input_files
#from .result import Result, from_dataset
from consumerdemands import engel_curves
from .regression import Regression, read_pickle #,read_sql

try: 
    from importlib.metadata import version # Set in file VERSION
except:
    from importlib_metadata import version # for python 3.7.9

__version__ = version('CFEDemands')

