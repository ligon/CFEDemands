import consumerdemands as demands
from . import estimation
from . import df_utils
from . import dgp
from . import input_files
#from .result import Result, from_dataset
from consumerdemands import engel_curves
from .regression import Regression, read_pickle #,read_sql

from importlib.metadata import version # Set in file VERSION
__version__ = version('CFEDemands')
