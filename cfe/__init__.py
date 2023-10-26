import consumerdemands as demands
from . import estimation
from . import df_utils
from . import dgp
from . import input_files
from .result import Result, from_dataset
from consumerdemands import engel_curves
from .regression import Regression, read_sql, read_pickle

from importlib.metadata import version
__version__ = version('CFEDemands')
