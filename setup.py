from distutils.core import setup
import sys

if not sys.version_info[0] == 2:
    sys.exit("Sorry, Python 3 is not supported (yet)")

setup(
    name='CFEDemands',
    version='0.1.4dev',
    author='Ethan Ligon',
    author_email='ligon@berkeley.edu',
    packages=['cfe',],
    license='Creative Commons Attribution-Noncommercial-ShareAlike 4.0 International license',
    description='Tools for estimating and computing Constant Frisch Elasticity (CFE) demands.',
    url='https://bitbucket.org/ligonresearch/cfedemands',
    long_description=open('README.txt').read(),
)
