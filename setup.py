from distutils.core import setup
import sys

setup(
    name='CFEDemands',
    author='Ethan Ligon',
    author_email='ligon@berkeley.edu',
    packages=['cfe',],
    license='Creative Commons Attribution-Noncommercial-ShareAlike 4.0 International license',
    description='Tools for estimating and computing Constant Frisch Elasticity (CFE) demands.',
    url='https://bitbucket.org/ligonresearch/cfedemands',
    long_description=open('README.txt').read(),
    setup_requires = ['pytest_runner'],
    tests_require = ['pytest']
)

