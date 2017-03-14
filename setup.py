from distutils.core import setup

setup(
    name='CFEDemands',
    version='0.1dev',
    author='Ethan Ligon',
    author_email='ligon@berkeley.edu',
    packages=['cfe',],
    license='Creative Commons Attribution-Noncommercial-ShareAlike 4.0 International license',
    description='Tools for estimating and computing Constant Frisch Elasticity (CFE) demands.',
    long_description=open('README.txt').read(),
)
