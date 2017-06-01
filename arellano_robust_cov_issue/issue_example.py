#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:53:21 2017

@author: meganlang
"""
import tempfile
import numpy as np
import pandas as pd
from numpy.linalg import norm
import sys
import cfe as nd

#Import dataset that's triggering the issue
df=pd.read_pickle('ghana.df')

#Do the usual transformations on it to create y and z datasets that cfe calls
mydf=df.copy()
mydf['HHSIZE']=df[['boys','girls','women','men']].sum(axis=1) #Why are there NAx in log hSize???
mydf['HHSIZE'].replace(to_replace=[0],value=[np.NaN],inplace=True)
mydf = mydf.dropna()
mydf['log HSize']=np.log(mydf['HHSIZE'])
mydf['Boys']=df['boys']
mydf['Girls']=df['girls']
mydf['Women']=df['women']
mydf['Men']=df['men']
mydf['Region']= 1

mydf.reset_index(inplace=True)
mydf.set_index(['HH','Round','Region'],drop=True,inplace=True)

mydf.index.set_names(['j','t','mkt'],inplace=True)

mydf.sortlevel(level=0,inplace=True)

y = mydf.ix[:,0:55].replace(0,np.nan) #Zeros to NaN
y = y.dropna(axis=0,how='all') #Drop all columns comprised entirely of zeros.

z=mydf[['Boys','Girls','Men','Women','log HSize']]

use_goods=y.columns[y.count()>500]
y=y[use_goods]

b,se=nd.estimation.estimate_reduced_form(y,z,return_se=True)

