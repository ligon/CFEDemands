#!/usr/bin/env python

"""
A collection of functions pertaining to graphing Engel Curves
"""

import variable_elasticity_utility as ves
import pylab as pl

def plot(p,alpha,gamma,phi,labels=[],ybounds=[0,10],fname=None,NegativeDemands=False):
    f=lambda y: ves.marshalliandemands(y,p,alpha,gamma,phi,NegativeDemands=NegativeDemands)

    if NegativeDemands:
        ymin=-sum([p[i]*phi[i] for i in range(len(p))])
    else:
        ymin=0
        
    if ybounds[0]<=ymin: ybounds[0]=ymin+1e-12
        
    Y=pl.linspace(ybounds[0],ybounds[1],100)
    pl.clf()
    p=pl.plot(Y,[f(y) for y in Y])
    if len(labels)>0:
        pl.legend(labels,loc=2)

    pl.xlabel('Total Expenditures')
    pl.ylabel('Particular Expenditures')

    if fname:
        pl.savefig(fname)
    else:
        pl.show()

    return p

def plot_shares(p,alpha,gamma,phi,labels=[],ybounds=[0,10],fname=None,NegativeDemands=False):

    y=linspace(sum(-phi*p)+0.00001,ybounds[1],100)

    pl.figure(1)
    x=array([budgetshares(xbar,p,alpha,gamma,phi,NegativeDemands=NegativeDemands) for xbar in y])

    p=pl.plot(y,x)

    pl.xlabel('Total Expenditures')
    pl.ylabel('Expenditure Share')

    if len(labels)>0:
        pl.legend(labels,loc=2)

    if fname:
        pl.savefig(fname)
    else:
        pl.show()

    return p

    
if __name__=='__main__':
    p=plot([1.,1.,1.],[1.,2.,3.],[1.,1.,1.],[.5,0.,0.])
    
