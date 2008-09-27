#!/usr/bin/env python

from scipy import optimize 
from numpy import *

#    assert n==len(alpha)==len(gamma)==len(phi)

def excess_expenditures(y,p,alpha,gamma,phi):
    """
    Return a function which will tell excess expenditures associated with a lambda.
    """
    n = len(p)
    def f(lbda):

        lbda=abs(lbda)
        d=0.0
        for i in range(n):
            d += ((alpha[i]/(p[i]*lbda))**(1/gamma[i]) - phi[i])*p[i] - y

        return d

    return f

def lambdavalue(y,p,alpha,gamma,phi):

    f = excess_expenditures(y,p,alpha,gamma,phi)

    return optimize.bisect(f,1e-20,2)

def main():

    y=5
    p=array([10.0,15.0])
    alpha=array([0.5,0.5])
    gamma=array([2.0,0.5])
    phi=array([0.0,0.0])

    print lambdavalue(y,p,alpha,gamma,phi)

if __name__=="__main__":
    main()
