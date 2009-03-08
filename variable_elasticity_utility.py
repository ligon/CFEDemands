#!/usr/bin/env python

from scipy import optimize 
from numpy import array, ones, zeros, sum

def check_args(p,alpha,gamma,phi):
    """
    Perform sanity check on inputs.  Supply default values if these are missing.
    """

    # Make sure all args are of type array:
    p=array(p)

    try: 
        len(alpha) # If len() not defined, then must be a singleton
        alpha=array(alpha)
    except TypeError: alpha=array([alpha])

    try:
        len(gamma) # If len() not defined, then must be a singleton
        gamma=array(gamma)
    except TypeError: gamma=array([gamma])

    try:
        len(phi) # If len() not defined, then must be a singleton
        phi=array(phi)
    except TypeError: phi=array([phi])

    n=len(p)

    if len(alpha)==1<n:
        alpha=ones(n)*alpha
    else:
        if not alpha.all():
            raise ValueError

    if len(gamma)==1<n:
        gamma=ones(n)*gamma
    else:
        if not gamma.all():
            raise ValueError
    
    if len(phi)==1<n:
        phi=ones(n)*phi
    else:
        if not phi.all():
            phi=zeros(n)

    return (n,alpha,gamma,phi)


def derivative(f):
    """
    Computes the numerical derivative of a function with a single scalar argument.

    BUGS: Would be better to actually take a limit, instead of assuming that h 
    is infinitesimal.  
    """
    def df(x, h=2e-5):
        return ( f(x+h/2) - f(x-h/2) )/h
    return df

def frischdemands(lbda,p,alpha,gamma,phi):
    """
    Given marginal utility of income lbda and prices, 
    returns a list of $n$ quantities demanded, conditional on 
    preference parameters (alpha,gamma,phi).
    """
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    x=[((alpha[i]/(p[i]*lbda))**(1/gamma[i]) - phi[i]) for i in range(n)]

    return x

def frischV(lbda,p,alpha,gamma,phi):
    """
    Returns value of Frisch Indirect Utility function
    evaluated at (lbda,p) given preference parameters (alpha,gamma,phi).
    """
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    x=frischdemands(lbda,p,alpha,gamma,phi)

    U=0
    for i in range(n):
        if gamma[i]==1:
            U += alpha[i]*log(x[i]+phi[i])
        else:
            U += alpha[i]*((x[i]+phi[i])**(1-gamma[i])-1)/(1-gamma[i])

    return U

 
def excess_expenditures(y,p,alpha,gamma,phi):
    """
    Return a function which will tell excess expenditures associated with a lambda.
    """
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    n = len(p)

    def f(lbda):

        lbda=abs(lbda)
        d=0.0
        for i in range(n):
            d += ((alpha[i]/(p[i]*lbda))**(1/gamma[i]) - phi[i])*p[i]

        return d - y

    return f

def excess_utility(U,p,alpha,gamma,phi):
    """
    Return a function which will tell excess utility associated with a lambda.
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    n = len(p)
    def f(lbda):

        return U - frischV(abs(lbda),p,alpha,gamma,phi)

    return f

def lambdavalue(y,p,alpha,gamma,phi,ub=10):
    """
    Given income y, prices p and preference parameters
    (alpha,gamma,phi), find the marginal utility of income lbda.
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    f = excess_expenditures(y,p,alpha,gamma,phi)

    try:
        return optimize.bisect(f,1e-20,ub)
    except ValueError:
#        print "Doubling upper bound of %f" % ub
        return lambdavalue(y,p,alpha,gamma,phi,ub*2.0)

def marshalliandemands(y,p,alpha,gamma,phi):

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    lbda=lambdavalue(y,p,alpha,gamma,phi)

    return frischdemands(lbda,p,alpha,gamma,phi)


def indirectutility(y,p,alpha,gamma,phi):
    """
    Returns utils associated with income y and prices p.
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    lbda=lambdavalue(y,p,alpha,gamma,phi)

    return frischV(lbda,p,alpha,gamma,phi)

def lambdaforU(U,p,alpha,gamma,phi,ub=10):
    """
    Given level of utility U, prices p, and preference parameters
    (alpha,gamma,phi), find the marginal utility of income lbda.
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    f = excess_utility(U,p,alpha,gamma,phi)

    # Our root-finder looks within an interval [1e-20,ub].  If root
    # isn't in this interval, optimize.bisect will raise a ValueError;
    # in this case, try again, but with a larger upper bound.
    try:
        return optimize.bisect(f,1e-20,ub)
    except ValueError:
        return lambdaforU(U,p,alpha,gamma,phi,ub*2.0)

def expenditurefunction(U,p,alpha,gamma,phi):

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    x=hicksiandemands(U,p,alpha,gamma,phi)

    return sum(array([p[i]*x[i] for i in range(n)]))


def hicksiandemands(U,p,alpha,gamma,phi):

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    lbda=lambdaforU(U,p,alpha,gamma,phi)

    return frischdemands(lbda,p,alpha,gamma,phi)

def expenditures(y,p,alpha,gamma,phi):

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    
    x=marshalliandemands(y,p,alpha,gamma,phi)

    return array([p[i]*x[i] for i in range(n)])

def budgetshares(y,p,alpha,gamma,phi):
    
    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)
    
    x=expenditures(y,p,alpha,gamma,phi)

    return array([x[i]/y for i in range(n)])

def share_income_elasticity(y,p,alpha,gamma,phi):
    """
    Expenditure-share elasticity with respect to total expenditures.
    """

    n,alpha,gamma,phi = check_args(p,alpha,gamma,phi)

    def w(xbar):
        return budgetshares(xbar,p,alpha,gamma,phi)

    dw=derivative(w)

    return [dw(y)[i]*(y/w(y)[i]) for i in range(n)]

def income_elasticity(y,p,alpha,gamma,phi):

    return array(share_income_elasticity(y,p,alpha,gamma,phi))+1.0
    

def main():

    y=6
    p=array([10.0,15.0])
    alpha=array([0.25,0.75])
    gamma=array([2.0,0.5])
    phi=array([1.0,0.0])

    print indirectutility(y,p,alpha,gamma,phi)
    print budgetshares(y,p,alpha,gamma,phi)
    print share_income_elasticity(y,p,alpha,gamma,phi)
    
    # Here's a test of the connections between different demand
    # representations:
    assert abs(y-expenditurefunction(indirectutility(y,p,alpha,gamma,phi),p,alpha,gamma,phi))<1e-6

    def V(xbar):
        return indirectutility(xbar,p,alpha,gamma,phi)

    dV=derivative(V)

    try:
        lbda=lambdavalue(y,p,alpha,gamma,phi)
        assert abs(dV(y)-lbda)<1e-6
    except AssertionError:
        print "dV=%f; lambda=%f" % (dV(y),lbda)

if __name__=="__main__":
    main()
