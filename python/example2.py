#!/usr/bin/python
import numpy as np
import sympy
from matrix_boyd import matrix_boyd, skew_prod, eval_expansion, poly2cheb

##
# Program Description
##
# This program calculates the recursion coefficients of the sequence of
# orthogonal matrix polynomials for a weight function of the form
#   W(x) = Q(x)/sqrt(1-x^2)
# where here Q(x) is a matrix-valued polynomial which is positive-definite and
# symmetric on the interval (-1,1).  In this example, we will use the specific value of 
#          [  1   a*x ]
#   Q(x) = |          |
#          [ a*x   1  ]
# where here |a| <= 1
##

##
# Specify Q
##
# Define the polynomial Q(x) by specifying the coefficients in the cosine
# expansion
#   Q(cos(t)) = Q0 + Q1*cos(t) + Q2*cos(2*t) + ... + Qd*cos(d*t)
##

nx = 2      # dimensions of Q
qd = 1      # degree of Q
Q = np.zeros([qd+1,nx,nx])

a = 0.9 # some random parameter
Q[0,:,:] = sympy.Matrix([[1,0],[0,1]])
Q[1,:,:] = sympy.Matrix([[0,a],[a,0]])

##
# Get the recursion coefficients
## 
# Use Gram-Schmidt to find the sequence of matrix-valued polynomials
#   P0(x), P1(x), P2(x), ...
# in terms of their cosine expansions
#   Pn(cos(t)) = F0 + F1*cos(t) + F2*cos(2*t) + ... + Fn*cos(n*t)
# where the sequence is uniquely determined by Fn=I.  Use it to calculate sequences of matrices
#   B0, B1, B2, B3, ...
#   C0, C1, C2, C3, ...
# satisfying the property that
#   x*Pn(x) = (I/2)*Pn+1(x) + Bn*Pn(x) + Cn*Pn-1(x)
##
degmax = 10 # maximum number of recurrence coefficients
def eyef(x):
  return np.eye(nx)
root, Beta0, B, C = matrix_boyd(eyef,Q,-1,1,degmax)
C = C*2 # rescaling

##
# Print out results
##
# We print out the results in pretty matrix form.  We can't use rational
# simplification this time, because the results are not rational
##

print("Printing the B sequence:")
for k in range(degmax):
  print("B%i" % k)
  #sympy.pprint(sympy.nsimplify(sympy.Matrix(B[k,:,:]),tolerance=1e-15,rational=True))
  #sympy.pprint(sympy.Matrix(B[k,:,:]))
  print(sympy.Matrix(B[k,:,:]))
  
print("Printing the C sequence:")
for k in range(1,degmax):
  print("C%i" % k)
  #sympy.pprint(sympy.nsimplify(sympy.Matrix(C[k,:,:]),tolerance=1e-15,rational=True))
  #sympy.pprint(sympy.Matrix(C[k,:,:]))
  print(sympy.Matrix(C[k,:,:]))


