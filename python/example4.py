#!/usr/bin/python
import numpy as np
import sympy
from recursion_coeff import get_recursion, skew_prod, poly2cheb

##
# Program Description
##
# This program calculates the recursion coefficients of the sequence of
# orthogonal matrix polynomials for a weight function of the form
#   W(x) = Q(x)/sqrt(1-x^2)
# where here Q(x) is a matrix-valued polynomial which is positive-definite and
# symmetric on the interval (-1,1).  In this example, we will use the specific value of 
#   Q(x) = L(x)*D(x)*L(x)^T
# where here L(x) is a lower triangular matrix whose diagonal entries are all 1.
##

##
# Specify L and D
##
# Define the polynomials L(x) and D(x) by specifying the polynomial coefficients
#   L(x) = L0 + L1*x + L2*x^2 + ... + Ld*x^d
#   D(x) = D0 + D1*x + D2*x^2 + ... + Dm*x^m
##

## CAREFUL: D diagonal and L lower triangular with 1's on diagonal !!!!

nx = 2      # dimensions of L
ld = 1      # degree of L
dd = 1      # degree of D
a,b = sympy.symbols("a b")
L = np.zeros([ld+1,nx,nx],dtype=int)*a
L[0,:,:] = sympy.Matrix([[1,0],[0,1]])
L[1,:,:] = sympy.Matrix([[0,0],[a,0]])

D = np.zeros([dd+1,nx,nx],dtype=int)*a
D[0,:,:] = sympy.Matrix([[1,0],[0,b]])
D[1,:,:] = sympy.Matrix([[0,0],[0,0]])

# convert to cosine series expansion coefficients
Lcheb = poly2cheb(L)
Dcheb = poly2cheb(D)

##
# Calculate Q
##
#   Q(x) = L(x)*D(x)*L(x)^T
##

LD = skew_prod(Lcheb,Dcheb)
Q = skew_prod(LD,Lcheb)

Q = sympy.nsimplify(Q,rational=True)

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
degmax = 20 # maximum number of recurrence coefficients
B, C = get_recursion(Q,-1,1,degmax)
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


