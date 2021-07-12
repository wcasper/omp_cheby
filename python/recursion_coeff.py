#!/usr/bin/python
import numpy as np
import sympy
from scipy.fftpack import dct
from scipy import linalg

z = sympy.symbols("z")

def skew_prod(P,Q):
  nx = len(P[0,:,0])
  degP = len(P[:,0,0])-1
  degQ = len(Q[:,0,0])-1
  d = degQ + degP

  tmp = np.zeros([d+1,nx,nx],dtype=int)*z
  for j in range(degQ+1):
    for k in range(degP+1):
      PQT = np.dot(P[k,:,:],np.transpose(Q[j,:,:]))
      if(j-k <0):
        tmp[k+j,:,:] += PQT*sympy.Rational(1,2)
        tmp[k-j,:,:] += PQT*sympy.Rational(1,2)
      else:
        tmp[j+k,:,:] += PQT*sympy.Rational(1,2)
        tmp[j-k,:,:] += PQT*sympy.Rational(1,2)
  return tmp

def inner_prod(P,Q,W0):
  nx = len(W0[0,:,0])
  degP = len(P[:,0,0])-1
  degQ = len(Q[:,0,0])-1
  degW = len(W0[:,0,0])-1
  d = degQ + degW

  tmp = np.zeros([d+1,nx,nx],dtype=int)*z

  # calculate the product W0*Q^T
  for j in range(degQ+1):
    for k in range(degW+1):
      #W0QT = np.dot(W0[k,:,:],np.transpose(Q[j,:,:]))
      W0QT = sympy.Matrix(W0[k,:,:])*(sympy.Matrix(Q[j,:,:]).transpose())
      if(j-k <0):
        tmp[k+j,:,:] += W0QT*sympy.Rational(1,2)
        tmp[k-j,:,:] += W0QT*sympy.Rational(1,2)
      else:
        tmp[j+k,:,:] += W0QT*sympy.Rational(1,2)
        tmp[j-k,:,:] += W0QT*sympy.Rational(1,2)


  # calculate the integral of P times W0*Q^T
  result = np.zeros([nx,nx],dtype=int)*z
  for i in range(min(degP,d)+1):
    #result += np.dot(P[i,:,:],tmp[i,:,:])
    result += sympy.Matrix(P[i,:,:])*sympy.Matrix(tmp[i,:,:])
  #result += np.dot(P[0,:,:],tmp[0,:,:])
  result += sympy.Matrix(P[0,:,:])*sympy.Matrix(tmp[0,:,:])
  result *= sympy.Rational(1,2)
    
  return result

def poly2cheb(W0poly):
  W0 = W0poly*0*z
  nx = len(W0[0,:,0])
  W0[0,:,:] = np.eye(nx,dtype=int)

  # convert each poly expansion from poly basis to cheby  basis
  for j in range(nx):
    for k in range(nx):
      tmp = np.polynomial.chebyshev.poly2cheb(W0poly[:,j,k])
      W0[:len(tmp),j,k] = tmp

  return W0

def get_recursion(W0,x0,x1,degmax):
  nx = len(W0[0,:,0])
  degW = len(W0[:,0,0])-1

  Bs = np.zeros([degmax,nx,nx],dtype=int)*z
  Cs = np.zeros([degmax,nx,nx],dtype=int)*z

  # use Gram-Schmidt to iteratively calculate orthogonal polynomials
  # also calculate the recursion relation coefficients and the
  # expansion coefficients of F(x)
  P0 = np.zeros([2,nx,nx],dtype=int)*z
  P1 = np.zeros([2,nx,nx],dtype=int)*z
  P1[0,:,:] = np.eye(nx,dtype=int)
  for d in range(1,degmax+1):
    P2 = np.zeros([d+1,nx,nx],dtype=int)*z
    for k in range(1,d):
      P2[k+1,:,:] += P1[k,:,:]*sympy.Rational(1,2)
      P2[k-1,:,:] += P1[k,:,:]*sympy.Rational(1,2)
    P2[1,:,:] += P1[0,:,:]

    B = np.dot(inner_prod(P2,P1,W0),sympy.Matrix(inner_prod(P1,P1,W0)).inv())
    if(d>1):
      C = np.dot(inner_prod(P2,P0,W0),sympy.Matrix(inner_prod(P0,P0,W0)).inv())
    else:
      C = np.zeros([nx,nx],dtype=int)*z

    B = sympy.simplify(B)
    C = sympy.simplify(C)

    Bs[d-1,:,:] = B
    Cs[d-1,:,:] = C

    for k in range(d-1):
      P2[k,:,:] -= np.dot(B,P1[k,:,:])
      P2[k,:,:] -= np.dot(C,P0[k,:,:])
    P2[d-1,:,:] -= np.dot(B,P1[d-1,:,:])

    P0 = P1
    P1 = P2

    print("B C for d=",d)
    print(B)
    print(C*2) # rescaled

  return Bs, Cs

def matrix_boyd(F,W0,x0,x1,degmax,coeffs=[]): 
  # get the recursion and expansion coefficients:
  # xP(x,n) = P(x,n+1) + B(n)P(x,n) + C(n)P(x,n-1)
  # F(x) = Beta[0]*P(x,0) + Beta[1]*P(x,1) + ...
  nx = len(W0[0,:,0])
  if(coeffs==[]):
    Beta, B, C = poly_expand(F,W0,x0,x1,degmax)
  else:
    foo, B, C = poly_expand(F,W0,x0,x1,degmax)
    Beta = coeffs

  zeros = []

  return zeros, Beta, B, C

