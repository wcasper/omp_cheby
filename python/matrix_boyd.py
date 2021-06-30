#!/usr/bin/python
import numpy as np
import sympy
from scipy.fftpack import dct
from scipy import linalg

def skew_prod(P,Q):
  nx = len(P[0,:,0])
  degP = len(P[:,0,0])-1
  degQ = len(Q[:,0,0])-1
  d = degQ + degP

  tmp = np.zeros([d+1,nx,nx])
  for j in range(degQ+1):
    for k in range(degP+1):
      PQT = np.dot(P[k,:,:],np.transpose(Q[j,:,:]))
      if(j-k <0):
        tmp[k+j,:,:] += PQT/2.0
        tmp[k-j,:,:] += PQT/2.0
      else:
        tmp[j+k,:,:] += PQT/2.0
        tmp[j-k,:,:] += PQT/2.0
  return tmp

def eval_expansion(x, Beta, B, C):
  nx = len(Beta[0,:,0])
  deg = len(C[:,0,0])+1
  P0 = np.eye(nx)*0
  P1 = np.eye(nx)

  F = np.dot(Beta[0,:,:],P1)
  for n in range(1,deg):
    P2 = P1*x - np.dot(B[n-1,:,:],P1) - np.dot(C[n-1,:,:],P0)
    F += np.dot(Beta[n,:,:],P2)
    P0 = P1
    P1 = P2
  return F

def inner_prod(P,Q,W0):
  nx = len(W0[0,:,0])
  degP = len(P[:,0,0])-1
  degQ = len(Q[:,0,0])-1
  degW = len(W0[:,0,0])-1
  d = degQ + degW

  tmp = np.zeros([d+1,nx,nx])

  # calculate the product W0*Q^T
  for j in range(degQ+1):
    for k in range(degW+1):
      W0QT = np.dot(W0[k,:,:],np.transpose(Q[j,:,:]))
      if(j-k <0):
        tmp[k+j,:,:] += W0QT/2.0
        tmp[k-j,:,:] += W0QT/2.0
      else:
        tmp[j+k,:,:] += W0QT/2.0
        tmp[j-k,:,:] += W0QT/2.0


  # calculate the integral of P times W0*Q^T
  result = np.zeros([nx,nx])
  for i in range(min(degP,d)+1):
    result += np.dot(P[i,:,:],tmp[i,:,:])
  result += np.dot(P[0,:,:],tmp[0,:,:])
  result /= 2.0
    
  return result

def poly2cheb(W0poly):
  W0 = W0poly*0
  nx = len(W0[0,:,0])
  W0[0,:,:] = np.eye(nx)

  # convert each poly expansion from poly basis to cheby  basis
  for j in range(nx):
    for k in range(nx):
      tmp = np.polynomial.chebyshev.poly2cheb(W0poly[:,j,k])
      W0[:len(tmp),j,k] = tmp

  return W0

def poly_expand(F,W0,x0,x1,degmax):
  nx = len(W0[0,:,0])
  degW = len(W0[:,0,0])-1

  Bs = np.zeros([degmax,nx,nx])
  Cs = np.zeros([degmax,nx,nx])

  # get the cosine expansion of F to high enough degree
  n = (degmax+degW)*2 + 20
  Phi  = np.zeros([n+1,nx,nx])
  Beta = np.zeros([degmax+1,nx,nx])
  for k in range(n+1):
    Phi[k,:,:] = F(x0 + (np.cos(np.pi*k/n) + 1.0)*(x1-x0)/2.0)
  for i in range(nx):
    for j in range(nx):
      Phi[:,i,j] = dct(Phi[:,i,j],type=1)/n
      Phi[0,i,j] /= 2.0
      Phi[n,i,j] /= 2.0

  # use Gram-Schmidt to iteratively calculate orthogonal polynomials
  # also calculate the recursion relation coefficients and the
  # expansion coefficients of F(x)
  P0 = np.zeros([2,nx,nx])
  P1 = np.zeros([2,nx,nx])
  P1[0,:,:] = np.eye(nx)
  M = np.linalg.inv(inner_prod(P1,P1,W0))
  Beta[0,:,:] = np.dot(inner_prod(Phi,P1,W0),M)
  for d in range(1,degmax+1):
    P2 = np.zeros([d+1,nx,nx])
    for k in range(1,d):
      P2[k+1,:,:] += P1[k,:,:]/2.0
      P2[k-1,:,:] += P1[k,:,:]/2.0
    P2[1,:,:] += P1[0,:,:]

    B = np.dot(inner_prod(P2,P1,W0),np.linalg.inv(inner_prod(P1,P1,W0)))
    if(d>1):
      C = np.dot(inner_prod(P2,P0,W0),np.linalg.inv(inner_prod(P0,P0,W0)))
    else:
      C = np.zeros([nx,nx])

    Bs[d-1,:,:] = B
    Cs[d-1,:,:] = C

    for k in range(d-1):
      P2[k,:,:] -= np.dot(B,P1[k,:,:])
      P2[k,:,:] -= np.dot(C,P0[k,:,:])
    P2[d-1,:,:] -= np.dot(B,P1[d-1,:,:])

    M = np.linalg.inv(inner_prod(P2,P2,W0))
    Beta[d,:,:] = np.dot(inner_prod(Phi[:,:,:],P2,W0),M)

    P0 = P1
    P1 = P2

    #print("B C M for d=",d)
    #print(B)
    #print(C)
    #print(np.dot(np.linalg.inv(inner_prod(P0,P0,W0)),inner_prod(P1,P1,W0)))

  return Beta, Bs, Cs

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

  ## make companion matrix ##
  comp = np.zeros([nx*degmax,nx*degmax])
  for k in range(0,degmax-1):
    comp[k*nx:(k+1)*nx,(k+1)*nx:(k+2)*nx] = np.eye(nx)
    comp[k*nx:(k+1)*nx,k*nx:(k+1)*nx] = B[k,:,:]
    comp[(k+1)*nx:(k+2)*nx,k*nx:(k+1)*nx] = C[k+1,:,:]
  for k in range(degmax):
    comp[(degmax-1)*nx:degmax*nx,k*nx:(k+1)*nx] = Beta[k,:,:]
  comp[(degmax-1)*nx:degmax*nx,(degmax-1)*nx:degmax*nx] -= np.dot(Beta[degmax,:,:],B[degmax-1,:,:])
  comp[(degmax-1)*nx:degmax*nx,(degmax-2)*nx:(degmax-1)*nx] -= np.dot(Beta[degmax,:,:],C[degmax-1,:,:])


  CC = np.zeros([degmax*nx,degmax*nx])
  for k in range(degmax):
    CC[k*nx:(k+1)*nx,k*nx:(k+1)*nx] = np.eye(nx)
  CC[(degmax-1)*nx:degmax*nx,(degmax-1)*nx:degmax*nx] = -Beta[degmax,:,:]

  evals = linalg.eigvals(comp,b=CC)
  zeros = []
  for val in evals:
    if np.abs(np.imag(val)) < 1e-12 and np.abs(np.real(val)) < 1.0:
      zeros.append(x0 + (np.real(val)+1.0)*(x1-x0)/2.0)

  return zeros, Beta, B, C

