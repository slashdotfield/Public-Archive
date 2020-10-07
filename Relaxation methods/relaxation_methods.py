# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:49:48 2020

@author: scfor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

def RelaxationJacobi(u,f,h):
    unew = np.zeros(len(u))
    unew[0] = u[0]
    unew[-1] = u[-1]
    for i in range(1,len(u)-2):
        unew[i] = (u[i-1] + u[i+1]- ((h**2) * f[i]))/2
    return unew

def RelaxationGS(u,f,h):
    for i in range(1,len(u)-2):
        u[i] = (u[i-1] + u[i+1]- ((h**2) * f[i]))/2
    return u

def RelaxationSOR(u,f,h):
    omega = 2/(1 + np.sinh(np.pi*h))
    for i in range(1,len(u)-2):
        ustar = (u[i-1] + u[i+1]- ((h**2) * f[i]))/2
        u[i] = (1-omega)*u[i] + omega * ustar
    return u

def GetRes(u,f,h):
    res = np.zeros(len(u))
    for i in range(2,len(u)-2):
        res[i] = f[i] - (u[i-1]- (2 * u[i]) + u[i+1])/(h**2)
    return res

def GetL2Norm(u,h):
    L2norm = 0
    for i in range(len(u)):
        L2norm += h * (u[i]**2)
    return np.sqrt(L2norm)

#initial values
n = 641
h = 1./(n-1)
x = np.linspace(0,1,n)

#output 1
#source = -2 * (np.pi**2) * np.sin(2 * np.pi * x) - 16 * (np.pi**2) * \
#    np.sin(8 * np.pi * x) + 36 * (np.pi**2) * np.sin(12 * np.pi * x)
#solution = np.sin(2 * np.pi * x) * (np.sin(5 * np.pi *x)**2)

#output 2
source = 144 * (np.pi**2) * np.sin(24 * np.pi * x) - 450 * (np.pi**2) * \
    np.sin(30 * np.pi * x) + 324 * (np.pi**2) * np.sin(36 * np.pi * x)
solution = np.sin(30 * np.pi * x) * (np.sin(3 * np.pi *x)**2)

#simulation parameters
maxsteps = 41308
minerror = 1E-3
framelength = 12

#Jacobi
uJacobi = np.zeros(n)
resJacobi = [GetL2Norm(GetRes(uJacobi,source,h),h)]

start = time.time()
for _ in range(maxsteps):
    uJacobi = RelaxationJacobi(uJacobi,source,h)
    resJacobi.append(GetL2Norm(GetRes(uJacobi,source,h),h))
    if resJacobi[-1] < minerror:
        break
print('Jacobi time: {0}'.format(time.time()-start))

#Gauss-Seidel
uGS = np.zeros(n)
resGS = [GetL2Norm(GetRes(uGS,source,h),h)]

start = time.time()
for _ in range(maxsteps):
    uGS = RelaxationGS(uGS,source,h)
    resGS.append(GetL2Norm(GetRes(uGS,source,h),h))
    if resGS[-1] < minerror:
        break
print('Gauss-Seidel time: {0}'.format(time.time()-start))

#Successive Overrelaxation
uSOR = np.zeros(n)
resSOR = [GetL2Norm(GetRes(uSOR,source,h),h)]

start = time.time()
for _ in range(maxsteps):
    uSOR = RelaxationSOR(uSOR,source,h)
    resSOR.append(GetL2Norm(GetRes(uSOR,source,h),h))
    if resSOR[-1] < minerror:
        break
print('Successive Overrelaxation time: {0}'.format(time.time()-start))

fig = plt.figure()
fig.set_figheight(7)
fig.set_figwidth(7)
gridspec.GridSpec(8,8)

#plot solution
plt.subplot2grid((8,8), (0,0), colspan=8, rowspan=5)
plt.title('Relaxation methods comparison - 1D')
plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.plot(x,uSOR,label='Successive Overrelaxation')
plt.plot(x,uGS,label='Gauss-Seidel')
plt.plot(x,uJacobi,label='Jacobi')
plt.xlim(0,1)
plt.ylim(-1,1)
plt.legend(bbox_to_anchor=(.995, .98), loc=1, borderaxespad=0.)

## plot energies    
plt.subplot2grid((8,8), (6,0), colspan=8, rowspan=2)
plt.title('$L^2$-norm of the residual error')
plt.xlabel('Iteration (N)')
plt.ylabel('$\\log_{10} || r ||_2$')
plt.plot(range(min(len(resSOR),maxsteps)),np.log10(resSOR))
plt.plot(range(min(len(resGS),maxsteps)),np.log10(resGS))
plt.plot(range(min(len(resJacobi),maxsteps)),np.log10(resJacobi))
plt.xlim(0,maxsteps)
plt.ylim(-3,3)
plt.show()