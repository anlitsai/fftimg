#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:58:59 2020

@author: altsai
"""
# https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html#and-n-d-discrete-fourier-transforms


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft,fftn,ifftn
import math

'''
# Number of sample points
N = 300
# sample spacing
T = 1.0 / 600.0
a1=1
a2=10
b1=0.7
b2=50
c1=0.5
c2=120
d1=0.2
d2=190


x = np.linspace(0.0, N*T, N)
y1=a1*np.sin(a2 * 2.0*np.pi*x)
#y1=a1*np.exp(1-2*np.pi*x)
y2=b1*np.sin(b2 * 2.0*np.pi*x)
y3=c1*np.sin(c2 * 2.0*np.pi*x)
y4=d1*np.sin(d2 * 2.0*np.pi*x)
y = y1+y2+y3+y4
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)


ttl1a=str('y1=sin(')+str(a2)+str('*2\u03c0x)')
ttl1b=str('y2=')+str(b1)+str('*sin(')+str(b2)+str('*2\u03c0x)')
ttl1c=str('y3=')+str(c1)+str('*sin(')+str(c2)+str('*2\u03c0x)')
ttl1d=str('y4=')+str(d1)+str('*sin(')+str(d2)+str('*2\u03c0x)')

#plt.plot(x,y)
plt.plot(x,y1,label=ttl1a)
plt.plot(x,y2,label=ttl1b)
plt.plot(x,y3,label=ttl1c)
plt.plot(x,y4,label=ttl1d)
#ttl1=str('y1,y2,y3')
#plt.title(ttl1)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.grid()
plt.show()

plt.plot(x,y,label='y=y1+y2+y3+y4')


#plt.title('y=y1+y2+y3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.grid()
plt.show()


plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlabel('Spatial Frequency')
plt.ylabel('Intensity')
plt.grid()
plt.show()
'''

from mpl_toolkits import mplot3d

def f(xi, yi):
    return np.sin(np.sqrt(xi ** 2 + yi ** 2)) #/(math.sqrt(2*math.pi))

def f2pi(xi, yi):
    return np.sin(np.sqrt((xi*2*np.pi) ** 2 + (yi*2*np.pi) ** 2)) #/(math.sqrt(2*math.pi))



a=6
n=50

x = np.linspace(-a, a, n)
y = np.linspace(-a, a, n)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
Z2pi = f2pi(X, Y)
Zfft = fftn(Z).real


fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 4, cmap='binary')
ax.contour3D(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#from LightPipes import *

#x2d = np.linspace(-N*T, N*T, N*2)
#y2d = np.linspace(-N*T, N*T, N*2)
#z2d = np.meshgrid(x2d,y2d)
#amp=a1*np.sin(a2 * 2.0*np.pi*x)
#y1=a1*np.sin(a2 * 2.0*np.pi)

#y2d=np.sin(x2d)
#z2d=np.zeros(x.size)
#ax.plot(x2d,y2d,z2d)
ax.plot_surface(X,Y,Z,cmap='rainbow')
ax.plot_wireframe(X,Y,Z,color='black',linewidth=0.1)

#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()




ax = plt.axes(projection='3d')
fig = plt.figure()
ax.plot_surface(X,Y,Z2pi,cmap='rainbow')
#ax.plot_wireframe(X,Y,Z,color='black',linewidth=0.1)
plt.show()

#ax.plot_trifurf(X,Y,Z,cmap='rainbow',edgecolor='none')
#plt.show()
ax = plt.axes(projection='3d')
fig = plt.figure()
ax.plot_surface(X,Y,Zfft,cmap='rainbow')

#https://scipy-lectures.org/advanced/image_processing/
#https://vimsky.com/zh-tw/examples/detail/python-method-numpy.fft.ifftn.html


plt.show()
'''
xx = np.linspace(0.0, N*T, N)
yy = np.linspace(0.0, N*T, N)


XX,YY=np.meshgrid(xx,yy)
ZZ=f(XX,YY)

xxf=np.arange(0,N,0.5)
yyf=np.arange(0,N,0.5)
XXf,YYf=np.meshgrid(xxf,yyf)
ZZf=f(XXf,YYf)

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.contour3D(XX,YY,ZZ,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.plot_surface(XXf,YYf,ZZf,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
#cma='rainbow'
#ax.scatter(XX,YY,ZZ,cmap='viridis',linewidth=0.5)
#ax.plot_trisurf(XX,YY,ZZ,cmap='viridis',edgecolor='none')

#ax.wireframe(XX,YY,ZZ,color='black')
ax.view_init(60,35)
ax.set_title('2D')
#yy1=math.sqrt((a1*np.sin(a2 * 2.0*np.pi*x))**2+(a1*np.sin(a2 * 2.0*np.pi*x))**2)
#ax.
'''

'''
import matplotlib.cm as cm
N = 30
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
xf = np.zeros((N,N))
xf[0, 5] = 1
xf[0, N-5] = 1
Z = ifftn(xf)
ax1.imshow(xf, cmap=cm.Reds)
ax4.imshow(np.real(Z), cmap=cm.gray)
xf = np.zeros((N, N))
xf[5, 0] = 1
xf[N-5, 0] = 1
Z = ifftn(xf)
ax2.imshow(xf, cmap=cm.Reds)
ax5.imshow(np.real(Z), cmap=cm.gray)
xf = np.zeros((N, N))
xf[5, 10] = 1
xf[N-5, N-10] = 1
Z = ifftn(xf)
ax3.imshow(xf, cmap=cm.Reds)
ax6.imshow(np.real(Z), cmap=cm.gray)
plt.show()
'''

#ap=float(dist(256)le 50.)
#ap=float(math.d dist(256)le 50.)
#beam=ittf2(ap)


#import numpy.fft.fft2 as fft2
#import numpy.fft.ifft2 as ifft2

#fft2(3,4)


