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
from mpl_toolkits import mplot3d



def radius(xi, yi):
    return np.sqrt(xi ** 2 + yi ** 2)




nx=100
ny=nx
ap1=np.zeros((nx,ny))
x=np.linspace(-1,1,nx)
y=np.linspace(-1,1,ny)
X,Y=np.meshgrid(x,y)
R = radius(X, Y)



mask=R<0.5
#R[mask]=1
#print(R)
ap1[mask]=1
#r=math.sqrt(x**2+y**2)
#ap(find(R<0.5))=1

#ap = (dist(256) <= 50.).astype(float)
#beam = ifftn(ap)  
#fig = plt.figure() #fig = plt.figure()
plt.imshow(ap1)

#ax.plot(ap)
plt.show()

# http://allstuffedhere.blogspot.com/2010/06/entry-8-digital-aperture-and-scilab.html

#If = fftn(ap).real
Ifi = ifftn(ap1).real

'''
ax = plt.axes(projection='3d')
fig = plt.figure()
ax.plot_surface(X,Y,If,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
plt.show()
'''


ax = plt.axes(projection='3d')
fig = plt.figure()
ax.plot_surface(X,Y,Ifi,rstride=1,cstride=1,cmap='rainbow',edgecolor='none')
#ax.plot_surface(X,Y,Ifi,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.contour3D(X,Y,Ifi,color='black')
plt.show()



'''
#mask2= ((R<0.5) & (R>0.4)) | ((R<0.3) & (R>0.2)) | (R<0.1)
mask21= (R<0.1)
mask22= ((R<0.3) & (R>0.2)) 
mask23= ((R<0.5) & (R>0.4))
mask24= ((R<0.7) & (R>0.6))
mask25= ((R<0.9) & (R>0.8))

#R[mask]=1
#print(R)
ap2=np.zeros((nx,ny))
ap2[mask21]=1
ap2[mask22]=0.9
ap2[mask23]=0.8
ap2[mask24]=0.7
ap2[mask25]=0.6

plt.imshow(ap2)
plt.show()

Ifi2 = ifftn(ap2).real
ax = plt.axes(projection='3d')
fig = plt.figure()
#ax.plot_surface(X,Y,Ifi2,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.plot_surface(X,Y,Ifi2,rstride=1,cstride=1,cmap='rainbow',edgecolor='none')
ax.contour3D(X,Y,Ifi2,color='black')
plt.show()




#mask2= ((R<0.5) & (R>0.4)) | ((R<0.3) & (R>0.2)) | (R<0.1)
mask31= (R<0.1)
mask32= ((R<0.2) & (R>0.1)) 
mask33= ((R<0.3) & (R>0.2))
mask34= ((R<0.4) & (R>0.3))
mask35= ((R<0.5) & (R>0.4))

#R[mask]=1
#print(R)
ap3=np.zeros((nx,ny))
ap3[mask31]=1
ap3[mask32]=0.9
ap3[mask33]=0.8
ap3[mask34]=0.7
ap3[mask35]=0.6

plt.imshow(ap3)
plt.show()

Ifi3 = ifftn(ap3).real
ax = plt.axes(projection='3d')
fig = plt.figure()
#ax.plot_surface(X,Y,Ifi2,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.plot_surface(X,Y,Ifi3,rstride=1,cstride=1,cmap='rainbow',edgecolor='none')
ax.contour3D(X,Y,Ifi3,color='black')
plt.show()





#mask2= ((R<0.5) & (R>0.4)) | ((R<0.3) & (R>0.2)) | (R<0.1)
mask41= (R<0.1)
mask42= ((R<0.2) & (R>0.1)) 
mask43= ((R<0.3) & (R>0.2))
mask44= ((R<0.4) & (R>0.3))
mask45= ((R<0.5) & (R>0.4))

#R[mask]=1
#print(R)
ap4=np.zeros((nx,ny))
ap4[mask41]=0.6
ap4[mask42]=0.7
ap4[mask43]=0.8
ap4[mask44]=0.9
ap4[mask45]=1

plt.imshow(ap4)
plt.show()

Ifi4 = ifftn(ap4).real
ax = plt.axes(projection='3d')
fig = plt.figure()
#ax.plot_surface(X,Y,Ifi2,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.plot_surface(X,Y,Ifi4,rstride=1,cstride=1,cmap='rainbow',edgecolor='none')
ax.contour3D(X,Y,Ifi4,color='black')
plt.show()





#mask2= ((R<0.5) & (R>0.4)) | ((R<0.3) & (R>0.2)) | (R<0.1)
mask51= (R<0.1)
mask52= ((R<0.3) & (R>0.2)) 
mask53= ((R<0.5) & (R>0.4))
mask54= ((R<0.7) & (R>0.6))
mask55= ((R<0.9) & (R>0.8))

#R[mask]=1
#print(R)
ap5=np.zeros((nx,ny))
ap5[mask51]=0.6
ap5[mask52]=0.7
ap5[mask53]=0.8
ap5[mask54]=0.9
ap5[mask55]=1

plt.imshow(ap5)
plt.show()

Ifi5 = ifftn(ap5).real
ax = plt.axes(projection='3d')
fig = plt.figure()
#ax.plot_surface(X,Y,Ifi2,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.plot_surface(X,Y,Ifi5,rstride=1,cstride=1,cmap='rainbow',edgecolor='none')
ax.contour3D(X,Y,Ifi5,color='black')
plt.show()
'''





ap6=np.zeros((nx,ny))
mask61= (R<0.1)

#R[mask]=1
#print(R)
ap6[mask61]=1

plt.imshow(ap6)
plt.show()

Ifi6 = ifftn(ap6).real
ax = plt.axes(projection='3d')
fig = plt.figure()
#ax.plot_surface(X,Y,Ifi2,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.plot_surface(X,Y,Ifi6,rstride=1,cstride=1,cmap='rainbow',edgecolor='none')
ax.contour3D(X,Y,Ifi6,color='black')
plt.show()



ap7=np.zeros((nx,ny))
mask71= (R<0.05)
#R[mask]=1
#print(R)
ap7[mask71]=1


plt.imshow(ap7)
plt.show()

Ifi7 = ifftn(ap7).real
ax = plt.axes(projection='3d')
fig = plt.figure()
#ax.plot_surface(X,Y,Ifi2,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.plot_surface(X,Y,Ifi7,rstride=1,cstride=1,cmap='rainbow',edgecolor='none')
ax.contour3D(X,Y,Ifi7,color='black')
plt.show()



from numpy.fft import ifftn,fft2

ap8=np.zeros((nx,ny))
mask81= (R<0.05)
#R[mask]=1
#print(R)
ap8[mask81]=1


plt.imshow(ap8)
plt.show()

Ifi8 = fft2(ap8).real
ax = plt.axes(projection='3d')
fig = plt.figure()
#ax.plot_surface(X,Y,Ifi2,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.plot_surface(X,Y,Ifi8,rstride=1,cstride=1,cmap='rainbow',edgecolor='none')
ax.contour3D(X,Y,Ifi8,color='black')
plt.show()


print('(9)')
#sigmax=1
#sigmay=sigmax
#aa=10

def g1d(xi,x0,sigmax,ai):
    g1dx=np.exp(-0.5*((xi-x0)**2/sigmax**2))/ai
    return g1dx


ap90=np.ones((nx,ny))
gx=g1d(x,0,1,10)
gy=g1d(y,0,1,10)
#gx=np.exp(-0.5*((x-0)**2/sigmax**2))/aa
#gy=np.exp(-0.5*((y-0)**2/sigmay**2))/aa

GX,GY=np.meshgrid(gx,gy)
GR = radius(GX, GY)
ap9=ap90*GR
Ifi9 = fft2(ap9).real
#mask91= (R<0.05)
#R[mask]=1
#print(R)
#ap9[mask91]=1
#ap9=np.exp(2*x * np.pi * np.arange(nx) / nx)


fig = plt.figure()
plt.imshow(ap9)



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x,y,ap9,rstride=1,cstride=1,cmap='rainbow',edgecolor='none')
plt.show()


ax = plt.axes(projection='3d')
fig = plt.figure()
ax.plot_surface(X,Y,Ifi9,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
#ax.contour3D(X,Y,Ifi9,color='black')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(GX,GY,Ifi9,cmap='rainbow',linewidth=0.1)
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(GX,GY,Ifi9,cmap='rainbow',linewidth=0.1)
#ax.set_zlim(0,1)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(GX,GY,Ifi9,100,cmap='rainbow',linewidth=0.1)
plt.show()



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(GX,GY,Ifi9,cmap='rainbow',linewidth=0.5)
plt.show()








ax.view_init(45,35)
fig




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


