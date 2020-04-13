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



