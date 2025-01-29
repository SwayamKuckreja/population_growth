#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:49:05 2023

@author: dubbel
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:07:08 2023

@author: dubbel
"""

import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
     



sigma = 1.  # Standard deviation.
mu = 10.  # Mean.
tau = .05  # Time constant.
     
dt = .001 
dtc=dt*5.0 # Time step.
T = 1.  # Total time.
n = int(T / dt)  # Number of time steps.
nc= int(T/ dtc)
t = np.linspace(0., T, n)  # Vector of times.
tc= np.linspace(0., T, nc) 

sigma_bis = sigma * np.sqrt(2. / tau)
sqrtdt = np.sqrt(dt)
     

a=1.0
b=1.0
x = np.zeros(n)
w = np.zeros(n)
g= np.zeros(n)
gc =np.zeros(nc)
xc=np.zeros(nc)
wc=np.zeros(nc)
 
     
w[0]=0
x[0]=1
g[0]=1
xc[0]=1
gc[0]=1
wc[0]=0

for i in range(n - 1):
    w[i + 1] = w[i]+sqrtdt * np.random.randn()
    if (i%5==0):
        wc[i//5]=w[i]
    x[i+1] =x[i]+a*x[i]*dt +b*x[i]*(w[i+1]-w[i])    
    g[i+1]=g[0]*np.exp((a-b*b/2.0)*i*dt+b*w[i])   

dt=5*dt
    
for i in range(nc-1):       
        gc[i]=gc[0]*np.exp((a-b*b/2.0)*i*dt+b*wc[i]) 
        xc[i+1] =xc[i]+a*xc[i]*dt +b*xc[i]*(wc[i+1]-wc[i]) 
    
    
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.plot(t,g,'ro',markersize=1)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(tc, xc, lw=2)
ax.plot(tc,gc,'ro',markersize=1)


ntrials = 10000
X = np.zeros(ntrials)
     
bins = np.linspace(-2., 14., 100)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for i in range(n):
    # We update the process independently for
    # all trials
    X +=  b*b*dt*X*0.5+sqrtdt * np.random.randn(ntrials)
    # We display the histogram for a few points in
    # time
    if i in (5, 50, 900):
        hist, _ = np.histogram(X, bins=bins)
        ax.plot((bins[1:] + bins[:-1]) / 2, hist,
                {5: '-', 50: '.', 900: '-.', }[i],
                label=f"t={i * dt:.2f}")
    ax.legend()
plt.show()
