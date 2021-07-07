#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import save
import rebound
import random
from multiprocessing import Pool 
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

#Rebound Orbital Elements

# In[ ]:


def Simulation(par):
    sim = rebound.Simulation()

    e_b,a_p = par[0],par[1] 
    a_b = 1.
    m1 =1.
   
    sim.add(m=m1, hash = "Star1") 
    
    mu = 0.5
    m2 = (m1*mu)/(1-mu)
    f_b=np.random.rand()*2.*np.pi
    sim.add(m =m2, a= a_b, e=e_b,f=f_b,  hash = "Star2")
    
    e_p = 0
    f_p=np.random.rand()*2.*np.pi
    sim.add(m=0.,a=a_p,e=e_p,f=f_p, hash = "Planet1")
    
    sim.move_to_com()
    max_dist = 1000*a_b

    Torb = 2.*np.pi
    Noutputs = 100
    Norb_max = 1e4 
    Tmin = 0.
    Tmax = Norb_max*Torb
    times = np.linspace(Tmin, Tmax, Noutputs)
   
    tmax_arr = []
    for i,time in enumerate(times):
        sim.integrate(time)
        p = sim.particles[2]
        d2 = p.x**2 + p.y**2 + p.z**2
        tmax_arr.append(time)
        if(d2>max_dist**2):
            break   
            
    #print(tmax_arr[-1])
    
    return tmax_arr[-1] #tmax    
    
        


# In[ ]:


random.seed(1)
N =25
ab = 1
ebs = np.linspace(0.,0.7,N)
aps = ab*np.linspace(1.,5.,N)
params = [(eb,ap) for eb in ebs for ap in aps]


pool = rebound.InterruptiblePool()

# TO TIME ONE CALL TO THE METHOD
import time
import itertools
start = time.time()
num = 1
for _ in itertools.repeat(None,num):
   stime = pool.map(Simulation,params) #survival times
end = time.time()
print("Time elapsed is",end-start)

#stime = pool.map(Simulation,params) #survival times

save('stime1.npy', stime)
Times = np.load('stime1.npy')
print(Times)
stime = np.array(stime).reshape(N,N)
stime = np.nan_to_num(stime)
stime = stime.T
#save('stime.npy',stime)

#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib

t,ax = plt.subplots(1,1,figsize=(7,5))
extent=[ebs.min(), ebs.max(), aps.min(), aps.max()]

ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xlabel("Binary Eccentricity $e_b$ ")
ax.set_ylabel("Test particle semimajor axis $a_p$")
im = ax.imshow(stime, aspect='auto', origin="lower", interpolation='nearest', cmap="viridis",extent=extent)


ebs = np.linspace(0.,0.7,N)
ab_s = np.zeros(N)
for i,eb in enumerate(ebs):
    ab_s[i] = 2.278 + 3.824*eb - 1.71*(eb**2)
   
plt.plot(ebs,ab_s,'c', marker = "^",markersize = 7)
plt.xlabel('$e_b$')
plt.ylabel('$a_b(a_c$)')
plt.title('Critical semimajor axis $a_c$ as a function of eccentricity $e_b$')


cb = plt.colorbar(im, ax=ax)
cb.solids.set_rasterized(True)
cb.set_label("Particle Survival Times")
#plt.show()

#plt.savefig("Classic_results.pdf")

