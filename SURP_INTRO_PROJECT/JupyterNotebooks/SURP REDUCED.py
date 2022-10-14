#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import save
import rebound
import random
from multiprocessing import Pool
from myfunctions import Simulation
from myfunctions import Plotting



if __name__ == "__main__":
    random.seed(1)
    Ne, Na,Nmu, Np = 2,2,2,10 #Np = number of test particles per (eb,ap)tuple 
    ab = 1
    ebs = np.linspace(0.,0.7,Ne)
    aps = ab*np.linspace(1.,5.,Na)
    mus = np.linspace(0.1,0.9,Nmu)

    params = [(eb,ap,Np,mu) for eb in ebs for ap in aps for mu in mus]
    #print(params[0:0])


    pool = rebound.InterruptiblePool(processes = 16) #add number of processors same as number requested on Sunnyvale

    # TO TIME ONE CALL TO THE METHOD
    import time
    import itertools
    start = time.time()
    num = 1
    for _ in itertools.repeat(None,num):
       stime = pool.map(Simulation.Simulation,params) #survival times
    end = time.time()
    print("Time elapsed is",end-start)
        
    stime = np.array(stime).reshape([Nmu,Ne,Na])
    stime = np.nan_to_num(stime)
    stime = stime.T
    #print(stime)
    save('stime3.npy',stime)
    
    for i,mu in enumerate(mus):
        Plotting.Plotting(ebs,aps,Na,stime[i,:,:],mu)

