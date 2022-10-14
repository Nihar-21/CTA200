#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import save
import rebound
import random
from multiprocessing import Pool
from myfunctions import SimulationLL as simL
#from myfunctions import Plotting0
#from myfunctions import PlottingLL



if __name__ == "__main__":
    random.seed(1)
    
    Ne, Na,Nmu,Np = 25,25,1,30 #Np = number of test particles per (eb,ap)tuple 
    ab = 1
    ebs = np.linspace(0.,0.7,Ne)
    aps = ab*np.linspace(1.,5.,Na)
    mus = np.array([0.35])


 
    params = [(eb,ap,mu,Np) for eb in ebs for ap in aps for mu in mus]

    pool = rebound.InterruptiblePool(processes = 16) #add number of processors same as number requested on Sunnyvale  
    
    import time
    import itertools
    start = time.time()
    num = 1
    for _ in itertools.repeat(None,num):
        stime = pool.map(simL.SimulationLL,params) #survival times
    end = time.time()
    print("Time elapsed in minutes is {}".format((end-start)/60.0))
    
    #print(stime)
    #print(stime.shape)
    save('stime_eb25_ap25_Mu0.35_Np30_logsp0.npy',stime)
    stime = np.array(stime).reshape([Ne,Na,Nmu,Np])
    stime = np.nan_to_num(stime)
    #stime = stime.T
    print(stime)
    print(stime.shape)
    
    save('stime_eb25_ap25_Mu0.35_Np30_logsp1.npy',stime)
    
#     for i,eb in enumerate(ebs):
#         for j,ap in enumerate(aps):
#             for k,mu in enumerate(mus):
#                 PlottingLL.PlottingLL(stime[i,j,k,:],eb,ap,mu,Np) 
            
#     for i,mu in enumerate(mus):
#         for j in range(0,Np):      
#             Plotting0.Plotting0(ebs,aps,Na,stime[:,:,i,j].T,mu) 
            
            
            
            
            
        


