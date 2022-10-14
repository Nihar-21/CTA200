
import numpy as np
from numpy import save
import rebound
import random
from multiprocessing import Pool 


#get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
# from matplotlib import ticker
# from matplotlib.colors import LogNorm
# import matplotlib


def Simulation(par):
    
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    
    e_b,a_p, mu,Np = par[0],par[1], par[2], par[3]
    
    #**********STAR1***********
    a_b = 1.
    m1 =1.
    sim.add(m=m1, hash = "Star1") 
   
    #***********STAR2**********
    #mu = 0.5  #*******
    m2 = (m1*mu)/(1-mu)
    f_b=np.random.rand()*2.*np.pi
    sim.add(m =m2, a= a_b, e=e_b,f=f_b,  hash = "Star2")
    
    #*********TEST PARTCILES*****
    for p in range(Np):
        sim.add(m=0.,a=a_p,e=0,f=np.random.rand()*2.*np.pi)
    
    #*****RUN SIMULATION & ARCHIVE*****
    sim.move_to_com()
    
#***********************SIM ARCHIVE********************************************************************************* 
#     sim.automateSimulationArchive("archive_eb{:.3f}_ap{:.3f}_Np{:.2f}.bin".format(e_b,a_p,Np),interval = 1e3, deletefile = True)



    max_dist = 100*a_b
    Torb = 2.*np.pi
    Norb_max = 1e4 
    Tmax = Norb_max*Torb
    Tmin = 0
    Noutputs = 100
    times = np.linspace(Tmin, Tmax, Noutputs)
    
    survtime= np.zeros(Np)  #survival times array
    
    for i,time in enumerate(times):
        sim.integrate(time, exact_finish_time = 0)
        
        for j in reversed(range(2,sim.N)):
            p = sim.particles[j]
            if (p.x**2 + p.y**2) > 100*2:
                survtime[j-2] = time
                #print('removing planet {0}',j)
                sim.remove(j)
                #print('{0} planets remaining',sim.N-2)
        
        if sim.N==2:
            break

    
    survtime[(survtime==0)] = time
    #print(survtime)
    
    print('simulation finished, {} planets remianing',(len(sim.particles)-2))
    
    #print(np.mean(survtime))
    #return np.mean(survtime) #instead of all 10 planets individually??
    return survtime   

    
    
    
    
    
    
#**************************SIM ARCHIVE*****************************************************************    
# CHECKING SIMULATION ARCHIVE RETREIVAL
#     sa = rebound.SimulationArchive("archive_eb{:.3f}_ap{:.3f}_Np{:.2f}.bin".format(e_b,a_p,Np)) 
#     print("Number of snapshots: %d" % len(sa))
#     print("Time of first and lastsnap shots are: %.1f, %.1f" % (sa.tmin, sa.tmax))
#     sim = sa[1]
#     print(sim.t, sim.particles[2])
#     print("Survival time ",tmax_arr[-1])



