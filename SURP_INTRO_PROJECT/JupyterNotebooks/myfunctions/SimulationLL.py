
import numpy as np
from numpy import save
import rebound
import random
from multiprocessing import Pool 



def SimulationLL(par):
    
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    
    e_b,a_p,mu,Np = par[0],par[1], par[2], par[3]
#     print('params = {}',e_b,a_p,mu,Np)
    #**********STAR1***********
    a_b = 1.
    m1 =1.
    sim.add(m=m1, hash = "Star1") 
   
    #***********STAR2**********
    m2 = (m1*mu)/(1-mu)
    f_b=np.random.rand()*2.*np.pi
    sim.add(m =m2, a= a_b, e=e_b,f=f_b,  hash = "Star2")
    
    #*********TEST PARTCILES*****
    for p in range(Np):
        sim.add(m=0.,a=a_p,e=0,f=np.random.rand()*2.*np.pi)
    
    #*****RUN SIMULATION & ARCHIVE*****
    sim.move_to_com()
    
#***********************SIM ARCHIVE********************************************************************************* 
   # sim.automateSimulationArchive("archive_eb{:.3f}_ap{:.3f}_Mu{:.3f}_Np{:.2f}_logsp.bin".format(e_b,a_p,mu,Np),interval = 1e3, deletefile = True)


#*********************TIME LOGSPACE*************************#
    max_dist = 100*a_b
    Torb = 2.*np.pi
    Norb_max = 1e4 
    Tmax = Norb_max*Torb
    Tmin = 10*Torb
    Noutputs = 100
    times = np.logspace(np.log10(Tmin),np.log10(Tmax),Noutputs)

#***************SETTING TIME************************#
#     tfinal = 2e3
# #     tfinal = 1e4*2*np.pi
# #     times = np.append(np.linspace(0,1000,10000),np.linspace(1000,1e4*2*np.pi,10000))
#     times = np.append(np.linspace(0,500,10000),np.linsapce(500,tfinal,1000)[1:])
#     print(times)
    
    survtime= np.zeros(Np)  #survival times array
    
    for i,time in enumerate(times):
        
        sim.integrate(time, exact_finish_time = 0)
#         sim.integrate(time, exact_finish_time = False)  #changed from exact_finish_time = 0 (07/28/22)


        
        for j in reversed(range(2,sim.N)):
#             print('sim start 1')
#             print('************************')
            p = sim.particles[j]
#             o = sim.particles[j].calculate_orbit()
#             p0 = sim.particles[0]
#             p1 = sim.particles[1]
#             d = np.sqrt(p.x**2 +p.y**2)
#             d0 = np.sqrt((p.x-p0.x)**2 +(p.y-p0.y)**2)
#             d1 = np.sqrt((p.x-p1.x)**2 + (p.y-p1.y)**2)
#             print('sim start 2')
            
            if (p.x**2 + p.y**2) > 20**2: #or o.e > 1.0:    #was 100*2, should have been 100**2 (07/28/22)
#             if d > 20 or d0 < 0.25 or d1 <0.25 or o.e>1.0:
                survtime[j-2] = time      #sim.t??
#                 print(f'planet {j} ejected at time = {time/60:.5f} mins, (r = {np.sqrt(p.x**2 + p.y**2):.5f},e={o.e:.5f})')
                sim.remove(j)
#                 print('{0} planets remaining',sim.N-2)
#                 print(f'{sim.N-2} planets remaining')
        
#         if time in times[::1000]:
#             print(f'a = {ap}, e = {eb}, mu = {mu}: time = {time}, {sim.N-2} planets remaining')
        
        if sim.N==2:
            break

    
    survtime[(survtime==0)] = time
    #print(survtime)
    
    print("simulation finished, {} planets remaining, mean survival time = {}".format(len(sim.particles)-2),np.mean(survtime))
    
    #print(np.mean(survtime))
    return survtime   

    
    
    
    
    
    
#**************************SIM ARCHIVE*****************************************************************    
# CHECKING SIMULATION ARCHIVE RETREIVAL
#     sa = rebound.SimulationArchive("archive_eb{:.3f}_ap{:.3f}_Np{:.2f}.bin".format(e_b,a_p,Np)) 
#     print("Number of snapshots: %d" % len(sa))
#     print("Time of first and lastsnap shots are: %.1f, %.1f" % (sa.tmin, sa.tmax))
#     sim = sa[1]
#     print(sim.t, sim.particles[2])
#     print("Survival time ",tmax_arr[-1])



