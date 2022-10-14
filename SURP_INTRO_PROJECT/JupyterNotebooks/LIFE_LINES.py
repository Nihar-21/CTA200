import numpy as np
from numpy import save
import rebound
import random
from multiprocessing import Pool
from myfunctions import SimulationLL as simL


def HW_crit(e,m):
    # eq 3. from H&W(1999)
    # e,m = eb, mu
    a_crit = 1.6 + 5.1*e - 2.22*e**2 + 4.12*m - 4.27*e*m - 5.09*m**2 + 4.61*e**2*m**2
    return a_crit


if __name__ == "__main__":
    random.seed(1)
    
    Na1, Na2 = 4, 4
    Ne, Na,Nmu,Np = 5,Na1+(Na2*2)-2,1,10 #Np = number of test particles per (eb,ap)tuple 
#     Ne,Na, Nmu, Np =  1,10,1,10
    #Na = 50
    ab = 1
    ebs = np.array([0])

#     ebs = np.linspace(0.,0.9,Ne) #changed from 0.7 to 0.9 (07/28/22)
    aps = ab*np.linspace(1.,5.,Na)
    mus = np.array([0.5])
    
 
#     params = [(eb,ap,mu,Np) for eb in ebs for ap in aps for mu in mus]
#     print(params)

    params = []
    
    a_array = np.zeros((Na,Ne))
    e_array = np.zeros((Na,Ne))
    
    print(a_array)
    print(e_array)
    
    for i,e in enumerate(ebs):
        for j,mu in enumerate(mus):
            ac = HW_crit(e,mu)
            upper, lower = ac*1.15, ac*0.75
            av = np.append(np.linspace(1.,lower, Na2)[:-1],np.linspace(lower,upper,Na1))
            av = np.append(av, np.linspace(upper, 5.,Na2)[1:])
#             for a in av:
            for a,k in enumerate(av):
                params.append((e,a,mu,Np))
                a_array[i,k] = a
                e_array[i,k] = e
    
    
    pool = rebound.InterruptiblePool(processes = 16) #add number of processors same as number requested on Sunnyvale  

    import time
    import itertools
    start = time.time()
    num = 1
    for _ in itertools.repeat(None,num):
        stime = pool.map(simL.SimulationLL,params) #survival times
    end = time.time()
    print("Time elapsed in minutes is {}".format((end-start)/60.0))
    
    
    stime = np.array(stime).reshape([Ne,Na,Nmu,Np])
    stime = np.nan_to_num(stime)
    
    print(stime)
    print(stime.shape)
    
    #     params = [(eb,ap,mu,Np) for eb in ebs for ap in aps for mu in mus]
    params = []

    a_array = np.zeros((Na,Ne))
    stime_a = np.zeros((Na,Ne))
    e_array = np.zeros((Na,Ne))

    for i,e in enumerate(ebs):
        for j,mu in enumerate(mus):
            ac = HW_crit(e,mu)
            upper, lower = ac*1.15, ac*0.75
            av = np.append(np.linspace(1.,lower, Na2)[:-1],np.linspace(lower,upper,Na1))
            av = np.append(av, np.linspace(upper, 5.,Na2)[1:])
    #             for a in av:
            for k,a in enumerate(av):
                params.append((e,a,mu,Np))
                a_array[k,i] = a
                e_array[k,i] = e
                stime_a[k,i] = np.mean(stime[i,k,0,:])

    print(a_array)
    print(e_array)
    print(params)
    print(stime_a)
    
    
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(e_array, a_array, stime_a)
    fig.colorbar(cp) # Add a colorbar to a plot

    ab_s = np.zeros(Ne)
    for i,eb in enumerate(ebs):
        ab_s[i] = 1.6 + 5.1*eb-2.22*(eb**2)+4.12*mu-4.27*eb*mu-5.09*(mu**2)+4.61*(eb**2)*mu**2
       #ab_s[i] = 2.278 + 3.824*eb - 1.71*(eb**2)

    ax.plot(ebs,ab_s,'c', marker = "^",markersize = 7)
    ax.set_xlabel('$e_b$')
    ax.set_ylabel('$a_b(a_c$)')
    # plt.title(' Mu {} Critical semimajor axis $a_c$ as a function of eccentricity $e_b$'.format(mu))
    plt.show()
