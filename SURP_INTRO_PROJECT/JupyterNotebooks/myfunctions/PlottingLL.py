# given stime, will convert data to a Data Frame with Event Observations Data (0 or 1)
# Then plots per given eb,ap,mu value (the number of planets for each simulation)
# subplot format



import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter,NelsonAalenFitter
from lifelines import ExponentialFitter
from lifelines import PiecewiseExponentialFitter 
#FOR SURVIVAL REGRESSION
# from lifelines import CoxPHFitter
# import rebound





get_ipython().run_line_magic('matplotlib', 'inline')


def PlottingLL(eb,ap,mu,stime,Np):                    
   
                
    #**************EVENT OBSERVATION*****************#
    E = np.zeros(Np).astype(int)
    
    #may need to change to stime[0,i]
    for i in range(0,Np,1):
        if ((stime[i]) > 62831):
            E[i] = 0
        else:
            E[i] = 1


#     for i,time in enumerate(stime):
#         print(time[i])
#         if (time[i] > 62831):
#             E[i] = 0
#         else:
#             E[i] = 1




    #***************LOAD SIMULATION ARCHIVE************#
#     sa = rebound.SimulationArchive("archive_eb{:.3f}_ap{:.3f}_Mu{:.3f}_Np{:.2f}_logsp.bin".format(eb,ap,mu,Np)) 
#     print("Number of snapshots: %d" % len(sa))
#     print("Time of first and lastsnap shots are: %.1f, %.1f" % (sa.tmin, sa.tmax))
#     sim = sa[1]
#     print(sim.t, sim.particles[2])
#     print("Survival time ",tmax_arr[-1])


        
    #***************MAKING A DATA FRAME****************#
    data1 = {'T':stime[0], 'E':E}
    df = pd.DataFrame(data=data1)

    
    T = df['T']
    E = df['E']
    
#*****************For non-simulations just return T,E,df for CRM or CureModelLL.py**********************#
#     return T,E,df
#*******************************************************************************************************#

#     #*********************SUBPLOTS***************************************#
#     fig,axs = plt.subplots(1,1)
#     #*********************KMF FITTING************************************#
  
#     plt.xscale('log')
#     #axs[1].set_xscale('log')
#     kmf = KaplanMeierFitter()
#     kmf.fit(T,E)
    
#     kmf.survival_function_
#     kmf.survival_function_.plot(ax=axs)
#     kmf.plot_survival_function(ax=axs[0], at_risk_counts = True)
#     print(kmf.median_survival_time_) 
    
    
#****************************************FOR LIFELINES.ipynb**************************************#    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xscale('log')
    ax.set_yscale('log')
    kmf = KaplanMeierFitter()
    kmf.fit(T, E).plot(ax=ax)
    plt.title('For (eb,ap,mu) = {}'.format((eb,round(ap,3),mu)))
#*************************************************************************************************#

  

    
#     #***********ESTIMATING CUMULATIVE HAZARD RATE USING NELSON_AALEN***********#
#     naf = NelsonAalenFitter()
#     naf.fit(T,E)
#     #naf.cumulative_hazard_.plot(ax=axs[1])
#     #axs[1].set_yscale('log')
#     naf.plot_cumulative_hazard(ax=axs[1])
    
    #********************HAZARD FUNCTION***************************#
#     bandwidth = 8.0
#     naf.plot_hazard(ax = axs[0],bandwidth=bandwidth)
    
    
#     plt.suptitle('For (eb,ap,mu) = {}'.format((eb,round(ap,3),mu)))
    
    
#     #***********EXPONENTIAL FITTER**********************************#
#     epf = ExponentialFitter().fit(T,E)
#     #epf.plot_hazard(ax=axs[0]) 
#     epf.plot_cumulative_hazard(ax=axs[1])
#     epf.print_summary(3)
    
    
#     #************PIECEWISEEXPONENTIALFITTER*************************#
#     pf = PiecewiseExponentialFitter(breakpoints=[40,60]).fit(T,E)
#     ax = pf.plot(ax=axs[1])
#     #pf.plot_hazard(ax=axs[0])
#     #ax = naf.plot(ax=ax, ci_show = False)
#     pf.print_summary(3)
    
    
#     #**********IMAGE SAVING*****************************************#
#     a= 0
#     plt.savefig("PHYS4010H_S_H_Func.png")
#     a= a+1
#     plt.show()
