
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter,NelsonAalenFitter


get_ipython().run_line_magic('matplotlib', 'inline')


def PlottingLL(Pstime,eb,ap,Np):
    
    
    #***************DATA***********************************************#
    stime = Pstime
    print(stime)
    
    #***************EVENT OBSERVATION**********************************#
    Npp = Np
    E = np.zeros(Npp).astype(int)
    for i,time in enumerate(stime):
        if (time >= 62831):
            E[i] = 0
        else:
            E[i] = 1
            
    #***************MAKING A DATA FRAME*********************************#
    data1 = {'T':stime, 'E':E}
    df = pd.DataFrame(data=data1)
    print(df)

    T = df['T']
    E = df['E']
    #return T,E
    
    #*********************SUBPLOTS***************************************#
    fig,axs = plt.subplots(1,2)
    #*********************KMF FITTING************************************#
    kmf = KaplanMeierFitter()
    kmf.fit(T,E)
    kmf.survival_function_
    #kmf.plot_survival_function()
    kmf.survival_function_.plot(ax=axs[0])    
    
    #***********ESTIMATING HAZARD RATE USING NELSON_AALEN***************#
    naf = NelsonAalenFitter()
    naf.fit(T,E)
    #print(naf.cumulative_hazard_.head())
    #naf.plot_cumulative_hazard()
    naf.cumulative_hazard_.plot(ax=axs[1])
    
    
    plt.suptitle('For (eb,ap) = {}'.format((eb,ap)))

      
    plt.show()
    plt.savefig("Classic_results.pdf")
