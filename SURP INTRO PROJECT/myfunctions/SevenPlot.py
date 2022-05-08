import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


get_ipython().run_line_magic('matplotlib', 'inline')


#stimes = entire 3D array of survival times 
def SevenPlot(stimes,Np,aps): 
    
    #ax = plt.subplot(111)
    
    for x in range(3,10,1):
        stime = stimes[0,x,:]
        
        E = np.zeros(Np).astype(int)
        for i in range(0,30,1):
            if((stime[0,i]) > 62831):
                E[i] = 0
            else:
                E[i] = 1
                
         
        data1 = {'T':stime[0], 'E':E}
        df = pd.DataFrame(data=data1)

        T = df['T']
        E = df['E']
        
        kmf = KaplanMeierFitter()

        kmf.fit(T,E, label = "ap = {}".format(aps[x]))
        kmf.plot_survival_function(at_risk_counts = True)
        
    plt.tight_layout() 
          
            
    plt.title("Survival functions as a function of Planetary Semimajor Axis (ap)")
    
                                              


        
        
        