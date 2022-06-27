# given survival times of specific (mu,eb, 7 *ap) values, makes a large data frame with all survival times, - with respective ap values --along with each surival time event observation (0 or 1).
# Then conduct survival regression for the ap covariates.
# EVENTUALLY - will need to be able to regress against multiple covariates (including mu and eb values). 



import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib
import numpy as np
import pandas as pd
#FOR SURVIVAL REGRESSION
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter, LogLogisticFitter
from myfunctions import PlottingLL, CureModel  





get_ipython().run_line_magic('matplotlib', 'inline')


def CureModelLL(ebs, aps_0, mu, stime, Np):      
    

                
    #*************************************EVENT OBSERVATION**************************************************#
    E = np.zeros(Np).astype(int)
    
    for i in range(0,30,1):
        if ((stime[0,i]) > 62831):
            E[i] = 0
        else:
            E[i] = 1

    
     #**************************************MAKING A DATA FRAME 2************************************************#
    data1 = {'T':stime[0], 'E':E}
    df = pd.DataFrame(data=data1)

    
    T = df['T']
    E = df['E']
    
   
    
    #************************************CURE MODEL FITTER*************************************#

   
    fig, ax = plt.subplots(figsize=(12, 4))

    #ax.set_ylabel("S(t)")  
    plt.rc('legend', fontsize=(7))
    
        

    cure_model = CureModel.CureFitter().fit(T, E, timeline=np.arange(1, 10000))
    ax = cure_model.plot_survival_function(figsize=(8,4))
    cure_model.print_summary(4)

    kmf = KaplanMeierFitter()
    kmf.fit(T, E).plot(ax=ax)
    
    ax.set_xscale('log')

    p_val = np.array([cure_model.p_])
    lambda_val =  np.array([cure_model.lambda_]) 
    
    return p_val, lambda_val

#     print("---")
#
#     print("Estimated lower bound: {:.2f} ({:.2f}, {:.2f})".format(
#             cure_model.summary.loc['p_', 'coef'],
#             cure_model.summary.loc['lambda_', 'coef'],
#             cure_model.summary.loc['p_',  'coef upper 95%'],
#             cure_model.summary.loc['p_',  'coef lower 95%'],
#         )
#     )

   
    
        
    plt.title('(eb,ap,mu)={}'.format((round(ebs,3),round(aps_0,3),mu)), fontsize = 12)
#     plt.savefig('aps+ aps3+ aps5(210 values)) : (eb,ap,mu)={}.png'.format((round(ebs,3),round(aps_0,3),mu)),dpi = 250)

 







 
