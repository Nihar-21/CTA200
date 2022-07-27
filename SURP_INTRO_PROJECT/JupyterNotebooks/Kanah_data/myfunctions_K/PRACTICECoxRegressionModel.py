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
from lifelines import CoxPHFitter, KaplanMeierFitter
from myfunctions import PRACTICE_PlottingLL  





get_ipython().run_line_magic('matplotlib', 'inline')


def PRACTICECoxRegressionModel(stimes, N, ap_s, ap_s2, ap_s3, ap_s5, ebb_,eap_, ebs, aps_0, mu, stime, Np):      
    
    #ebs = singular value = 0, 0.175, 0.35, 0.525, 0.7
    #stimes = 1050 values ( 5 eb values x 7 ap values x 30 survival times)
    #stime = 30 values for each aps_) value
    #N = 1050
    #Np = 30
    #ap_s, ap_s2, ap_s3 = array of 1050 values used for the data frame
                
    #*************************************EVENT OBSERVATION**************************************************#
    E = np.zeros(N).astype(int)
    
    for i,time in enumerate(stimes):
        if (time > 62831):
            E[i] = 0
        else:
            E[i] = 1

        
    
     #**************************************MAKING A DATA FRAME 2************************************************#
    data1 = {'T':stimes, 'E':E, 'aps':ap_s}
    df = pd.DataFrame(data=data1)

    T = df['T']
    E = df['E']
    aps = df['aps']   
    #print(df)
    
    #************************************COX PH FITTER*************************************#

    m,n = 0,0 #columns & rows 
     
    fig,axes = plt.subplots(5,7)
   
    
    for x in range(0,25,6):
        for y in range(3,10,1):
            
             
            axes[m,n].set_xscale('log')
            axes[m,n].set_ylabel("S(t)")
            
            KT,KE,Kdf = PRACTICE_PlottingLL.PRACTICE_PlottingLL(ebs[x].any(),aps_0[y].any(),mu,stime[x,y, :].any(),Np)
            kmf = KaplanMeierFitter().fit(KT, KE, label='KaplanMeierFitter')
            kmf.plot_survival_function(ax = axes[m,n])

            cph = CoxPHFitter()


            df['aps**3'] = df['aps']**3
            df['aps**5'] = df['aps']**5
            cph.fit(df,duration_col = 'T', event_col = 'E')#, show_progress = True)
            cph.plot_partial_effects_on_outcome(covariates = ['aps', 'aps**3','aps**5'], values = [[1.5,1.5**3,1.5**5],[2.0,2.0**3, 2.0**5],[2.5,2.5**3,2.5**5]], ax = axes[m,n], cmap = 'viridis')





    
    #cph.fit(df,duration_col = 'T', event_col = 'E', show_progress = True, formula = "aps + I(aps**3) + I(aps**5)")
    #cph.fit(df,duration_col = 'T', event_col = 'E', formula = "aps + eb")

#     cph.print_summary()
#     b0 = cph.baseline_survival_
#     coeff = cph.params_
#     conf = cph.confidence_intervals_
#     var = cph.variance_matrix_
#     print(b0)
#     print(coeff)
#     print(conf)
#     print(var)


           
    plt.title('Formula(aps vs. aps3 (1050 values)): (eb,ap,mu)={}'.format(ebs,aps_0,mu), fontsize = 12)
    #plt.savefig('Formula(aps vs. aps3 (all values)) : (eb,ap,mu)={}.png'.format((ebs,round(aps_0,3),mu)))
       
  
            
            
            
 





       
    
    #***************************************PLOTTING*****************************************#

    #values = [1.167,1.333,2.667,2.833,3.00,3.167,3.333,3.50,3.667,3.833,4.00,4.167,4.333,4.50,4.667,4.80,5.00]
    
    #fig,axs = plt.subplots(7,1, figsize = (15,30))
    
    #fig,axs = plt.subplots(1,7, figsize = (15,30))

#     for i in range (0,6,1):
        
#         #axs[i].set_yscale('log')
#         axs[i].set_xscale('log')
#         i = i+1
    
    
    #fig,axes = plt.subplots(7, figsize = (15,30), dpi = 450)
    #i = 0
    
    #FOR LOOP********************************************************************************************

    #     for x in range(3,10,1):
#         plt.xscale('log')
#         plt.ylabel("S(t)")
        
#         KT,KE,Kdf = PlottingLL.PlottingLL(ebs[0],aps_[x],mu,stime[0,x,:],Np)
#         kmf = KaplanMeierFitter().fit(KT, KE, label='KaplanMeierFitter')
#         kmf.plot_survival_function()

#         cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [round(aps_[x],3)])

#         plt.title('(eb,ap,mu)={}'.format((ebs[0],round(aps_[x],3),mu)), fontsize = 7)


#         #kmf = KaplanMeierFitter()
#         #kmf.fit(KT,KE)
#         #kmf.plot_survival_function(ax=axes[i][0])

#FOR LOOOP**********************************************************************************************************

#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [1.5], ax = axs[0])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [1.667], ax = axs[1])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [1.833],ax = axs[2])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [2.0], ax = axs[3])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [2.167], ax = axs[4])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [2.33], ax = axs[5])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [2.50], ax = axs[6])


  
                                        
    
    
    
    
    

    
    

 
