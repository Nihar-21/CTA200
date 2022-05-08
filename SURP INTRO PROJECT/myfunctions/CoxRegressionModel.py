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
from myfunctions import PlottingLL  





get_ipython().run_line_magic('matplotlib', 'inline')


def CoxRegressionModel(stimes,N,ap_s, ebs, aps_, mu, stime,Np):                    
    #ebs = singular value, eb = 0
    #aps_ = singular value
    #stimes = 210
    #stime = 30 values
                
    #**************EVENT OBSERVATION*****************#
    E = np.zeros(N).astype(int)
    
#     for i in range(0,N,1):
#         if ((stime[i]) > 62831):
#             E[i] = 0
#         else:
#             E[i] = 1

    for i,time in enumerate(stimes):
        #print(time)
        if (time > 62831):
            E[i] = 0
        else:
            E[i] = 1

        
    #***************MAKING A DATA FRAME****************#
    data1 = {'T':stimes, 'E':E, 'aps':ap_s}
    df = pd.DataFrame(data=data1)

    T = df['T']
    E = df['E']
    aps = df['aps']
    
    #print(df)
    
    
    #************************************COX PH FITTER*************************************#

    
    cph = CoxPHFitter()
    #cph.fit(df,duration_col = 'T', event_col = 'E')
    cph.fit(df,duration_col = 'T', event_col = 'E', formula = "aps + aps*aps + aps*aps*aps")

    cph.print_summary()
    
    fig,axes = plt.subplots()

    axes.set_xscale('log')
    axes.set_ylabel("S(t)")
        
    KT,KE,Kdf = PlottingLL.PlottingLL(ebs,aps_,mu,stime,Np)
    kmf = KaplanMeierFitter().fit(KT, KE, label='KaplanMeierFitter')
    kmf.plot_survival_function(ax = axes)

    cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [round(aps_,3)], ax = axes)

    plt.title('aps^4(eb,ap,mu)={}'.format((ebs,round(aps_,3),mu)), fontsize = 12)
    #plt.savefig('aps^4(eb,ap,mu)={}.png'.format((ebs,round(aps_,3),mu)))

    
    #Goodness of Fit 
    #cph.check_assumptions(df) 
    #******************************Prediction***********************************************#
#     cph.predict_survival_function(df)
#     cph.predict_median(df)
#     cph.predict_partial_hazard(df)
    #cph.plot('logx')
    
    
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
#         plt.savefig('(eb,ap,mu)={}.png'.format((ebs[0],round(aps_[x],3),mu)))

#FOR LOOOP**********************************************************************************************************
 


    

  

        
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [1.5], ax = axs[0])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [1.667], ax = axs[1])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [1.833],ax = axs[2])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [2.0], ax = axs[3])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [2.167], ax = axs[4])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [2.33], ax = axs[5])
#     cph.plot_partial_effects_on_outcome(covariates = 'aps', values = [2.50], ax = axs[6])


    #plt.savefig('image.png')



                                        
    
    
    
    
    

    
    

 
