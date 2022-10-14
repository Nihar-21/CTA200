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


def CRM(stimes1,stimes2,stimes3,stimes4,N1, N2, N3, N4,ap_s1, ap_s2, ap_s3,ap_s4,ebs_0,aps_0,mu,stime,Np):

 
                
    #*************************************EVENT OBSERVATION**************************************************#
    E1 = np.zeros(N1).astype(int)
    for i,time in enumerate(stimes1):
        if (time > 62831):
            E1[i] = 0
        else:
            E1[i] = 1
            
    E2 = np.zeros(N2).astype(int)
    for i,time in enumerate(stimes2):
        if (time > 62831):
            E2[i] = 0
        else:
            E2[i] = 1
            
    E3 = np.zeros(N3).astype(int)
    for i,time in enumerate(stimes3):
        if (time > 62831):
            E3[i] = 0
        else:
            E3[i] = 1
        
    E4 = np.zeros(N4).astype(int)
    for i,time in enumerate(stimes4):
        if (time > 62831):
            E4[i] = 0
        else:
            E4[i] = 1
  
    
     #**************************************MAKING A DATA FRAME 2************************************************#
    data1 = {'T1':stimes1, 'E1':E1, 'aps1':ap_s1}
    df1 = pd.DataFrame(data=data1)
    T1 = df1['T1']
    E1 = df1['E1']
    aps1 = df1['aps1']

    #print(df)
    
    data2 = {'T2':stimes2, 'E2':E2, 'aps2':ap_s2}
    df2 = pd.DataFrame(data=data2)
    T2 = df2['T2']
    E2 = df2['E2']
    aps2 = df2['aps2']
    
    data3 = {'T3':stimes3, 'E3':E3, 'aps3':ap_s3}
    df3 = pd.DataFrame(data=data3)
    T3 = df3['T3']
    E3 = df3['E3']
    aps3 = df3['aps3']
    
    data4 = {'T4':stimes4, 'E4':E4, 'aps4':ap_s4}
    df4 = pd.DataFrame(data=data4)
    T4 = df4['T4']
    E4 = df4['E4']
    aps4 = df4['aps4']
    #************************************COX PH FITTER*************************************#

     
    fig,axes = plt.subplots()
    axes.set_xscale('log')
    axes.set_ylabel("S(t)")
    plt.rc('legend', fontsize=(7))

    
        
    KT,KE,Kdf = PlottingLL.PlottingLL(ebs_0,aps_0,mu,stime,Np)
    kmf = KaplanMeierFitter().fit(KT, KE, label='KaplanMeierFitter')
    kmf.plot_survival_function(ax = axes)
    

    cph = CoxPHFitter()
      
    df1['aps1**3'] = df1['aps1']**3
    df2['aps2**3'] = df2['aps2']**3
    df3['aps3**3'] = df3['aps3']**3
    df4['aps4**3'] = df4['aps4']**3

    
    cph.fit(df1,duration_col = 'T1', event_col = 'E1')# show_progress = True)
#     cph.print_summary()
    cph.plot_partial_effects_on_outcome(covariates = ['aps1', 'aps1**3'], values = [[round(aps_0,3),round(aps_0**3,3)]], ax = axes, cmap = 'viridis')
    
    
    cph.fit(df2,duration_col = 'T2', event_col = 'E2')# show_progress = True)
    cph.plot_partial_effects_on_outcome(covariates = ['aps2', 'aps2**3'], values = [[round(aps_0,3),round(aps_0**3,3)]], plot_baseline = False,ax = axes, cmap = 'spring')
    cph.baseline_survival_.plot(ax=axes, ls = ":", color=f"C{0}")

#     cph.print_summary()

    
    cph.fit(df3,duration_col = 'T3', event_col = 'E3')# show_progress = True)
    cph.plot_partial_effects_on_outcome(covariates = ['aps3', 'aps3**3'], values = [[round(aps_0,3),round(aps_0**3,3)]], plot_baseline = False, ax = axes, cmap = 'Wistia')
    cph.baseline_survival_.plot(ax=axes, ls = ":", color=f"C{2}")

#     cph.print_summary()

    
    cph.fit(df4,duration_col = 'T4', event_col = 'E4')# show_progress = True)
    cph.plot_partial_effects_on_outcome(covariates = ['aps4', 'aps4**3'], values = [[round(aps_0,3),round(aps_0**3,3)]], plot_baseline = False,ax = axes, cmap = 'Pastel1')
    cph.baseline_survival_.plot(ax=axes, ls = ":", color=f"C{4}")

#     cph.print_summary()

#     initial_point = np.array([6.42,-1.27,0.08]), step_size = 0.1-0.5



    
    plt.title('aps+ aps3 (4plot):(eb,ap,mu)={}'.format((round(ebs_0,3),round(aps_0,3),mu)), fontsize = 12)
    plt.savefig('aps+ aps3 (4plot) : (eb,ap,mu)={}.png'.format((round(ebs_0,3),round(aps_0,3),mu)),dpi = 250)


    

    


    
#     b0 = cph.baseline_survival_
#     coeff = cph.params_
#     conf = cph.confidence_intervals_
#     var = cph.variance_matrix_
#     print(b0)
#     print(coeff)
#     print(conf)
#     print(var)
    
    
    
    
  
            
            
            
 






            
            

    

 
