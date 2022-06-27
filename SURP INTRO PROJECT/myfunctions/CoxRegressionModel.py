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
from myfunctions import PlottingLL, CustomCM 





get_ipython().run_line_magic('matplotlib', 'inline')


def CoxRegressionModel(stimes, N, ap_s, ebs, aps_0, mu, stime, Np):      
    
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

        
    #**************************************MAKING A DATA FRAME 1************************************************#
#     data1 = {'T':stimes, 'E':E, 'aps':ap_s, 'aps2':ap_s2, 'aps3':ap_s3, 'aps5': ap_s5,'eap':eap_, 'eb': ebb_}
#     df = pd.DataFrame(data=data1)

#     T = df['T']
#     E = df['E']
#     aps = df['aps']   
#     aps2 = df['aps2']
#     aps3 = df['aps3']
#     aps5 = df['aps5']
#     eap = df['eap']
#     eb = df['eb']

    #print(df)
    
     #**************************************MAKING A DATA FRAME 2************************************************#
    data1 = {'T':stimes, 'E':E, 'aps':ap_s}
    df = pd.DataFrame(data=data1)

    T = df['T']
    E = df['E']
    aps = df['aps']
#     eb = df['eb']

    #print(df)
    
    #************************************COX PH FITTER*************************************#

     
    fig,axes = plt.subplots()

    axes.set_xscale('log')
    axes.set_ylabel("S(t)")
        
    KT,KE,Kdf = PlottingLL.PlottingLL(ebs,aps_0,mu,stime,Np)
    kmf = KaplanMeierFitter().fit(KT, KE, label='KaplanMeierFitter')
    kmf.plot_survival_function(ax = axes)

    cph = CoxPHFitter()
    
    #cph.fit(df,duration_col = 'T', event_col = 'E', formula = "eap")
    #cph.fit(df,duration_col = 'T', event_col = 'E')
    
#     df['aps**3'] = df['aps']**3
#     df['aps**5'] = df['aps']**5
#     df['eb**2'] = df['eb']**2
#     df['eb*aps'] = df['eb']*df['aps']
#     df['(eb**2)*(aps)'] = df['eb**2']*df['aps']


#     df.isnull().sum()

    
    cph.fit(df,duration_col = 'T', event_col = 'E')# show_progress = True)
    
#     initial_point = np.array([6.42,-1.27,0.08]), step_size = 0.1-0.5
    
    plt.rc('legend', fontsize=(7))
        
    cph.plot_partial_effects_on_outcome(covariates = ['aps'], values = [[aps_0]], ax = axes, cmap = 'viridis')
#     cph.print_summary()
    plt.title('aps+ aps3 +aps5 (210 values):(eb,ap,mu)={}'.format((round(ebs,3),round(aps_0,3),mu)), fontsize = 12)
#     plt.savefig('aps+ aps3+ aps5(210 values)) : (eb,ap,mu)={}.png'.format((round(ebs,3),round(aps_0,3),mu)),dpi = 250)

    return df 





#**************************************SET 1 & SET 4 & SET 5 & SET 6 & SET 7***************************************#
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps', 'aps**3','aps**5'], values = [[aps_0,aps_0**3,aps_0**5]], ax = axes, cmap = 'viridis')
   
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','aps**5','eb'], values = [[aps_0,aps_0**3,aps_0**5,round(ebs,3)]], plot_baseline = False, ax = axes, cmap = 'autumn')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','aps**5','eb', 'eb**2'], values = [[aps_0,aps_0**3,aps_0**5,round(ebs,3),round(ebs**2,3)]],plot_baseline = False, ax = axes, cmap = 'winter')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','aps**5','eb', 'eb**2', 'eb*aps'], values = [[aps_0,aps_0**3,aps_0**5,round(ebs,3),round(ebs**2,3),round(ebs*aps_0,3)]],plot_baseline = False, ax = axes, cmap = 'spring')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','aps**5','eb','eb*aps'], values = [[aps_0,aps_0**3,aps_0**5,round(ebs,3),round(ebs*aps_0,3)]],plot_baseline = False, ax = axes, cmap = 'Pastel1')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','aps**5','eb', 'eb**2','(eb**2)*(aps)','eb*aps'], values = [[aps_0,aps_0**3,aps_0**5,round(ebs,3),round(ebs**2,3),round((ebs**2)*(aps_0)),round(ebs*aps_0,3)]],plot_baseline = False, ax = axes, cmap = 'Wistia')
        
#     cph.print_summary()
    
       
#     plt.title('Set 5 (18,750 values):(eb,ap,mu)={}'.format((round(ebs,3),round(aps_0,3),mu)), fontsize = 12)
#     plt.savefig('Set 5 (18 750 values)) : (eb,ap,mu)={}.png'.format((round(ebs,3),round(aps_0,3),mu)),dpi = 250)

#***************************************SET 2 & SET 8*************************************************************************# 
    
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps', 'aps**3'], values = [[aps_0,aps_0**3]], ax = axes, cmap = 'viridis')
   
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','eb'], values = [[aps_0,aps_0**3,round(ebs,3)]], plot_baseline = False, ax = axes, cmap = 'autumn')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','eb', 'eb**2'], values = [[aps_0,aps_0**3,round(ebs,3),round(ebs**2,3)]],plot_baseline = False, ax = axes, cmap = 'winter')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','eb', 'eb**2', 'eb*aps'], values = [[aps_0,aps_0**3,round(ebs,3),round(ebs**2,3),round(ebs*aps_0,3)]],plot_baseline = False, ax = axes, cmap = 'spring')

#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','eb','eb*aps'], values = [[aps_0,aps_0**3,round(ebs,3),round(ebs*aps_0,3)]],plot_baseline = False, ax = axes, cmap = 'Pastel1')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','eb', 'eb**2','(eb**2)*(aps)','eb*aps'], values = [[aps_0,aps_0**3,round(ebs,3),round(ebs**2,3),round((ebs**2)*(aps_0)),round(ebs*aps_0,3)]],plot_baseline = False, ax = axes, cmap = 'Wistia')

#     cph.print_summary()
#     plt.title('Set 8 (18,750 values):(eb,ap,mu)={}'.format((round(ebs,3),round(aps_0,3),mu)), fontsize = 12)
#     plt.savefig('Set 8 (18,750 values)) : (eb,ap,mu)={}.png'.format((round(ebs,3),round(aps_0,3),mu)),dpi = 250)
    
    
#*************************************SET 3 & SET 9*************************************************************************#
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps'], values = [[aps_0]], ax = axes, cmap = 'viridis')
   
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','eb'], values = [[aps_0,round(ebs,3)]], plot_baseline = False, ax = axes, cmap = 'autumn')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','eb', 'eb**2'], values = [[aps_0,round(ebs,3),round(ebs**2,3)]],plot_baseline = False, ax = axes, cmap = 'winter')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','eb', 'eb**2', 'eb*aps'], values = [[aps_0,round(ebs,3),round(ebs**2,3),round(ebs*aps_0,3)]],plot_baseline = False, ax = axes, cmap = 'spring')

#     cph.plot_partial_effects_on_outcome(covariates = ['aps','eb','eb*aps'], values = [[aps_0,round(ebs,3),round(ebs*aps_0,3)]],plot_baseline = False, ax = axes, cmap = 'Pastel1')
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','eb', 'eb**2','(eb**2)*(aps)','eb*aps'], values = [[aps_0,round(ebs,3),round(ebs**2,3),round((ebs**2)*(aps_0)),round(ebs*aps_0,3)]],plot_baseline = False, ax = axes, cmap = 'Wistia')

#     cph.print_summary()
    
       
#     plt.title('Set 9 (18,750 values):(eb,ap,mu)={}'.format((round(ebs,3),round(aps_0,3),mu)), fontsize = 12)
#     plt.savefig('Set 9 (18,750 values)) : (eb,ap,mu)={}.png'.format((round(ebs,3),round(aps_0,3),mu)),dpi = 250)

#****************************************************************************************************************#

    
#     b0 = cph.baseline_survival_
#     coeff = cph.params_
#     conf = cph.confidence_intervals_
#     var = cph.variance_matrix_
#     print(b0)
#     print(coeff)
#     print(conf)
#     print(var)
    
#******************************************************************************************************************#
    
    
    
    
    
    
    
    
    
#     cph.plot_partial_effects_on_outcome(covariates = ['aps','aps**3','aps**5','eb', 'eb**2', 'eb*aps'], values = [[1.5,1.5**3,1.5**5,0.5, 0.5**2, 0.5*1.5],[2.0,2.0**3,2.0**5,0.5,0.5**2,0.5*2.0],[2.5,2.5**3,2.5**5,0.5,0.5**2,0.5*2.5]], ax = axes, cmap = 'winter')
    
#     print("Summary 2")
#     cph.print_summary()
#     b1 = cph.baseline_survival_
#     print(b1)

    
    #cph.fit(df,duration_col = 'T', event_col = 'E', show_progress = True, formula = "aps + I(aps**3) + I(aps**5)")
    #cph.fit(df,duration_col = 'T', event_col = 'E', formula = "aps + eb")


           
  
       
        
            
        #cph.plot_partial_effects_on_outcome(['aps', 'eb'], values = [[2.5,0],[2.3,0]], plot_baseline = False, ax = axes, cmap = "Blues")
    #cph.plot_partial_effects_on_outcome(['aps', 'eb'], values = [[2.0,0],[1.5,0]], plot_baseline = False, ax = axes, cmap = "coolwarm")
    #cph.baseline_survival_.plot(ax=axes, ls = ":", color=f"C{i}")


    #cph.fit(df,duration_col = 'T', events_col = 'E', formula = "eb + aps + I(aps**3)")
    #cph.print_summary()
    #cph.plot_partial_effects_on_outcome(covariates = ['aps'], values = [round(aps_0,3)], ax = axes)

            
            
            
            
            
            
 






            
            
    
    #>>>>>>>>>>>>QUESTION: difference between the coefficients we get from no formula vs. those with the formula; do the formulas improve fit that is already provided with no formula? (which is a better estimate?)
   
    #cph.fit(df,duration_col = 'T', event_col = 'E', formula = "aps + I(aps**2) + I(aps**3)") 
    #cph.plot_partial_effects_on_outcome(covariates = ['aps','aps2','aps3'], values = [round(aps_,3),round(aps_,3)], ax = axes)
    #cph.plot_partial_effects_on_outcome(covariates = ['eap'], values = [round(eap_0,3)], ax = axes)

       
    
    #******************************************EXTRA********************************************************#
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


#     for i in range(0,N,1):
#         if ((stime[i]) > 62831):
#             E[i] = 0
#         else:
#             E[i] = 1
                                        
    
    
    
    
    

    
    

 
