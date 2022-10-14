from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from lifelines import KaplanMeierFitter



def KaplanMeierEst(ebs, aps, mu, stime, Np):

    Stime = stime[0]
    Stime.sort()     #Sort the suvrival times in ascending order 

    
    #*******************************MAKING 'ti','ni' & 'di' ARRAYS*******************************#
    ti, cnt = np.unique(Stime, return_counts= True, axis=None) 
    ti,cnt = np.append([1],ti),np.append([0],cnt)

    #*************************************EVENT OBSERVATION***************************************#
    E = np.zeros(len(ti)).astype(int)

    for i in range(1,len(ti),1):
        if ((ti[i]) > 62831):
            E[i] = 0
        else:
            E[i] = 1
            
    #****************************************di & ni**********************************************#     
    di = np.zeros(len(ti))
    for i in range(len(ti)):
        if E[i] == 1:
            di[i] = cnt[i]
    
    ni = np.zeros(len(ti)-1).astype(int)
    ni = np.append([Np], ni)
    
    for i in range(1, len(ni),1):
        ni[i] = (ni[i-1]-di[i-1])
  

    #**************************************SURVIVAL FUNCTION****************************************#
    
    S = np.ones(len(ti))
#     V = np.zeros(len(ti))
    V0 = np.zeros(len(ti))
    
    for i in range(0,len(ti)-1,1):
        S[i+1] = S[i]*(1-(di[i+1]/ni[i+1]))
    
    V0 = S**2*np.cumsum(di/ni/(ni-di))
    
    for i in range(len(ti)):
        if di[i] == ni[i]:
            V0[i] = 0
    
    Err0 = 1.96*np.sqrt(V0)
    Err1 = -1.96*np.sqrt(V0)
    
#     Var_sum = np.zeros(len(ti))
#     Var_sum[0] = di[0]/ni[0]/(ni[0]-di[0])
#     for i in range(0,len(ti)-1,1):
#         Var_sum[i+1] = Var_sum[i]+(di[i+1]/ni[i+1]/(ni[i+1]-di[i+1]))
#     for i in range(len(ti)):
#         V[i] = S[i]**2*Var_sum[i]    
#     S2 = S**2
#     S2 = df2['S2']
#     Var_sum = df2['Var_sum']  
#     V = df2['V']      
           
    #**************************************DATA FRAME***********************************************#
    data1 = {'T':ti, 'cnt':cnt, 'E':E,'ni':ni, 'di':di,'S(t)':S,  'Var':V0, 'Err+': Err0}
    df1 = pd.DataFrame(data=data1)
     
    T = df1['T']         #array of unique survival times (including ti = 0)
    cnt = df1['cnt']     #number of particles with same ti (cnt = 0 at ti = 0)
    E = df1['E']         #particle ejection ('yes' = 1; 'no' = 0)
    ni = df1['ni']       #num of particles at risk of ejection at time ti (includes di)
    di = df1['di']       #num of particles ejected at ti
    S = df1['S(t)']      #Survival probability
    V0 = df1['Var']      #Variance 
    Err0 = df1['Err+']   #Error

    print('*****************************')
    print(df1)
    delta = np.array(E)  #For calculating LOG LIKELIHOOD FUNCTION (0 if ti = T; 1 if ti = ejection - so just used E for now)
    print(delta)

    #**************************************INTERPOLATION******************************************#
    ti = ti.tolist()
    S = S.tolist()
#     if ti[-1] < 62831:
#         ti.append(62831.853072)
#         S.append(S[-1])
#         Err0.append(Err0[-1])
#     print('ti {}'.format(len(ti),ti))
    
    Inter_ti_num = 100*(len(ti)) + len(ti)   #Inter_S_num = Inter_ti_num
    Inter_ti = np.zeros(Inter_ti_num)
    Inter_S = np.zeros(Inter_ti_num)
    Inter_Err0 = np.zeros(Inter_ti_num)

    j = 0
    for i in range(len(ti)-1):
        Inter_ti[j:j+102] = np.linspace(ti[i],ti[i+1],102)
        j = j+101
        
    k = 0  
    for i in range(len(ti)):
        Inter_S[k:k+100] = S[i]
        Inter_Err0[k:k+100] = Err0[i]
        k = k+100
        
    Inter_ti = Inter_ti.tolist()
    Inter_S = Inter_S.tolist()
    Inter_Err0 = Inter_Err0.tolist()

    
#     print('1 Inter_ti:',len(Inter_ti),Inter_ti)
#     print('1:',len(Inter_S),Inter_S)  
#     print('1 Inter_S:',len(Inter_S),Inter_S)  

    index = (len(ti)-1)*100 + (len(ti))
    del Inter_ti[index:]
    del Inter_S[index:]
    del Inter_Err0[index:]
        
#     print('2:',len(Inter_ti),Inter_ti)
#     print('2:',len(Inter_S),Inter_S)  
    
    fig, ax = plt.subplots(1,2,figsize=(12, 4))
    ax[0].set_xlim(10,7000)
    ax[0].set_ylim(-0.2,1.2)
    ax[0].set_xlabel("timeline")
    ax[0].set_ylabel("Survival Probability (S(t))", size = 10)  
#     ax.errorbar(Inter_ti,Inter_S,yerr = Err0,fmt = '*',color = 'orange',ecolor = 'lightgreen', elinewidth = 2, capsize=4)
    ax[0].set_xscale('log')
    ax[0].set_title('Survival Function {}'.format((round(ebs,3),round(aps,3),mu,Np)))
# #     ax.legend(loc='lower left', ncol=2)
    ax[0].plot(Inter_ti,Inter_S)
#     plt.plot(ti, S)


    #*******************ERROR***************************************#
    Inter_ti_np = np.array(Inter_ti)
    Inter_S_np = np.array(Inter_S)
    Inter_Err0_np = np.array(Inter_Err0)
    ax[0].fill_between(Inter_ti_np, Inter_S_np - Inter_Err0_np, Inter_S_np + Inter_Err0_np, facecolor='#7EFF99')
 
                
    #********************************LIFELINES: Event Observation*******************************#
    E_L = np.zeros(Np).astype(int)
    
    for i in range(0,30,1):
        if ((stime[0,i]) > 62831):
            E_L[i] = 0
        else:
            E_L[i] = 1

    
     #*****************************LIFELINES: Making a Data Frame*******************************#
    data1 = {'T_L':stime[0], 'E_L':E_L}
    df = pd.DataFrame(data=data1)
#     print(df)
    
    T_L = df['T_L']
    E_L = df['E_L']
    delta = np.array(E_L)  #For calculating LOG LIKELIHOOD FUNCTION (0 if ti = T; 1 if ti = ejection - so just used E for now)
    print(delta)
    

    #******************************LIFELINES: Kaplan Meier Estimate*****************************#
    kmf = KaplanMeierFitter()
    kmf.fit(T_L, E_L).plot(ax=ax[1])
    ax[1].set_xlabel("timeline")
    ax[1].set_title('LIFELINES {}'.format((round(ebs,3),round(aps,3),mu,Np)))
    plt.show()
    
    
    #*******************************LIKELIHOOD FUNCTION FOR CURE MODEL*******************************#

    #For the cure model S(t) = p + (1-p)*(exp(-t/lambda))
    
    def log_likelihood(theta,t, delta): #S,S_err) : #delta):
        p = theta 
        lambda_ = 100
        delta0 = delta
#         print(p)
        
        h = (1/lambda_)*((p/(1-p))*np.exp(t/lambda_) + 1)**(-1)
        H = -np.log(p + (1-p)*np.exp(-t/lambda_))
        logL = np.sum(delta0*np.log(h)-H)
        return logL

    def log_prior(theta):
        p = theta 
       
        if 0.0<p<1.0: #np.all(0.0<p) and np.all(p<1.0):
           
            return 0.0 
        return -np.inf 
    
    def log_prob(theta, T, delta): # S,Serr): # delta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, T, delta)# S, Serr)# delta)

    from scipy.optimize import minimize 
    np.random.seed()
    nll = lambda *args: log_prob(*args) #/ -log_likelihood() or -log_prob()?
    initial = 0.01 #?
#     soln = minimize(nll,initial, args= (T_L, delta))#S,Err0)) # delta))
    print(len(T_L), len(delta))
#     p = soln.x
    p = initial 
    theta0 = 0.5
    print(log_prob(theta0, T_L, delta)) 

   
    print('p = {}'.format(p))
    
    
    #*******************************************EMCEE**********************************#
    import emcee

#     pos = soln.x + 1e-4 * np.random.randn(32,1)
    pos = initial + 1e-4 * np.random.randn(32,1)

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=(T_L, delta)#, S, Err0)
    )
    sampler.run_mcmc(pos, 1000, progress=True);   #suggested amoungt 5000
    
    
    fig, axes = plt.subplots(1,1, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["p"]
    for i in range(ndim):
#         ax = axes[i]
        ax = axes
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels)
#         ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

#     axes[-1].set_xlabel("step number");
    axes.set_xlabel("step number"); 
    
    tau = sampler.get_autocorr_time()
    print(tau)
    
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)
    
    
    
    import corner

    fig = corner.corner(
        flat_samples, labels=labels, truths=[p]
    );




