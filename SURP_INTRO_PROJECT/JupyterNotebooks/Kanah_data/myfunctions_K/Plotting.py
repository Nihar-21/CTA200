#Subplot plotting of colour maps (5 by 5 etc) 

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


def Plotting(ebs,aps,Ne,stime,Np):
    
    print(stime.shape)
    
    
    fig,ax = plt.subplots(nrows =5, ncols = 5, figsize=(45,45))
    l = 0
    for i in range(5):
        for j in range(5):
    #for i in range(Np):
            
        
            extent=[ebs.min(), ebs.max(), aps.min(), aps.max()]

            ax[i,j].set_xlim(extent[0], extent[1])
            ax[i,j].set_ylim(extent[2], extent[3])
            ax[i,j].set_xlabel("Binary Eccentricity $e_b$ ",fontsize =30)
            #print(stime[:,:,l].T)
            ax[i,j].set_ylabel("Test particle semimajor axis $a_p$",fontsize =30)
            im = ax[i,j].imshow(stime[:,:,l].T, aspect='auto', origin="lower", interpolation='nearest', cmap="viridis",extent=extent)
            l = l+1


        #   ebs = np.linspace(0.,0.7,Ne)
            ab_s = np.zeros(Ne)
            #mu = 0.5
            for k,eb in enumerate(ebs):
                #ab_s[k] = 1.6 + 5.1*eb-2.22*(eb**2)+4.12*mu-4.27*eb*mu-5.09*(mu**2)+4.61*(eb**2)*mu**2
                ab_s[k] = 2.278 + 3.824*eb - 1.71*(eb**2)

            ax[i,j].plot(ebs,ab_s,'c', marker = "^",markersize = 7)
            ax[i,j].set_xlabel('$e_b$',fontsize =30)
            ax[i,j].set_ylabel('$a_b(a_c$)',fontsize = 30 )
            ax[i,j].set_title('Planet = {} : $a_c$ vs $e_b$'.format(l),fontsize = 35)


            cb = plt.colorbar(im, ax=ax[i,j])
            cb.solids.set_rasterized(True)
            cb.set_label("Particle Survival Times",fontsize = 25)
            

       # plt.show()
    #plt.savefig("Classic_results.pdf")
    
