
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


def Plotting(ebs,aps,Na,stime,mu):
   
    t,ax = plt.subplots(1,1,figsize=(7,5))
    extent=[ebs.min(), ebs.max(), aps.min(), aps.max()]

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Binary Eccentricity $e_b$ ")
    ax.set_ylabel("Test particle semimajor axis $a_p$")
    im = ax.imshow(stime, aspect='auto', origin="lower", interpolation='nearest', cmap="viridis",extent=extent)

    mu = MU
    for i,eb in enumerate(ebs):
        ab_s[i] = 1.6 + 5.1*eb-2.22*(eb**2)+4.12*mu-4.27*eb*mu-5.09*(mu**2)+4.61*(eb**2)*mu**2
       #ab_s[i] = 2.278 + 3.824*eb - 1.71*(eb**2)

    plt.plot(ebs,ab_s,'c', marker = "^",markersize = 7)
    plt.xlabel('$e_b$')
    plt.ylabel('$a_b(a_c$)')
    plt.title('MU = {} : Critical semimajor axis $a_c$ as a function of eccentricity $e_b$'.format(mu))


    cb = plt.colorbar(im, ax=ax)
    cb.solids.set_rasterized(True)
    cb.set_label("Particle Survival Times")
      
    plt.show()
    #plt.savefig("Classic_results.pdf")
