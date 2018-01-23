from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_atlas():
    """"""
    ra0, dec0 = np.radians(77.16), np.radians(46.92 - 90)

    # euler rotations
    D = np.array([[np.cos(ra0), np.sin(ra0), 0], [-np.sin(ra0), np.cos(ra0), 0], [0, 0, 1]])
    C = np.array([[np.cos(dec0), 0, np.sin(dec0)], [0, 1, 0], [-np.sin(dec0), 0, np.cos(dec0)]])
    B = np.diag(np.ones(3))

    R = np.dot(B, np.dot(C, D))
    Rinv = np.linalg.inv(R)
    
    l0 = np.linspace(0, 2*np.pi, 500)
    b0 = np.zeros(500)

    xeq, yeq, zeq = myutils.eq2car(l0, b0)
    eq = np.column_stack((xeq, yeq, zeq))

    eq_rot = np.zeros(np.shape(eq))
    for i in range(np.size(l0)):
        eq_rot[i] = np.dot(Rinv, eq[i])
    
    l0_rot, b0_rot = myutils.car2eq(eq_rot[:, 0], eq_rot[:, 1], eq_rot[:, 2])
    ra_s, dec_s = np.degrees(l0_rot), np.degrees(b0_rot)
    ind_s = (ra_s>17) & (ra_s<30)
    ra_s = ra_s[ind_s]
    dec_s = dec_s[ind_s]
    
    plt.close()
    plt.figure(figsize=(6,6))
    
    plt.plot(np.degrees(l0_rot), np.degrees(b0_rot), 'k-')
    plt.plot(ra_s, dec_s, 'ro')
    
    plt.xlim(50, -10)
    plt.ylim(-40, -10)

def atlas_track():
    """"""
    ra0, dec0 = np.radians(77.16), np.radians(46.92 - 90)

    # euler rotations
    D = np.array([[np.cos(ra0), np.sin(ra0), 0], [-np.sin(ra0), np.cos(ra0), 0], [0, 0, 1]])
    C = np.array([[np.cos(dec0), 0, np.sin(dec0)], [0, 1, 0], [-np.sin(dec0), 0, np.cos(dec0)]])
    B = np.diag(np.ones(3))

    R = np.dot(B, np.dot(C, D))
    Rinv = np.linalg.inv(R)
    
    l0 = np.linspace(0, 2*np.pi, 500)
    b0 = np.zeros(500)

    xeq, yeq, zeq = myutils.eq2car(l0, b0)
    eq = np.column_stack((xeq, yeq, zeq))

    eq_rot = np.zeros(np.shape(eq))
    for i in range(np.size(l0)):
        eq_rot[i] = np.dot(Rinv, eq[i])
    
    l0_rot, b0_rot = myutils.car2eq(eq_rot[:, 0], eq_rot[:, 1], eq_rot[:, 2])
    ra_s, dec_s = np.degrees(l0_rot), np.degrees(b0_rot)
    ind_s = (ra_s>17) & (ra_s<30)
    ra_s = ra_s[ind_s]
    dec_s = dec_s[ind_s]
    
    return (ra_s, dec_s)

# effects of different params on stream shape

def plot_potstream2(n, pparams=[430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1)], potential='gal', age=None):
    """Plot observed stream and and a model in a test potential"""

    obsmode = 'equatorial'
    footprint = 'sdss'
    
    # Load streams
    if n==-1:
        observed = load_gd1(present=[0,1,2,3])
        if age==None: age = 1.4*u.Gyr
        mi = 2e4*u.Msun
        mf = 2e-1*u.Msun
        x0, v0 = gd1_coordinates()
        xlims = [[190, 130], [0, 350]]
        ylims = [[15, 65], [5, 10], [-250, 150], [0, 250]]
        loc = 2
        name = 'GD-1'
        footprint = 'sdss'
    elif n==-3:
        observed = load_tri(present=[0,1,2,3])
        if age==None: age = 5*u.Gyr
        mi = 2e4*u.Msun
        mf = 2e-1*u.Msun
        x0, v0 = tri_coordinates()
        xlims = [[25, 19], [0, 350]]
        ylims = [[10, 50], [20, 45], [-175, -50], [0, 250]]
        loc = 1
        name = 'Triangulum'
        footprint = 'sdss'
    elif n==-4:
        observed = load_atlas(present=[0,1,2,3])
        if age==None: age = 2*u.Gyr
        mi = 2e4*u.Msun
        mf = 2e-1*u.Msun
        x0, v0 = atlas_coordinates()
        xlims = [[35, 10], [0, 350]]
        ylims = [[-40, -20], [15, 25], [50, 200], [0, 250]]
        loc = 3
        name = 'ATLAS'
        footprint = 'none'
    else:
        observed = load_pal5(present=[0,1,2,3])
        if age==None: age = 2.7*u.Gyr
        mi = 1e5*u.Msun
        mf = 2e4*u.Msun
        x0, v0 = pal5_coordinates2()
        xlims = [[245, 225], [0, 350]]
        ylims = [[-4, 10], [21, 27], [-80, -20], [0, 250]]
        loc = 3
        name = 'Pal 5'
        footprint = 'sdss'
    
    #########################
    # Plot observed streams
    modcol = 'r'
    fsize = 18
    obscol = '0.5'
    
    plt.close()
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    
    plt.sca(axes[0])
    plt.plot(observed.obs[0], observed.obs[1], 's', color=obscol, mec='none', ms=8)
    
    plt.xlim(xlims[0][0], xlims[0][1])
    plt.ylim(ylims[0][0], ylims[0][1])
    plt.xlabel("R.A. (deg)", fontsize=fsize)
    plt.ylabel("Dec (deg)", fontsize=fsize)
    
    plt.sca(axes[1])
    plt.plot(observed.obs[0], observed.obs[2], 's', color=obscol, mec='none', ms=8)
    
    plt.xlim(xlims[0][0], xlims[0][1])
    plt.ylim(ylims[1][0], ylims[1][1])
    plt.xlabel("R.A. (deg)", fontsize=fsize)
    plt.ylabel("Distance (kpc)", fontsize=fsize)
    
    plt.sca(axes[2])
    if np.shape(observed.obs)[0]>3:
        rvsample = observed.obs[3]>MASK
        plt.plot(observed.obs[0][rvsample], observed.obs[3][rvsample], 's', color=obscol, mec='none', ms=8, label='Observed')
    
    plt.xlim(xlims[0][0], xlims[0][1])
    plt.ylim(ylims[2][0], ylims[2][1])
    plt.xlabel("R.A. (deg)", fontsize=fsize)
    plt.ylabel("Radial velocity (km/s)", fontsize=fsize)
    
    ########################
    # Potential parameters
    vcpars = [x.value for x in pparams]
    
    modcols = [mpl.cm.bone(0.2), mpl.cm.bone(0.4), mpl.cm.bone(0.6), mpl.cm.bone(0.8)]
    # penarrubia+(2016) M_LMC = 2.5e11 Msun
    mass = [0, 1, 2.5, 5]
    oblateness = [0.8,0.9,1,1.1]
    pparams0 = pparams
    
    for i in range(4):
        modcol = modcols[i]
        if potential=='gal':
            pf = [3.4e10, 0.7, 1e11, 6.5, 0.26]
            uf = [u.Msun, u.kpc, u.Msun, u.kpc, u.kpc]
            pfixed = [x*y for x,y in zip(pf, uf)]
            pparams = pfixed + pparams0
            pparams[-1] = oblateness[i]*u.Unit(1)
            label = '$q_z$ = {:.1f}'.format(oblateness[i])
        elif potential=='lmc':
            mlmc, xlmc = lmc_properties()
            pf = [3.4e10, 0.7, 1e11, 6.5, 0.26]
            uf = [u.Msun, u.kpc, u.Msun, u.kpc, u.kpc]
            pfixed = [x*y for x,y in zip(pf, uf)]
            pparams = pfixed + pparams0 + [mass[i]*1e11*u.Msun] + [x for x in xlmc]
            label = 'M$_{LMC}$ = ' + '{:.1f}'.format(mass[i]) + '$\cdot 10^{11} M_\odot$'
        
        distance = 8.3*u.kpc
        mr = pparams[5]**2 * pparams[6] / G * (np.log(1 + distance/pparams[6]) - distance/(distance + pparams[6]))
        vc_ = np.sqrt(G*mr/distance)
        vsun['vcirc'] = np.sqrt((198*u.km/u.s)**2 + vc_**2)
        
        params = {'generate': {'x0': x0*u.kpc, 'v0': v0*u.km/u.s, 'potential': potential, 'pparams': pparams, 'minit': mi, 'mfinal': mf, 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': 1*u.Myr, 'age': age, 'nstars': 400, 'integrator': 'lf'}, 'observe': {'mode': obsmode, 'nstars':-1, 'sequential':True, 'errors': [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,3,4,5], 'observer': mw_observer, 'footprint': footprint}}
        
        stream = Stream(**params['generate'])
        stream.generate()
        stream.observe(**params['observe'])
        
        np.save('../data/stream_{0:d}_{1:s}_{2:d}'.format(n, potential, i), stream.obs)
        
        # Plot modeled streams
        plt.sca(axes[0])
        plt.plot(stream.obs[0], stream.obs[1], 'o', color=modcol, mec='none', ms=4)
        
        plt.sca(axes[1])
        plt.plot(stream.obs[0], stream.obs[2], 'o', color=modcol, mec='none', ms=4)
        
        plt.sca(axes[2])
        plt.plot(stream.obs[0], stream.obs[3], 'o', color=modcol, mec='none', ms=4, label=label)

    plt.legend(fontsize='xx-small', loc=loc, handlelength=0.2, frameon=False)
    plt.suptitle('{} stream'.format(name), fontsize='medium')
    
    plt.tight_layout(h_pad=0.02, w_pad=0.02)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig('../plots/tevo_{0:s}_{1:d}.png'.format(potential, n))
