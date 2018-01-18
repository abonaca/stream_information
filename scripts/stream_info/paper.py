from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .stream_info import *

def derivative_vis(name='atlas'):
    """Plot steps in calculating a stream derivative wrt a potential parameter"""
    
    # setup
    mock = pickle.load(open('../data/mock_{}.params'.format(name),'rb'))
    rotmatrix = mock['rotmatrix']
    xmm = mock['xi_range']
    Nobs = 50
    dV = 40*u.km/u.s
    
    # fiducial model
    pparams0 = [x for x in pparams_fid]
    fiducial = stream_model(name=name, pparams0=pparams0, rotmatrix=rotmatrix)
    
    # fiducial bspline
    k = 3
    isort = np.argsort(fiducial.obs[0])
    ra0 = np.linspace(np.percentile(fiducial.obs[0], 10), np.percentile(fiducial.obs[0], 90), Nobs)
    t = np.r_[(fiducial.obs[0][isort][0],)*(k+1), ra0, (fiducial.obs[0][isort][-1],)*(k+1)]
    bs_fiducial = scipy.interpolate.make_lsq_spline(fiducial.obs[0][isort], fiducial.obs[1][isort], t, k=k)
    
    # excursion models
    # less massive halo
    pparams1 = [x for x in pparams_fid]
    pparams1[5] = pparams1[5] - dV # this way to only change pparams1, and not pparams_fid
    stream1 = stream_model(name=name, pparams0=pparams1, rotmatrix=rotmatrix)
    
    isort1 = np.argsort(stream1.obs[0])
    ra1 = np.linspace(np.percentile(stream1.obs[0], 10), np.percentile(stream1.obs[0], 90), Nobs)
    t1 = np.r_[(stream1.obs[0][isort1][0],)*(k+1), ra1, (stream1.obs[0][isort1][-1],)*(k+1)]
    bs_stream1 = scipy.interpolate.make_lsq_spline(stream1.obs[0][isort1], stream1.obs[1][isort1], t1, k=k)
    
    # more massive halo
    pparams2 = [x for x in pparams_fid]
    pparams2[5] = pparams2[5] + dV
    stream2 = stream_model(name=name, pparams0=pparams2, rotmatrix=rotmatrix)
    
    isort2 = np.argsort(stream2.obs[0])
    ra2 = np.linspace(np.percentile(stream2.obs[0], 10), np.percentile(stream2.obs[0], 90), Nobs)
    t2 = np.r_[(stream2.obs[0][isort2][0],)*(k+1), ra2, (stream2.obs[0][isort2][-1],)*(k+1)]
    bs_stream2 = scipy.interpolate.make_lsq_spline(stream2.obs[0][isort2], stream2.obs[1][isort2], t2, k=k)
    
    # place observations in the range covered by the shortest model
    ra = np.linspace(np.min(stream2.obs[0]), np.max(stream2.obs[0]), Nobs)
    
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(9,7), sharex=True)
    lw = 2
    
    plt.sca(ax[0])
    plt.plot(fiducial.obs[0], fiducial.obs[1], 'o', color='0.8', ms=10, label='Fiducial stream model')
    plt.plot(fiducial.obs[0][isort], bs_fiducial(fiducial.obs[0][isort]), 'k-', lw=lw, label='B-spline fit')

    plt.plot(stream1.obs[0], stream1.obs[1], 'o', color='salmon', ms=10, zorder=0, label='$V_h$ = $V_{h, fid}$' + ' - {}'.format(dV.value) + ' km s$^{-1}$')
    plt.plot(stream1.obs[0][isort1], bs_stream1(stream1.obs[0][isort1]), '-', color='darkred', lw=lw, label='')

    plt.plot(stream2.obs[0], stream2.obs[1], 'o', color='skyblue', ms=10, zorder=0, label='$V_h$ = $V_{h, fid}$' + ' + {}'.format(dV.value) + ' km s$^{-1}$')
    plt.plot(stream2.obs[0][isort2], bs_stream2(stream2.obs[0][isort2]), '-', color='midnightblue', lw=lw, label='')
    
    ra_arr = 51.4
    c_arr = '0.2'
    plt.annotate('', xy=(ra_arr, bs_stream1(ra_arr)), xytext=(ra_arr, bs_stream2(ra_arr)), color=c_arr, fontsize=14, arrowprops=dict(color=c_arr, ec=c_arr, arrowstyle='<->', mutation_scale=15, lw=2))
    plt.text(50.5, -0.15, '$\Delta$ $\eta$', color=c_arr, fontsize='medium', ha='right', va='center')
    
    plt.ylabel('$\eta$ (deg)')
    plt.legend(fontsize='small', ncol=2, loc=4)

    plt.sca(ax[1])
    
    plt.plot(ra, (bs_stream2(ra) - bs_stream1(ra))/(2*dV), 'k-', lw=1.5*lw)
    
    plt.xlabel('$\\xi$ (deg)')
    plt.ylabel('$\Delta$ $\eta$ / $\Delta$ $V_h$ (deg km$^{-1}$ s)')
    
    plt.tight_layout()
    plt.savefig('../paper/derivative_vis.pdf')

def derivative_stepsize(name='atlas', tolerance=2, Nstep=10, log=True, layer=1):
    """Plot change in numerical derivative Delta y / Delta x as a function of step size Delta x"""
    
    # plot setup
    da = 4
    nrow = 6
    ncol = 6
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(da*ncol, da*3.9), squeeze=False, gridspec_kw = {'height_ratios':[1.2, 3, 1.2, 3, 1.2, 3]})
    
    for e, vary in enumerate(['progenitor', 'bary', 'halo']):
        pid, dp, vlabel = get_varied_pars(vary)
        Np = len(pid)
        plabels, units = get_parlabel(pid)
        punits = ['({})'.format(x) if len(x) else '' for x in units]
        
        t = np.load('../data/step_convergence_{}_{}_Ns{}_log{}_l{}.npz'.format(name, vlabel, Nstep, log, layer))
        dev = t['dev']
        step = t['step']
        dydx = t['ders']
        steps_all = t['steps_all'][:,::-1]
        Nra = np.shape(dydx)[-1]
        
        best = np.empty(Np)
    
        for p in range(Np):
            # choose step
            dmin = np.min(dev[p])
            dtol = tolerance * dmin
            opt_step = np.min(step[p][dev[p]<dtol])
            opt_id = step[p]==opt_step
            best[p] = opt_step
            
            ## largest step w deviation smaller than 1e-4
            #opt_step = np.max(step[p][dev[p]<1e-4])
            #opt_id = step[p]==opt_step
            #best[p] = opt_step
            
            plt.sca(ax[e*2][p])
            for i in range(5):
                for j in range(10):
                    plt.plot(steps_all[p], np.tanh(dydx[p,:,i,np.int64(j*Nra/10)]), '-', color='{}'.format(i/10), lw=0.5, alpha=0.5)

            plt.axvline(opt_step, ls='-', color='crimson', lw=3)
            plt.ylim(-1,1)
            plt.gca().set_xscale('log')
            
            if p==0:
                plt.ylabel('Derivative')
            #plt.title('{}'.format(plabels[p])+'$_{best}$ = '+'{:2.2g}'.format(opt_step), fontsize='small')
            
            plt.sca(ax[e*2+1][p])
            plt.plot(step[p], dev[p], 'ko', ms=8)
            
            plt.axvline(opt_step, ls='-', color='crimson', lw=3)
            plt.plot(step[p][opt_id], dev[p][opt_id], 'o', ms=8, color='crimson')
            
            plt.axhline(dtol, ls='-', color='salmon', lw=2)
            y0, y1 = plt.gca().get_ylim()
            plt.axhspan(y0, dtol, color='salmon', alpha=0.3, zorder=0)
            
            plt.gca().set_yscale('log')
            plt.gca().set_xscale('log')
            plt.xlabel('$\Delta$ {} {}'.format(plabels[p], punits[p]))
            if p==0:
                plt.ylabel('Derivative deviation')
            
            # share x axis in rows of 2
            ax[e*2][p].set_xlim(ax[e*2+1][p].get_xlim())
            ax[e*2][p].tick_params(labelbottom='off')
            
        for poff in range(p+1,ncol):
            plt.sca(ax[e*2][poff])
            plt.axis('off')
            plt.sca(ax[e*2+1][poff])
            plt.axis('off')
        
    plt.tight_layout(h_pad=0)
    plt.savefig('../paper/derivative_steps.pdf')


