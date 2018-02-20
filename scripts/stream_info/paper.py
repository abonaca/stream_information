from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .stream_info import *

def all_mocks():
    """"""
    done = get_done()
    N = len(done)
    
    ncol = 3
    nrow = np.int64(np.ceil(N/ncol))
    Np = ncol * nrow
    da = 3
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*da + 0.2*da, nrow*da))
    
    for i in range(N):
        plt.sca(ax[int(i/ncol)][i%ncol])
        
        observed = load_stream(done[i])
        model = stream_model(name=done[i])
        #dp_ = np.load('../data/streams/{}_poly.npz'.format(name))
        #p_ = dp_['p']
        #var = dp_['var']
        #poly = np.poly1d(p_)

        #if var=='ra':
            #x = np.linspace(np.min(ts['ra']), np.max(ts['ra']), 1000)
        #else:
            #x = np.linspace(np.min(ts['dec']), np.max(ts['dec']), 1000)
        #y = np.polyval(poly, x)
        
        #p = np.loadtxt('../data/streams/{}_poly.txt'.format(done[i]))
        #poly = np.poly1d(p)
        #x = np.linspace(np.min(observed.obs[0]), np.max(observed.obs[0]), 100)
        #y = poly(x)
        
        plt.plot(observed.obs[0], observed.obs[1], 'ko', ms=7, label='Observed')
        plt.plot(model.obs[0], model.obs[1], 'o', color='0.7', ms=5, label='Mock')
        #plt.plot(x, y, '-', color='cornflowerblue', lw=3, label='Track')
        
        xlims = list(plt.gca().get_xlim())
        ylims = list(plt.gca().get_ylim())
        dx = xlims[1] - xlims[0]
        dy = ylims[1] - ylims[0]
        
        delta = np.abs(dx - dy)
        if dx>dy:
            ylims[0] = ylims[0] - 0.5*delta
            ylims[1] = ylims[1] + 0.5*delta
        else:
            xlims[0] = xlims[0] - 0.5*delta
            xlims[1] = xlims[1] + 0.5*delta
        
        plt.xlim(xlims[1], xlims[0])
        plt.ylim(ylims[0], ylims[1])
        
        print(i, dx, dy)
        fancy_name = full_name(done[i])
        txt = plt.text(0.9, 0.9, fancy_name, fontsize='small', transform=plt.gca().transAxes, ha='right', va='top')
        txt.set_bbox(dict(facecolor='w', alpha=0.5, ec='none'))
        
        if i==0:
            plt.legend(frameon=False, fontsize='small', loc=3, handlelength=0.2)
        
        if int(i/ncol)==nrow-1:
            plt.xlabel('R.A. (deg)')
        if i%ncol==0:
            plt.ylabel('Dec (deg)')
    
    for i in range(N, Np):
        plt.sca(ax[int(i/ncol)][i%ncol])
        plt.gca().axis('off')
    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../paper/mocks.pdf')

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
    
    mpl.rcParams.update({'font.size': 22})
    plt.close()
    #fig, ax = plt.subplots(nrow, ncol, figsize=(da*ncol, da*3.9), squeeze=False, gridspec_kw = {'height_ratios':[1.2, 3, 1.2, 3, 1.2, 3]})
    
    fig = plt.figure(figsize=(da*ncol, da*3.9))
    outer_grid = mpl.gridspec.GridSpec(3, 1, wspace=0.0, hspace=0.35)
    
    for e, vary in enumerate(['progenitor', 'bary', 'halo']):
        inner_grid = mpl.gridspec.GridSpecFromSubplotSpec(2, ncol, subplot_spec=outer_grid[e], wspace=0.35, hspace=0.05, height_ratios=[1.2,3])
        
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
            
            # derivative deviation
            ax = plt.Subplot(fig, inner_grid[ncol+p])
            ax1 = fig.add_subplot(ax)
            
            plt.plot(step[p], dev[p], 'ko', ms=8)
            
            plt.axvline(opt_step, ls='-', color='crimson', lw=3)
            plt.plot(step[p][opt_id], dev[p][opt_id], 'o', ms=8, color='crimson')
            
            plt.axhline(dtol, ls='-', color='salmon', lw=2)
            y0, y1 = plt.gca().get_ylim()
            plt.axhspan(y0, dtol, color='salmon', alpha=0.3, zorder=0)
            
            plt.gca().set_yscale('log')
            plt.gca().set_xscale('log')
            plt.xlabel('$\Delta$ {} {}'.format(plabels[p], punits[p]), fontsize=28)
            plt.gca().xaxis.labelpad = 0.2
            if p==0:
                plt.ylabel('$\Delta$ $\dot{y}$', fontsize=28)
            
            # derivative
            ax = plt.Subplot(fig, inner_grid[p])
            ax0 = fig.add_subplot(ax, sharex=ax1)
            
            for i in range(5):
                for j in range(3):
                    plt.plot(steps_all[p], np.tanh(dydx[p,:,i,np.int64(j*Nra/3)]), '-', color='{}'.format(i/5), lw=1, alpha=1)

            plt.axvline(opt_step, ls='-', color='crimson', lw=3)
            ax0.set_xlim(ax1.get_xlim())
            plt.gca().set_xscale('log')
            plt.gca().tick_params(labelbottom='off')
            
            plt.ylim(-1,1)
            if p==0:
                plt.ylabel('$\dot{y}$', fontsize=28)
            
    plt.text(0.67, 0.3, 'for data dimensions $y_i$ and parameter x:', transform=fig.transFigure, ha='left', va='center', fontsize=24)
    plt.text(0.69, 0.25, '$\dot{y}$ = tanh (d$y_i$ / dx)', transform=fig.transFigure, ha='left', va='center', fontsize=24)
    plt.text(0.89, 0.17, '$\Delta$ $\dot{y}$ = $\sum_i [(dy_i/dx)|_{\Delta x_j} - (dy_i/dx)|_{\Delta x_{j-1}}]^2$\n$+ \sum_i  [(dy_i/dx)|_{\Delta x_j} - (dy_i/dx)|_{\Delta x_{j+1}}]^2$', transform=fig.transFigure, ha='right', va='center', fontsize=24)
    
    #plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../paper/derivative_steps.pdf')
    
    mpl.rcParams.update({'font.size': 18})


# results
def crb_2d(name='atlas', vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True):
    """Corner plot with covariances between the CRBs of all the model parameters"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    plabels, units = get_parlabel(pid)
    punits = [' ({})'.format(x) if len(x) else '' for x in units]
    params = ['$\Delta$ {}{}'.format(x, y) for x,y in zip(plabels, punits)]
    Nvar = len(pid)
    
    fig = None
    ax = None
    
    frac = [0.8, 0.5, 0.2]
    
    for e, Ndim in enumerate([3,4,6]):
        d = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        cxi = d['cxi']
        cx = stable_inverse(cxi)
        
        color = mpl.cm.bone(frac[e])
        fig, ax = corner_ellipses(cx, fig=fig, ax=ax, dax=2, color=color, alpha=0.7, lw=3)
        
        # labels
        for k in range(Nvar-1):
            plt.sca(ax[-1][k])
            plt.xlabel(params[k])
            
            plt.sca(ax[k][0])
            plt.ylabel(params[k+1])
        
    plt.tight_layout()
    plt.savefig('../paper/crb_correlations.pdf')

def sky(Ndim=6, vary=['progenitor', 'bary', 'halo'], component='halo', errmode='fiducial', align=True):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    names = get_done()
    
    tout = Table(names=('name', 'crb'))
    
    pparams0 = pparams_fid
    pid_comp, dp_fid2, vlabel2 = get_varied_pars(component)
    Np = len(pid_comp)
    pid_crb = myutils.wherein(np.array(pid), np.array(pid_comp))
    
    pparams_comp = [pparams0[x] for x in pid_comp]
    pparams_arr = np.array([x.value for x in pparams_comp])

    coll = np.load('../data/crb/cx_collate_multi1_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(errmode, Ndim, align, vlabel, component))
    p_all = coll['p']
    p_all = p_all / pparams_arr
    
    plt.close()
    fig, ax = plt.subplots(Np,1,figsize=(10,15), subplot_kw=dict(projection='mollweide'))
    
    for e, name in enumerate(names):
        crb_frac = p_all[e]
        #print(name, crb_frac)
        
        stream = np.load('../data/streams/mock_observed_{}.npy'.format(name))
        
        for i in range(Np):
            plt.sca(ax[i])
            color_index = np.array(crb_frac[:])
            color_index[color_index>0.2] = 0.2
            color_index /= 0.2
            color = mpl.cm.viridis(color_index[i])
            
            isort = np.argsort(stream[0])
            plt.plot(np.radians(stream[0][isort]), np.radians(stream[1][isort]), '-', color=color, lw=4) #, rasterized=True)
    
    for i in range(Np):
        plt.sca(ax[i])
        #plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.text(0.9, 0.9, '$\Delta$ {}'.format(get_parlabel(pid_comp[i])[0]), fontsize='medium', transform=plt.gca().transAxes, va='bottom', ha='left')
        plt.grid()
        
    plt.xlabel('RA')
    
    # add custom colorbar
    sm = plt.cm.ScalarMappable(cmap=mpl.cm.viridis, norm=plt.Normalize(vmin=0, vmax=20))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    
    if component=='bary':
        cb_pad = 0.1
    else:
        cb_pad = 0.06
    cb = fig.colorbar(sm, ax=ax.ravel().tolist(), pad=cb_pad, aspect=40, ticks=np.arange(0,21,5))
    cb.set_label('Cramer $-$ Rao bounds (%)')
    
    #plt.tight_layout()
    plt.savefig('../paper/crb_onsky_{}.pdf'.format(component)) #, dpi=200)

def sky_all(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True):
    """On-sky streams colorcoded by CRLB, for all potential parameters in separate panels"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    names = get_done()
    
    tout = Table(names=('name', 'crb'))
    
    pparams0 = pparams_fid
    
    for e, component in enumerate(['bary', 'halo']):
        pid_comp, dp_fid2, vlabel2 = get_varied_pars(component)
        Np = len(pid_comp)
        pid_crb = myutils.wherein(np.array(pid), np.array(pid_comp))
        
        pparams_comp = [pparams0[x] for x in pid_comp]
        pparams_arr = np.array([x.value for x in pparams_comp])

        coll = np.load('../data/crb/cx_collate_multi1_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(errmode, Ndim, align, vlabel, component))
        p_comp = coll['p']
        p_comp = p_comp / pparams_arr
        
        #print(np.shape(p_comp))
        if e==0:
            p_all = np.copy(p_comp)
            pid_all = pid_comp[:]
        else:
            p_all = np.hstack([p_all, p_comp])
            pid_all = pid_all + pid_comp
        
        #print(p_all)
        #print(np.shape(p_all))
    
    Ns, Np = np.shape(p_all)
    
    plt.close()
    fig, ax = plt.subplots(3,3,figsize=(15,7), subplot_kw=dict(projection='mollweide'))
    
    for e, name in enumerate(names):
        crb_frac = p_all[e]
        #print(name, crb_frac)
        
        stream = np.load('../data/streams/mock_observed_{}.npy'.format(name))
        
        for i in range(Np):
            plt.sca(ax[np.int64(i/3)][i%3])
            color_index = np.array(crb_frac[:])
            color_index[color_index>0.2] = 0.2
            color_index /= 0.2
            color = mpl.cm.viridis(color_index[i])
            
            isort = np.argsort(stream[0])
            plt.plot(np.radians(stream[0][isort]), np.radians(stream[1][isort]), '-', color=color, lw=4) #, rasterized=True)
    
    for i in range(Np):
        plt.sca(ax[np.int64(i/3)][i%3])
        #plt.xlabel('RA')
        #plt.ylabel('Dec')
        plt.text(0.9, 0.9, '$\Delta$ {}'.format(get_parlabel(pid_all[i])[0]), fontsize='medium', transform=plt.gca().transAxes, va='bottom', ha='left')
        plt.grid()
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.setp(plt.gca().get_yticklabels(), visible=False)
        
    #plt.xlabel('RA')
    
    # add custom colorbar
    sm = plt.cm.ScalarMappable(cmap=mpl.cm.viridis, norm=plt.Normalize(vmin=0, vmax=20))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    
    if component=='bary':
        cb_pad = 0.1
    else:
        cb_pad = 0.06
    cb = fig.colorbar(sm, ax=ax.ravel().tolist(), pad=cb_pad, aspect=40, ticks=np.arange(0,21,5))
    cb.set_label('Cramer $-$ Rao bounds (%)')
    
    plt.savefig('../paper/crb_onsky.pdf') #, dpi=200)

def nstream_improvement(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', component='halo', align=True, relative=False):
    """Show how much parameters improve by including additional streams"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    done = get_done()
    N = len(done)
    
    # choose the appropriate components:
    Nprog, Nbary, Nhalo, Ndipole, Nquad, Npoint = [6, 5, 4, 3, 5, 1]
    if 'progenitor' not in vary:
        Nprog = 0
    nstart = {'bary': Nprog, 'halo': Nprog + Nbary, 'dipole': Nprog + Nbary + Nhalo, 'quad': Nprog + Nbary + Nhalo + Ndipole, 'all': Nprog, 'point': 0}
    nend = {'bary': Nprog + Nbary, 'halo': Nprog + Nbary + Nhalo, 'dipole': Nprog + Nbary + Nhalo + Ndipole, 'quad': Nprog + Nbary + Nhalo + Ndipole + Nquad} #, 'all': np.shape(cx)[0], 'point': 1}
    
    if 'progenitor' not in vary:
        nstart['dipole'] = Npoint
        nend['dipole'] = Npoint + Ndipole
    
    if component in ['bary', 'halo', 'dipole', 'quad', 'point']:
        components = [component]
    else:
        components = [x for x in vary if x!='progenitor']
    
    pid_comp = pid[nstart[component]:nend[component]]
    plabels, units = get_parlabel(pid_comp)
    if relative:
        punits = [' (%)' for x in units]
    else:
        punits = [' ({})'.format(x) if len(x) else '' for x in units]
    params = ['$\Delta$ {}{}'.format(x, y) for x,y in zip(plabels, punits)]
    Nvar = len(pid_comp)
    
    pparams0 = pparams_fid
    pparams_comp = [pparams0[x] for x in pid_comp]
    pparams_arr = np.array([x.value for x in pparams_comp])

    median = np.empty((Nvar, N))
    x = np.arange(N) + 1
    
    da = 3
    ncol = 2
    nrow = np.int64(Nvar/ncol)
    w = 4 * da
    h = nrow * da

    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(w,h), sharex='col')
    
    for i in range(N):
        Nmulti = i+1
        t = np.arange(N, dtype=np.int64).tolist()
        all_comb = list(itertools.combinations(t, Nmulti))
        comb = sorted(list(set(all_comb)))
        Ncomb = len(comb)
        
        coll = np.load('../data/crb/cx_collate_multi{:d}_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(Nmulti, errmode, Ndim, align, vlabel, component))
        comb_all = coll['comb']
        cq_all = coll['cx']
        p_all = coll['p']
        if relative:
            p_all = p_all * 100 / pparams_arr
        
        median = np.median(p_all, axis=0)
        Ncomb = np.shape(comb_all)[0]
        
        nst = np.ones(Ncomb) * Nmulti
            
        for k in range(Nvar):
            plt.sca(ax[k%ncol][np.int64(k/ncol)])
            if (i==0) & (k==0):
                plt.plot(nst, p_all[:,k], 'o', color='0.8', ms=10, label='Single combination of N streams')
                plt.plot(Nmulti, median[k], 'wo', mec='k', mew=2, ms=10, label='Median over different\ncombinations of N streams')
            else:
                plt.plot(nst, p_all[:,k], 'o', color='0.8', ms=10)
                plt.plot(Nmulti, median[k], 'wo', mec='k', mew=2, ms=10)
            
            if Nmulti<=3:
                if Nmulti==1:
                    Nmin = 3
                else:
                    Nmin = 1
                ids_min = p_all[:,k].argsort()[:Nmin]
                
                for j_ in range(Nmin):
                    best_names = [done[np.int64(i_)] for i_ in comb[ids_min[j_]][:Nmulti]]
                    print(k, j_, best_names)
                    label = ', '.join(best_names)
                    
                    plt.text(Nmulti, p_all[ids_min[j_],k], '{}'.format(label), fontsize='xx-small')

    for k in range(Nvar):
        plt.sca(ax[k%ncol][np.int64(k/ncol)])
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        if relative:
            plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        
        plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        #plt.gca().xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        plt.ylabel(params[k])
        
        if k==0:
            plt.legend(frameon=False, fontsize='small', loc=1)
    
        if k%ncol==nrow-1:
            plt.xlabel('Number of streams in a combination')
    
    plt.tight_layout()
    plt.savefig('../paper/nstream_improvement.pdf')


# applications


# interpretation
def orbit_corr(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True):
    """Show how CRBs on different potential parameters depend on orbital properties of the progenitor"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    names = get_done()
    
    # get fiducial halo parameters
    c_halo = np.load('../data/crb/cx_collate_multi1_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(errmode, Ndim, align, vlabel, 'halo'))
    p_halo = c_halo['p_rel']
    
    # get fiducial bary parameters
    c_bary = np.load('../data/crb/cx_collate_multi1_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(errmode, Ndim, align, vlabel, 'bary'))
    p_bary = c_bary['p_rel']

    p = np.hstack([p_bary[:,2][:,np.newaxis], p_halo, p_halo[:,0][:,np.newaxis]/p_halo[:,1][:,np.newaxis]])
    t = Table.read('../data/crb/ar_orbital_summary.fits')
    
    nrow = 6
    ncol = 5
    da = 2.5
    cname = ['length', 'rcur', 'rapo', 'lx', 'lz']
    
    xlabels = ['Length (deg)', 'R$_{cur}$ (kpc)', 'R$_{apo}$ (kpc)', '|L$_x$|/|L|', '|L$_z$|/|L|']
    ylabels = ['log $M_d$', '$V_h$', '$R_h$', '$q_x$', '$q_z$', '$V_h$ / $R_h$']
    dylabels = ['$\Delta$ {}'.format(s) for s in ylabels]
    
    mask = t['name']!='ssangarius'
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*da, nrow*da), sharex='col', sharey='row')
    
    for i in range(nrow):
        for j in range(ncol):
            plt.sca(ax[i][j])
            plt.plot(t[cname[j]][mask], p[:,i][mask], 'o', ms=5, color='0.2')
            
            corr = scipy.stats.pearsonr(t[cname[j]][mask], p[:,i][mask])
            #print(corr)
            txt = plt.text(0.1, 0.1, '{:.3g}'.format(corr[0]), fontsize='x-small', transform=plt.gca().transAxes)
            txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))
            
            if j==0:
                plt.ylabel(dylabels[i])
            if i==nrow-1:
                plt.xlabel(xlabels[j])
            
            
        
        plt.gca().set_yscale('log')
        if i==2:
            plt.ylim(1e-1,1e1)
        else:
            plt.ylim(1e-2,1e0)
        #plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(2))
        
    
    plt.tight_layout()
    plt.savefig('../paper/orbit_correlations.pdf')

def vc():
    """"""
    t = Table.read('../data/crb/vc_orbital_summary.fits')
    N = len(t)
    fapo = t['rapo']/np.max(t['rapo'])
    fapo = t['rapo']/100
    flen = t['length']/np.max(t['length']) + 0.1
    
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    
    for i in range(N):
        color = mpl.cm.bone(fapo[i])
        lw = flen[i] * 5
        
        plt.sca(ax[0])
        plt.plot(t['r'][i], t['vc'][i], '-', color=color, lw=lw)
        
    plt.xlabel('R (kpc)')
    plt.ylabel('$\Delta$ $V_c$ / $V_c$')
    plt.ylim(0, 2.5)
    
    plt.sca(ax[1])
    plt.scatter(t['length'], t['vcmin'], c=fapo, cmap='bone', vmin=0, vmax=1)
    
    plt.xlabel('Length (deg)')
    plt.ylabel('min $\Delta$ $V_c$')
    plt.ylim(0, 2.5)
    
    plt.sca(ax[2])
    a = np.linspace(0,90,100)
    plt.plot(a, a, 'k-')
    plt.scatter(t['rcur'], t['rmin'], c=fapo, cmap='bone', vmin=0, vmax=1)
    
    plt.xlabel('$R_{apo}$ (kpc)')
    plt.ylabel('$R_{min}$ (kpc)')
    
    
    plt.tight_layout()
    plt.savefig('../paper/vc_crb.pdf')

def ar(current=False):
    """Explore constraints on radial acceleration, along the progenitor line"""
    t = Table.read('../data/crb/ar_orbital_summary.fits')
    N = len(t)
    fapo = t['rapo']/np.max(t['rapo'])
    fapo = t['rapo']/100
    flen = t['length']/(np.max(t['length']) + 10)
    fcolor = fapo
    
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    
    for i in range(N):
        color = mpl.cm.bone(fcolor[i])
        lw = flen[i] * 5
        
        plt.sca(ax[0])
        plt.plot(t['r'][i], t['ar'][i], '-', color=color, lw=lw)
        
    plt.xlabel('R (kpc)')
    plt.ylabel('$\Delta$ $a_r$ / $a_r$')
    plt.ylim(0, 2.5)
    
    plt.sca(ax[1])
    plt.scatter(t['length'], t['armin'], c=fcolor, cmap='bone', vmin=0, vmax=1)
    
    plt.xlabel('Length (deg)')
    plt.ylabel('min $\Delta$ $a_r$')
    #plt.ylim(0, 1)
    
    plt.sca(ax[2])
    a = np.linspace(0,90,100)
    plt.plot(a, a, 'k-')
    plt.plot(a, 2*a, 'k--')
    plt.plot(a, 3*a, 'k:')
    if current:
        plt.scatter(t['rcur'], t['rmin'], c=fcolor, cmap='bone', vmin=0, vmax=1)
        plt.xlabel('$R_{cur}$ (kpc)')
    else:
        plt.scatter(t['rapo'], t['rmin'], c=fcolor, cmap='bone', vmin=0, vmax=1)
        plt.xlabel('$R_{apo}$ (kpc)')
    plt.ylabel('$R_{min}$ (kpc)')
    
    plt.xlim(0,90)
    plt.ylim(0,90)
    
    
    plt.tight_layout()
    plt.savefig('../plots/ar_crb_current{:d}.pdf'.format(current))
    if not current:
        plt.savefig('../paper/ar_crb.pdf')


# tables
def table_obsmodes(verbose=True):
    """Save part of the latex table with information on observing modes"""
    
    obsmodes = pickle.load(open('../data/observing_modes.info', 'rb'))
    modes = ['fiducial', 'desi', 'gaia', 'exgal']
    names = obsmode_name(modes)
    
    fout = open('../paper/obsmodes.tex', 'w')
    
    for e, mode in enumerate(modes):
        sigmas = obsmodes[mode]['sig_obs'].tolist()
        line = '{} & {:.1f} & {:.1f} & {:.0f} & {:.1f} & {:.1f} \\\\'.format(names[e], *sigmas)
        line = line.replace('nan', 'N/A')
        
        if verbose: print(line)
        
        fout.write('{}\n'.format(line))
    
    fout.close()
    
