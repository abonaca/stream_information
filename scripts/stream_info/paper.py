from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import scipy.interpolate

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
    
    colplus = 'skyblue'
    colminus = 'salmon'
    colfid = '0.8'
    
    #colplus, colfid, colminus = [mpl.cm.Blues(x) for x in [0.2, 0.5, 0.7]]
    #colplus, colfid, colminus = [mpl.cm.RdBu(x) for x in [0.65, 0.75, 0.85]]
    
    plt.sca(ax[0])
    plt.plot(fiducial.obs[0], fiducial.obs[1], 'o', color=colfid, ms=10, label='Fiducial stream model')
    plt.plot(fiducial.obs[0][isort], bs_fiducial(fiducial.obs[0][isort]), 'k-', lw=lw, label='B-spline fit')

    plt.plot(stream1.obs[0], stream1.obs[1], 'o', color=colminus, ms=10, zorder=0, label='$V_h$ = $V_{h, fid}$' + ' - {}'.format(dV.value) + ' km s$^{-1}$')
    plt.plot(stream1.obs[0][isort1], bs_stream1(stream1.obs[0][isort1]), '-', color='darkred', lw=lw, label='')

    plt.plot(stream2.obs[0], stream2.obs[1], 'o', color=colplus, ms=10, zorder=0, label='$V_h$ = $V_{h, fid}$' + ' + {}'.format(dV.value) + ' km s$^{-1}$')
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
                for j in range(1,2):
                    #plt.plot(steps_all[p], np.tanh(dydx[p,:,i,np.int64(j*Nra/3)]), '-', color='{}'.format(i/5), lw=1, alpha=1)
                    plt.plot(steps_all[p], np.abs(dydx[p,:,i,np.int64(j*Nra/3)]), '-', color='0.3', lw=1, alpha=1)

            plt.axvline(opt_step, ls='-', color='crimson', lw=3)
            ax0.set_xlim(ax1.get_xlim())
            plt.gca().set_xscale('log')
            plt.gca().set_yscale('log')
            plt.gca().tick_params(labelbottom='off')
            
            #plt.ylim(-1,1)
            if p==0:
                plt.ylabel('$\dot{y}$', fontsize=28)
            
    plt.text(0.67, 0.3, 'for data dimensions $y_i$ and parameter x:', transform=fig.transFigure, ha='left', va='center', fontsize=24)
    #plt.text(0.69, 0.25, '$\dot{y}$ = tanh (d$y_i$ / dx)', transform=fig.transFigure, ha='left', va='center', fontsize=24)
    plt.text(0.69, 0.25, '$\dot{y}$ = |d$y_i$ / dx|', transform=fig.transFigure, ha='left', va='center', fontsize=24)
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
    dim_labels = ['3D fiducial data:\non-sky positions & distances', '4D fiducial data:\npositions & radial velocities', '6D fiducial data:\nfull phase space']
    
    for e, Ndim in enumerate([3,4,6]):
        d = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        cxi = d['cxi']
        cx = stable_inverse(cxi)
        
        color = mpl.cm.bone(frac[e])
        if Ndim==6:
            fig, ax, pcc = corner_ellipses(cx, fig=fig, ax=ax, dax=2, color=color, alpha=0.7, lw=3, correlate=True)
        else:
            fig, ax = corner_ellipses(cx, fig=fig, ax=ax, dax=2, color=color, alpha=0.7, lw=3)
        
        plt.sca(ax[0][-1])
        plt.plot([0,1], [1,1], '-', lw=3, color=color, alpha=0.7, label=dim_labels[e])
        
    # labels
    for k in range(Nvar-1):
        plt.sca(ax[-1][k])
        plt.xlabel(params[k])
        
        plt.sca(ax[k][0])
        plt.ylabel(params[k+1])
    
    
    plt.sca(ax[0][-1])
    plt.legend(fontsize='xx-large', loc=1, bbox_to_anchor=(0., 0.))
    plt.ylim(0,0.1)
    
    # correlations
    k = 0
    Npair = np.int64(Nvar*(Nvar-1)/2)
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar):
            plt.sca(ax[j-1][i])
            txt = plt.text(0.9, 0.9, '{:.2f}'.format(pcc[2,k]), fontsize='x-small', transform=plt.gca().transAxes, va='top', ha='right')
            txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))
            
            k += 1
    
    plt.tight_layout()
    plt.savefig('../paper/crb_correlations.pdf')

def crb_2d_all(comb=[[11,14]], vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True, relative=True):
    """Compare 2D constraints between all streams for a pair of parameters"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    names = get_done()
    N = len(names)
    
    # plot setup
    ncol = np.int64(np.ceil(np.sqrt(N)))
    nrow = np.int64(np.ceil(N/ncol))
    w_ = 10
    h_ = 1.1 * w_*nrow/ncol
    
    alpha = 1
    lw = 3
    frac = [0.8, 0.5, 0.2]
    
    # parameter pairs
    Ncomb = len(comb)

    for c in range(Ncomb):
        l, k = comb[c]
        plt.close()
        fig, ax = plt.subplots(nrow, ncol, figsize=(w_, h_), sharex=True, sharey=True)

        for i in range(N):
            plt.sca(ax[np.int64(i/ncol)][i%ncol])
            
            for e, Ndim in enumerate([3,4,6]):
                color = mpl.cm.bone(frac[e])
                
                fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, names[i], align, vlabel))
                cxi = fm['cxi']
                cx = stable_inverse(cxi)
                cx_2d = np.array([[cx[k][k], cx[k][l]], [cx[l][k], cx[l][l]]])
                if relative:
                    pk = pparams_fid[pid[k]].value
                    pl = pparams_fid[pid[l]].value
                    fid_2d = np.array([[pk**2, pk*pl], [pk*pl, pl**2]])
                    cx_2d = cx_2d / fid_2d * 100**2
                
                w, v = np.linalg.eig(cx_2d)
                if np.all(np.isreal(v)):
                    theta = np.degrees(np.arctan2(v[1][0], v[0][0]))
                    width = np.sqrt(w[0])*2
                    height = np.sqrt(w[1])*2
                    
                    e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=color, alpha=alpha, lw=lw)
                    plt.gca().add_patch(e)
            
            txt = plt.text(0.9, 0.9, full_name(names[i]), fontsize='small', transform=plt.gca().transAxes, ha='right', va='top')
            txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))
            if relative:
                plt.xlim(-20, 20)
                plt.ylim(-20,20)
            else:
                plt.gca().autoscale_view()
        
        plabels, units = get_parlabel([pid[k],pid[l]])
        if relative:
            punits = [' (%)' for x in units]
        else:
            punits = [' ({})'.format(x) if len(x) else '' for x in units]
        params = ['$\Delta$ {}{}'.format(x, y) for x,y in zip(plabels, punits)]
        
        for i in range(ncol):
            plt.sca(ax[nrow-1][i])
            plt.xlabel(params[0])
        
        for i in range(nrow):
            plt.sca(ax[i][0])
            plt.ylabel(params[1])
        
        for i in range(N, ncol*nrow):
            plt.sca(ax[np.int64(i/ncol)][i%ncol])
            plt.axis('off')
        
        #plt.sca(ax[1][3])
        #plt.setp(plt.gca().get_xticklabels(), visible=True)
        #plt.xlabel(params[0])

        plt.tight_layout(h_pad=0, w_pad=0)
    
    plt.savefig('../paper/crb2d_allstream.pdf')

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

def sky_all(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True, galactic=False):
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
        
        if e==0:
            p_all = np.copy(p_comp)
            pid_all = pid_comp[:]
        else:
            p_all = np.hstack([p_all, p_comp])
            pid_all = pid_all + pid_comp
    
    # calculate KL divergence to estimate whether constraints prior dominated
    prior_mat = priors('gd1', vary)
    prior_vec = np.diag(prior_mat)[6:]
    pparams_all = [pparams0[x] for x in pid_all]
    pparams_arr = np.array([x.value for x in pparams_all])
    prior_vec = prior_vec**-0.5 / pparams_arr
    
    kld = np.log(prior_vec/p_all)
    kld_min = 1e-2
    
    Ns, Np = np.shape(p_all)
    
    l = np.linspace(-180,180,100)
    b = np.zeros(100)
    plane_gal = coord.SkyCoord(l=l*u.deg, b=b*u.deg, frame=coord.Galactic)
    plane_eq = plane_gal.icrs

    #cx_ = plane_eq.ra.wrap_at(180*u.deg).rad
    cx_ = ((plane_eq.ra+90*u.deg).wrap_at(180*u.deg)).rad
    cy_ = plane_eq.dec.rad
    isort_ = np.argsort(cx_)
    
    plt.close()
    fig, ax = plt.subplots(3,3,figsize=(15,7), subplot_kw=dict(projection='mollweide'))
    
    for e, name in enumerate(names):
        crb_frac = p_all[e]
        #ls = [':' if np.isfinite(x) & (x<kld_min) else '-' for x in kld[e]]
        
        stream = np.load('../data/streams/mock_observed_{}.npy'.format(name))
        if not galactic:
            stream[0] += 90
        ceq = coord.SkyCoord(ra=stream[0]*u.deg, dec=stream[1]*u.deg, frame='icrs')
        cgal = ceq.galactic
        if galactic:
            if (np.min(cgal.l.deg)<180) & (np.max(cgal.l.deg)>180):
                wangle = 360*u.deg
            else:
                wangle = 180*u.deg
            cx = cgal.l.to(u.deg).wrap_at(wangle)
            cy = cgal.b.deg
        else:
            if (np.min(stream[0])<180) & (np.max(stream[0])>180):
                wangle = 360*u.deg
            else:
                wangle = 180*u.deg
            cx = ceq.ra.to(u.deg).wrap_at(wangle).value
            cy = ceq.dec.deg
        
        for i in range(Np):
            plt.sca(ax[np.int64(i/3)][i%3])
            color_index = np.array(crb_frac[:])
            color_index[color_index>0.2] = 0.2
            color_index /= 0.2
            color = mpl.cm.viridis(color_index[i])
            
            isort = np.argsort(cx)
            if np.isfinite(kld[e][i]) & (kld[e][i]<kld_min):
                plt.plot(np.radians(cx[isort]), np.radians(cy[isort]), ls='--', dashes=(0.2,0.2), color=color, lw=4) #, rasterized=True)
            else:
                print(name, i)
                plt.plot(np.radians(cx[isort]), np.radians(cy[isort]), ls='-', color=color, lw=4)
    
    for i in range(Np):
        plt.sca(ax[np.int64(i/3)][i%3])
        plt.text(0.9, 0.9, '$\Delta$ {}'.format(get_parlabel(pid_all[i])[0]), fontsize='medium', transform=plt.gca().transAxes, va='bottom', ha='left')
        plt.gca().xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi/3.))
        plt.gca().yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi/6.))
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.setp(plt.gca().get_yticklabels(), visible=False)
        plt.grid(color='0.7', ls=':')
        
        if not galactic:
            plt.plot(cx_[isort_], cy_[isort_], '--', color='0.7')
    
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
    
    plt.savefig('../paper/crb_onsky_gal{:d}.pdf'.format(galactic)) #, dpi=200)
    plt.savefig('../paper/crb_onsky_gal{:d}.png'.format(galactic)) #, dpi=200)

def sky_legend(galactic=False):
    """Label streams on the sky"""
    
    names = get_done()
    np.random.seed(538)
    print(names)
    lx_list = [133, 50, -140, -108, -15, -126, 72, 146, 126, 90, -98]
    ly_list = [-5, 17, 1, 1, 19, 64, -32, -26, 38, -3, 37]
    la_list = [70, 65, 90, 90, 92, 20, 0, 0, -50, -50, 35]
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(8,5), subplot_kw=dict(projection='mollweide'))
    plt.sca(ax)
    
    for e, name in enumerate(names):
        stream = np.load('../data/streams/mock_observed_{}.npy'.format(name))
        print(name, np.median(stream[0]))
        if not galactic:
            stream[0] += 90
        ceq = coord.SkyCoord(ra=stream[0]*u.deg, dec=stream[1]*u.deg, frame='icrs')
        cgal = ceq.galactic
        if galactic:
            if (np.min(cgal.l.deg)<180) & (np.max(cgal.l.deg)>180):
                wangle = 360*u.deg
            else:
                wangle = 180*u.deg
            cx = cgal.l.to(u.deg).wrap_at(wangle).value
            cy = cgal.b.deg
        else:
            if (np.min(stream[0])<180) & (np.max(stream[0])>180):
                wangle = 360*u.deg
            else:
                wangle = 180*u.deg
            cx = ceq.ra.to(u.deg).wrap_at(wangle).value
            cy = ceq.dec.deg
        
        isort = np.argsort(cx)
        color = (np.random.rand(1)*0.8)[0]
        scolor = '{}'.format(color)
        plt.plot(np.radians(cx[isort]), np.radians(cy[isort]), '-', color=scolor, lw=4)
        
        label = full_name(name)
        
        lx = np.radians(lx_list[e])
        ly = np.radians(ly_list[e])
        la = la_list[e]
        plt.text(lx, ly, label, fontsize='medium', color=scolor, rotation=la, ha='center', va='center')
    
    if galactic:
        plt.xlabel('l (deg)')
        plt.ylabel('b (deg)')
    else:
        l = np.linspace(-180,180,100)
        b = np.zeros(100)
        plane_gal = coord.SkyCoord(l=l*u.deg, b=b*u.deg, frame=coord.Galactic)
        plane_eq = plane_gal.icrs
        
        cx_ = ((plane_eq.ra+90*u.deg).wrap_at(wangle)).rad
        cy_ = plane_eq.dec.rad
        isort = np.argsort(cx_)
        plt.plot(cx_[isort], cy_[isort], '--', color='0.7')
        plt.text(np.radians(-132), np.radians(-34), 'b = 0', rotation=-35, fontsize='small', color='0.5', ha='center', va='center')
        plt.text(np.radians(72), np.radians(66), 'b = 0', rotation=30, fontsize='small', color='0.5', ha='center', va='center')
        
        tx_label = [150, 210, 270, 330, 30]
        for e, tx in enumerate([-120, -60, 0, 60, 120]):
            plt.text(np.radians(tx), np.radians(-60), '{:.0f}$\degree$'.format(tx_label[e]), ha='center', va='center', fontsize='small')
            
        
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.setp(plt.gca().get_yticklabels(), fontsize='small')
        plt.xlabel('R.A. (deg)', labelpad=30)
        plt.ylabel('Dec (deg)')
    
    plt.gca().xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi/3.))
    plt.gca().yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi/6.))
    plt.grid(color='0.7', ls=':')
    
    plt.tight_layout()
    plt.savefig('../paper/sky_legend_gal{:d}.pdf'.format(galactic))

def nstream_improvement(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', component='halo', align=True, relative=False, diag=False, itarget=0, flag_in=True):
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
    
    da = 3.2
    ncol = 2
    nrow = np.int64(Nvar/ncol)
    w = 4 * da
    h = nrow * da
    np.random.seed(938)

    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(w,h), sharex='all')
    
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
        
        plt.xlim(0.9, 13)
        
        for k in range(Nvar):
            plt.sca(ax[k%ncol][np.int64(k/ncol)])
            nst_off = (np.random.randn(Ncomb)-0.5)*0.01+1
            if (i==0) & (k==0):
                plt.plot(nst*nst_off, p_all[:,k], 'o', color='0.4', ms=2, label='Single combination\nof N streams')
                plt.plot(Nmulti, median[k], 'wo', mec='k', mew=2, ms=10, label='Median over different\ncombinations of N\nstreams')
            else:
                plt.plot(nst*nst_off, p_all[:,k], 'o', color='0.4', ms=2)
                plt.plot(Nmulti, median[k], 'wo', mec='k', mew=2, ms=10)
            
            # highlight combinations with the best stream
            if diag:
                Ncomb, Nstream = np.shape(comb_all)
                ilabel = ''
                
                if flag_in:
                    has_best = np.array([True if itarget in comb_all[i,:] else False for i in range(Ncomb)])
                    if (i==0) & (k==0):
                        ilabel = 'Has {}'.format(done[itarget])
                else:
                    has_best = np.array([True if itarget not in comb_all[i,:] else False for i in range(Ncomb)])
                    if (i==0) & (k==0):
                        ilabel = 'Without {}'.format(done[itarget])
                
                plt.plot((nst*nst_off)[has_best], p_all[:,k][has_best], 'o', color='orange', ms=2, label=ilabel)
            
            if Nmulti<=3:
                if Nmulti==1:
                    Nmin = 3
                    va = 'center'
                    ha = 'left'
                    orientation = 0
                    yoff = 1
                    xoff = 1.15
                else:
                    Nmin = 1
                    va = 'bottom'
                    ha = 'center'
                    orientation = 90
                    yoff = 1.2
                    xoff = 1.1
                ids_min = p_all[:,k].argsort()[:Nmin]
                
                for j_ in range(Nmin):
                    best_names = [done[np.int64(i_)] for i_ in comb[ids_min[j_]][:Nmulti]]
                    #print(k, j_, best_names)
                    label = ' + '.join(best_names)
                    
                    if j_==0:
                        label = '$\it{best}$: ' + label
                        
                        if Nmulti==1:
                            plt.plot([Nmulti*1.05, Nmulti*1.1], [p_all[ids_min[j_],k], p_all[ids_min[j_],k]], '-', color='0.', lw=0.75)
                    
                    mpl.rc('text', usetex=True)
                    plt.text(Nmulti*xoff, p_all[ids_min[j_],k]*yoff, '{}'.format(label), fontsize='xx-small', va=va, ha=ha, rotation=orientation)
                    mpl.rc('text', usetex=False)
                    if Nmin==1:
                        plt.plot([Nmulti*1.05, Nmulti*1.1], [p_all[ids_min[j_],k], p_all[ids_min[j_],k]], '-', color='0.', lw=0.75)
                        plt.plot([Nmulti*1.1, Nmulti*1.1], [p_all[ids_min[j_],k], p_all[ids_min[j_],k]*1.1], '-', color='0.', lw=0.75)

    for k in range(Nvar):
        plt.sca(ax[k%ncol][np.int64(k/ncol)])
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        if relative:
            plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        
        plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        plt.ylabel(params[k])
        
        if k==0:
            plt.legend(frameon=False, fontsize='x-small', loc=1)
    
        if k%ncol==nrow-1:
            plt.xlabel('Number of streams in a combination')
    
    plt.tight_layout()
    if diag:
        plt.savefig('../plots/nstream_improvement_diag_w{:1d}_{:02d}.png'.format(flag_in, itarget))
    else:
        plt.savefig('../paper/nstream_improvement.pdf')
        plt.savefig('../paper/nstream_improvement.png')

# applications
def ar(current=False, vary=['progenitor', 'bary', 'halo'], Nsight=50):
    """Explore constraints on radial acceleration, along the progenitor line"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    t = Table.read('../data/crb/ar_orbital_summary_{}_sight{:d}.fits'.format(vlabel, Nsight))
    N = len(t)
    fapo = t['rapo']/np.max(t['rapo'])
    fapo = t['rapo']/100
    flen = t['length']/20
    flen = 0.8 * flen / np.max(flen)
    fcolor = 0.8-flen
    #fcolor[fcolor<0] = 0
    #print(fcolor)
    #print(t.colnames)
    
    # sort in fiducial order
    names = get_done()
    order = np.zeros(len(names), dtype=np.int64)
    tlist = list(t['name'])
    
    for e, n in enumerate(names):
        order[e] = tlist.index(n)
    
    # best acceleration
    armin = np.median(t['armin'], axis=1)
    armin_err = 0.5 * (np.percentile(t['armin'], 84, axis=1) - np.percentile(t['armin'], 16, axis=1))
    rmin = np.median(t['rmin'], axis=1)
    rmin_err = 0.5 * (np.percentile(t['rmin'], 84, axis=1) - np.percentile(t['rmin'], 16, axis=1))
    
    #acolor = np.log(armin)
    acolor = armin
    acolor = flen
    fcolor = acolor/np.max(acolor)+0.2
    lw = 4
    
    plt.close()
    fig = plt.figure(figsize=(13,7))
    
    gs1 = mpl.gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.5])
    gs1.update(left=0.07, right=0.95, bottom=0.6, top=0.95, hspace=0.1, wspace=0.3)
    gs2 = mpl.gridspec.GridSpec(1, 3)
    gs2.update(left=0.07, right=0.95, bottom=0.1, top=0.45, hspace=0.1, wspace=0.3)
    
    ax = [[],[]]
    for i in range(3):
        ax[0] = ax[0] + [plt.subplot(gs1[i])]
        ax[1] = ax[1] + [plt.subplot(gs2[i])]
    
    for i_ in range(N):
        i = order[i_]
        color = mpl.cm.binary(fcolor[i])
        
        plt.sca(ax[0][0])
        if i_==0:
            plt.plot(t['r'][i][0], t['dar'][i][1]/t['ar'][i][1], '--', color='0.7', lw=10, alpha=0.2, zorder=0, label='$a_r$')
        plt.plot(t['r'][i][0], t['dar'][i][1], '-', color=color, lw=lw, alpha=0.7, label=t['name'][i])
        
        plt.sca(ax[0][1])
        plt.plot(t['r'][i][0], t['ar'][i][1], '-', color=color, lw=lw, alpha=0.7)
        
    plt.sca(ax[0][0])
    plt.xlabel('R (kpc)')
    plt.ylabel('$\Delta$ $a_r$ (pc Myr$^{-2}$)')
    plt.legend(ncol=2, frameon=False, fontsize='medium', handlelength=1, columnspacing=1, loc=2, bbox_to_anchor=(2.3,1))
    
    plt.sca(ax[0][1])
    plt.xlabel('R (kpc)')
    plt.ylabel('$\Delta$ $a_r$ / $a_r$')
    
    ax[0][0].set_yscale('log')
    ax[0][1].set_yscale('log')
    ax[0][2].axis('off')
    
    plt.sca(ax[1][0])
    plt.scatter(t['length'], armin, c=fcolor, cmap='binary', vmin=0, vmax=1, s=70, edgecolors='k')
    plt.errorbar(t['length'], armin, yerr=armin_err, color='0.3', fmt='none', zorder=0)
    
    plt.xlabel('Length (deg)')
    plt.ylabel('min ($\Delta$ $a_r$ / $a_r$)')
    #plt.ylim(0, 3.5)
    
    plt.sca(ax[1][1])
    a = np.linspace(0,90,100)
    plt.plot(a, a, 'k-')
    plt.scatter(t['rcur'], rmin, c=fcolor, cmap='binary', vmin=0, vmax=1, s=70, edgecolors='k')
    plt.errorbar(t['rcur'], rmin, yerr=rmin_err, color='0.3', fmt='none', zorder=0)
    plt.xlabel('$R_{cur}$ (kpc)')
    plt.ylabel('$R_{min}$ (kpc)')
    
    plt.xlim(0,90)
    plt.ylim(0,90)
    
    plt.sca(ax[1][2])
    a = np.linspace(0,90,100)
    plt.plot(a, a, 'k-')
    plt.scatter(t['rapo'], rmin, c=fcolor, cmap='binary', vmin=0, vmax=1, s=70, edgecolors='k')
    plt.errorbar(t['rapo'], rmin, yerr=rmin_err, color='0.3', fmt='none', zorder=0)
    plt.xlabel('$R_{apo}$ (kpc)')
    plt.ylabel('$R_{min}$ (kpc)')
    
    plt.xlim(0,90)
    plt.ylim(0,90)
    
    #gs1.tight_layout(fig)
    #gs2.tight_layout(fig)
    plt.savefig('../paper/ar_crb.pdf')

# interpretation
def orbit_corr(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True, Nsight=1):
    """Show how CRBs on different potential parameters depend on orbital properties of the progenitor"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    names = get_done()
    
    # get fiducial halo parameters
    c_halo = np.load('../data/crb/cx_collate_multi1_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(errmode, Ndim, align, vlabel, 'halo'))
    p_halo = c_halo['p_rel']
    
    # get fiducial bary parameters
    c_bary = np.load('../data/crb/cx_collate_multi1_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(errmode, Ndim, align, vlabel, 'bary'))
    p_bary = c_bary['p_rel']

    #p = np.hstack([p_bary[:,2][:,np.newaxis], p_halo, p_halo[:,0][:,np.newaxis]/p_halo[:,1][:,np.newaxis]])
    p = np.hstack([p_bary[:,2][:,np.newaxis], p_halo]) #, np.sqrt(4*p_halo[:,0][:,np.newaxis]**2 + p_halo[:,1][:,np.newaxis]**2) ])
    t = Table.read('../data/crb/ar_orbital_summary_{}_sight{:d}.fits'.format(vlabel, Nsight))
    
    nrow = 5
    ncol = 5
    da = 2.5
    cname = ['length', 'rapo', 'lx', 'lz']
    xvar = [t[i_] for i_ in cname]
    #xvar[1] = (pparams_fid[6].value - t['rperi']) / (pparams_fid[6].value - t['rapo'])
    #xvar[1] = t['rperi']/t['rapo']
    #xvar[3] = t['lx'] / np.sqrt(t['ly']**2 + t['lz']**2)
    #xvar[2] = np.sqrt(t['lx']**2 + t['ly']**2)
    #xvar[2] = np.sqrt(t['Labs'][:,0]**2 + t['Labs'][:,1]**2)/t['Lmod']
    xvar = xvar + [t['ecc']]
    
    xlabels = ['Length (deg)', 'R$_{apo}$ (kpc)', '|L$_x$|/|L|', '|L$_z$|/|L|', 'e']
    #ylabels = ['log $M_d$', '$V_h$', '$R_h$', '$q_x$', '$q_z$', '$V_h$ / $R_h$']
    ylabels = ['log $M_d$', '$V_h$', '$R_h$', '$q_x$', '$q_z$'] #, '$M_h$']
    dylabels = ['$\Delta$ {}'.format(s) for s in ylabels]
    
    mask = t['name']!='ssangarius'
    cback = '#FFD8A7'
    cback = [1, 237/255, 214/255]
    cback = [0.5, 0.5, 0.5]
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*da, nrow*da), sharex='col', sharey='row')
    
    for i in range(nrow):
        for j in range(ncol):
            plt.sca(ax[i][j])
            #plt.plot(xvar[j][mask], p[:,i][mask], 'o', ms=5, color='0.2')
            plt.scatter(xvar[j][mask], p[:,i][mask], c=xvar[0][mask], vmax=30, cmap='binary', s=65, edgecolors='k')
            
            corr = scipy.stats.pearsonr(xvar[j][mask], p[:,i][mask])
            fs = np.abs(corr[0])*10+7
            txt = plt.text(0.9, 0.9, '{:.3g}'.format(corr[0]), size=fs, transform=plt.gca().transAxes, ha='right', va='top')
            #txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))
            
            if np.abs(corr[0])>0.5:
                balpha = 0
            else:
                balpha = 0.6 - np.abs(corr[0])
            ctuple = tuple(cback + [balpha])
            plt.gca().set_facecolor(color=ctuple)
            
            if j==0:
                plt.ylabel(dylabels[i])
            if i==nrow-1:
                plt.xlabel(xlabels[j])
            
            
        
        plt.gca().set_yscale('log')
        if i==2:
            plt.ylim(1e-1,1e1)
        else:
            plt.ylim(8e-3,8e-1)
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

#def ar(current=False, vary=['progenitor', 'bary', 'halo']):
    #"""Explore constraints on radial acceleration, along the progenitor line"""
    #pid, dp_fid, vlabel = get_varied_pars(vary)
    #t = Table.read('../data/crb/ar_orbital_summary_{}.fits'.format(vlabel))
    #N = len(t)
    #fapo = t['rapo']/np.max(t['rapo'])
    #fapo = t['rapo']/100
    #flen = t['length']/(np.max(t['length']) + 10)
    #fcolor = fapo
    
    #plt.close()
    #fig, ax = plt.subplots(1, 4, figsize=(20,5))
    
    #for i in range(N):
        #color = mpl.cm.bone(fcolor[i])
        #lw = flen[i] * 5
        
        #plt.sca(ax[0])
        #plt.plot(t['r'][i], t['ar'][i], '-', color=color, lw=lw)
        
    #plt.xlabel('R (kpc)')
    #plt.ylabel('$\Delta$ $a_r$ / $a_r$')
    #plt.ylim(0, 2.5)
    
    #plt.sca(ax[1])
    #plt.scatter(t['length'], t['armin'], c=fcolor, cmap='bone', vmin=0, vmax=1)
    
    #plt.xlabel('Length (deg)')
    #plt.ylabel('min $\Delta$ $a_r$')
    ##plt.ylim(0, 1)
    
    #plt.sca(ax[2])
    #a = np.linspace(0,90,100)
    #plt.plot(a, a, 'k-')
    #plt.plot(a, 2*a, 'k--')
    #plt.plot(a, 3*a, 'k:')
    #plt.scatter(t['rcur'], t['rmin'], c=fcolor, cmap='bone', vmin=0, vmax=1)
    #plt.xlabel('$R_{cur}$ (kpc)')
    #plt.ylabel('$R_{min}$ (kpc)')
    
    #plt.xlim(0,90)
    #plt.ylim(0,90)
    
    #plt.sca(ax[3])
    #a = np.linspace(0,90,100)
    #plt.plot(a, a, 'k-')
    #plt.plot(a, 2*a, 'k--')
    #plt.plot(a, 3*a, 'k:')
    #plt.scatter(t['rapo'], t['rmin'], c=fcolor, cmap='bone', vmin=0, vmax=1)
    #plt.xlabel('$R_{apo}$ (kpc)')
    #plt.ylabel('$R_{min}$ (kpc)')
    
    #plt.xlim(0,90)
    #plt.ylim(0,90)
    
    #plt.tight_layout()
    #plt.savefig('../plots/ar_crb_{}.pdf'.format(vlabel))

def min_ar():
    """"""
    t = Table.read('../data/crb/ar_orbital_summary.fits')
    print(t.colnames)
    imin = 10
    
    plt.close()
    plt.figure()
    
    for s in range(11):
        idmin = np.r_[True, t['dar'][s][1:] < t['dar'][s][:-1]] & np.r_[t['dar'][s][:-1] < t['dar'][s][1:], True]
        print(t['name'][s], t['r'][s][idmin], t['rmin'][s])
        plt.plot(t['r'][s], t['dar'][s], '-')

# expected ar
def latte_ar(vary=['progenitor', 'bary', 'halo', 'dipole', 'quad', 'octu'], xmax=100):
    """"""
    plt.close()
    plt.figure(figsize=(9,6))
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    Nsight = 50
    t = Table.read('../data/ar_constraints_{}_sight{}.fits'.format(vlabel, Nsight))
    
    colors = [mpl.cm.magma(i_) for i_ in [0.2, 0.5, 0.8]]
    
    for e, halo in enumerate(['i', 'f', 'm']):
        color = colors[e]
        
        f = h5py.File('/home/ana/data/latte/m12{:1s}/snapshot_600.0.hdf5'.format(halo), 'r')
        dm = f['PartType0']

        x = dm['Coordinates'][::20] / 0.7
        pot = dm['Potential'][::20]
        
        # center halo
        imin = np.argmin(pot)
        xmin = x[imin]
        x = x - xmin
        r = np.linalg.norm(x, axis=1)
        
        # get median potential in radial shells
        r_bins = np.linspace(0, xmax, 50)
        idx  = np.digitize(r, bins=r_bins)
        pot_med = np.array([np.nanmedian(pot[idx==i]) for i in range(1, len(r_bins))])
        r_med = r_bins[1:]
        
        # calculate radial acceleration
        ar = (pot_med[2:] - pot_med[:-2]) / (r_med[2:] - r_med[:-2]) * u.km**2 * u.s**-2 * u.kpc**-1
        r_ar = r_med[1:-1]
        #print(e, np.all(np.isfinite(pot_med)), np.shape(x))
        #print(pot_med)
        
        # stream constraints
        r_stream = t['rcur']
        ind = np.argsort(r_stream)
        fint = scipy.interpolate.interp1d(r_ar, ar, )
        ar_stream = fint(r_stream)*ar.unit
        ar_err = ar_stream * t['armin']
        ar_err = ar_stream * 0.05
        
        #plt.plot(x[:,0], x[:,1], 'o', label=halo)
        
        plt.plot(r_ar, ar.to(u.pc*u.Myr**-2), '-', lw=2, color=color, label='Latte m12{:1s}'.format(halo))
        plt.plot(r_stream, ar_stream.to(u.pc*u.Myr**-2), 'o', color=color)
        plt.errorbar(r_stream, ar_stream.to(u.pc*u.Myr**-2).value, yerr=ar_err.to(u.pc*u.Myr**-2).value, color=color, fmt='none', zorder=0)
        
        plt.fill_between(r_stream[ind], ((ar_stream-ar_err).to(u.pc*u.Myr**-2).value)[ind], ((ar_stream+ar_err).to(u.pc*u.Myr**-2).value)[ind], color=color, alpha=0.3)
        
        f.close()

    plt.legend(frameon=False, fontsize='medium', handlelength=1)
    plt.gca().set_yscale('log')
    plt.ylim(0.5, 75)
    plt.xlabel('r (kpc)')
    plt.ylabel('$a_r$ (pc Myr$^{-2}$)')

    plt.tight_layout()
    plt.savefig('../plots/latte_ar_multistream.pdf')

# tables
def table_obsmodes(verbose=True):
    """Save part of the latex table with information on observing modes"""
    
    obsmodes = pickle.load(open('../data/observing_modes.info', 'rb'))
    modes = ['fiducial', 'desi', 'gaia']
    names = obsmode_name(modes)
    
    fout = open('../paper/obsmodes.tex', 'w')
    
    for e, mode in enumerate(modes):
        sigmas = obsmodes[mode]['sig_obs'].tolist()
        line = '{} & {:.1f} & {:.1f} & {:.0f} & {:.1f} & {:.1f} \\\\'.format(names[e], *sigmas)
        line = line.replace('nan', 'N/A')
        
        if verbose: print(line)
        
        fout.write('{}\n'.format(line))
    
    fout.close()

def orbit_properties(name):
    """Print properties of a stream"""
    t = Table.read('../data/crb/ar_orbital_summary.fits')
    t = t[t['name']==name]
    
    for k in t.colnames:
        print(k, np.array(t[k])[0])

def orbit_properties_all(p):
    """"""
    t = Table.read('../data/crb/ar_orbital_summary.fits')
    if p in t.colnames:
        Table([t['name'], t[p]]).pprint(max_lines=20)
