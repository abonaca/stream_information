from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy
import astropy.units as u
from astropy.constants import G
from astropy.table import Table
import astropy.coordinates as coord
import gala.coordinates as gc

cold = ['ACS', 'ATLAS', 'Ach', 'Alp', 'Coc', 'GD1', 'Hyl', 'Kwa', 'Let', 'Mol', 'Mur', 'NGC5466', 'Oph', 'Ori', 'Orp', 'PS1A', 'PS1B', 'PS1C', 'PS1D', 'PS1E', 'Pal5', 'Pho', 'San', 'Sca', 'Sty', 'TriPis', 'WG1', 'WG2', 'WG3', 'WG4']

def show_streams():
    """Read streams from Mateu+(2017) and plot them in equatorial coordinates"""

    t = Table.read('/home/ana/projects/python/galstreams/footprints/galstreams.footprint.ALL.dat', format='ascii.commented_header')
    t.pprint()
    
    ids = np.unique(t['NSt'])
    print(np.array(np.unique(t['IDst'])))
    
    cold = ['ACS', 'ATLAS', 'Ach', 'Alp', 'Coc', 'GD1', 'Hyl', 'Kwa', 'Let', 'Mol', 'Mur', 'NGC5466', 'Oph', 'Ori', 'Orp', 'PS1A', 'PS1B', 'PS1C', 'PS1D', 'PS1E', 'Pal5', 'Pho', 'San', 'Sca', 'Sty', 'TriPis', 'WG1', 'WG2', 'WG3', 'WG4']
    print(len(cold))
    
    clouds = ['Her', 'Eri', 'Mon', 'PAndAS', 'Pis', 'SgrL10', 'Tri', 'VODVSS', 'EBS']
    
    short = ['Pal15', 'Cet']
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,10))
    
    plt.sca(ax[0])
    for i, n in enumerate(cold):
        ind = t['IDst']==n
        color = mpl.cm.Spectral(i/len(cold))
        plt.plot(t['RA_deg'][ind], t['DEC_deg'][ind], 'o', label=n)
        plt.text(t['RA_deg'][ind][0], t['DEC_deg'][ind][0], n, fontsize='x-small')
    
    plt.legend(fontsize='xx-small', ncol=6, handlelength=0.2)
    plt.gca().invert_xaxis()
    
    plt.sca(ax[1])
    for i, n in enumerate(clouds+short):
        ind = t['IDst']==n
        plt.plot(t['RA_deg'][ind], t['DEC_deg'][ind], 'o', label=n)
    
    plt.legend(fontsize='x-small', ncol=6, handlelength=0.2)
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('../plots/halo_structures.png', dpi=200)

class Stream():
    def __init__(self, x0=[]*u.kpc, v0=[]*u.km/u.s, progenitor={'coords': 'galactocentric', 'observer': {}, 'pm_polar': False}, potential='nfw', pparams=[], minit=2e4*u.Msun, mfinal=2e4*u.Msun, rcl=20*u.pc, dr=0.5, dv=2*u.km/u.s, dt=1*u.Myr, age=6*u.Gyr, nstars=600, integrator='lf'):
        """Initialize """
        setup = {}
        if progenitor['coords']=='galactocentric':
            setup['x0'] = x0
            setup['v0'] = v0
        elif (progenitor['coords']=='equatorial') & (len(progenitor['observer'])!=0):
            if progenitor['pm_polar']:
               a = v0[1].value
               phi = v0[2].value
               v0[1] = a*np.sin(phi)*u.mas/u.yr
               v0[2] = a*np.cos(phi)*u.mas/u.yr
            # convert positions
            xeq = coord.SkyCoord(x0[0], x0[1], x0[2], **progenitor['observer'])
            xgal = xeq.transform_to(coord.Galactocentric)
            setup['x0'] = [xgal.x.to(u.kpc), xgal.y.to(u.kpc), xgal.z.to(u.kpc)]*u.kpc
            
            # convert velocities
            setup['v0'] = gc.vhel_to_gal(xeq.icrs, rv=v0[0], pm=v0[1:], **vsun)
            #setup['v0'] = [v.to(u.km/u.s) for v in vgal]*u.km/u.s
        else:
            raise ValueError('Observer position needed!')
        
        setup['dr'] = dr
        setup['dv'] = dv
        
        setup['minit'] = minit
        setup['mfinal'] = mfinal
        setup['rcl'] = rcl
        
        setup['dt'] = dt
        setup['age'] = age
        setup['nstars'] = nstars
        setup['integrator'] = integrator
        
        setup['potential'] = potential
        setup['pparams'] = pparams
        
        self.setup = setup
        self.setup_aux = {}

        self.fill_intid()
        self.fill_potid()
        
        self.st_params = self.format_input()
    
    def fill_intid(self):
        """Assign integrator ID for a given integrator choice
        Assumes setup dictionary has an 'integrator' key"""
        
        if self.setup['integrator']=='lf':
            self.setup_aux['iaux'] = 0
        elif self.setup['integrator']=='rk':
            self.setup_aux['iaux'] = 1
            
    def fill_potid(self):
        """Assign potential ID for a given potential choice
        Assumes d has a 'potential' key"""
        
        if self.setup['potential']=='nfw':
            self.setup_aux['paux'] = 3
        elif self.setup['potential']=='log':
            self.setup_aux['paux'] = 2
        elif self.setup['potential']=='point':
            self.setup_aux['paux'] = 0
        elif self.setup['potential']=='gal':
            self.setup_aux['paux'] = 4
        elif self.setup['potential']=='lmc':
            self.setup_aux['paux'] = 6
        elif self.setup['potential']=='dipole':
            self.setup_aux['paux'] = 8
        elif self.setup['potential']=='quad':
            self.setup_aux['paux'] = 9
            
    def format_input(self):
        """Format input parameters for streakline.stream"""
        
        p = [None]*12
        
        # progenitor position
        p[0] = self.setup['x0'].si.value
        p[1] = self.setup['v0'].si.value
        
        # potential parameters
        p[2] = [x.si.value for x in self.setup['pparams']]
        
        # stream smoothing offsets
        p[3] = [self.setup['dr'], self.setup['dv'].si.value]
        
        # potential and integrator choice
        p[4] = self.setup_aux['paux']
        p[5] = self.setup_aux['iaux']
        
        # number of steps and stream stars
        p[6] = int(self.setup['age']/self.setup['dt'])
        p[7] = int(p[6]/self.setup['nstars'])
        
        # cluster properties
        p[8] = self.setup['minit'].si.value
        p[9] = self.setup['mfinal'].si.value
        p[10] = self.setup['rcl'].si.value
        
        # time step
        p[11] = self.setup['dt'].si.value
        
        return p
    
    def generate(self):
        """Create streakline model for a stream of set parameters"""
        
        #xm1, xm2, xm3, xp1, xp2, xp3, vm1, vm2, vm3, vp1, vp2, vp3 = streakline.stream(*p)
        stream = streakline.stream(*self.st_params)
        
        self.leading = {}
        self.leading['x'] = stream[:3]*u.m
        self.leading['v'] = stream[6:9]*u.m/u.s

        self.trailing = {}
        self.trailing['x'] = stream[3:6]*u.m
        self.trailing['v'] = stream[9:12]*u.m/u.s
    
    def observe(self, mode='cartesian', units=[], errors=[], nstars=-1, sequential=False, present=[], logerr=False, observer={'z_sun': 0.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_ra': 300*u.deg, 'galcen_dec': 20*u.deg}, vobs={'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}, footprint='none', rotmatrix=None):
        """Observe the stream
        stream.obs holds all observations
        stream.err holds all errors"""
        
        x = np.concatenate((self.leading['x'].to(u.kpc).value, self.trailing['x'].to(u.kpc).value), axis=1) * u.kpc
        v = np.concatenate((self.leading['v'].to(u.km/u.s).value, self.trailing['v'].to(u.km/u.s).value), axis=1) * u.km/u.s
        
        if mode=='cartesian':
            # returns coordinates in following order
            # x(x, y, z), v(vx, vy, vz)
            if len(units)<2:
                units.append(self.trailing['x'].unit)
                units.append(self.trailing['v'].unit)
            
            if len(errors)<2:
                errors.append(0.2*u.kpc)
                errors.append(2*u.km/u.s)
            
            # positions
            x = x.to(units[0])
            ex = np.ones(np.shape(x))*errors[0]
            ex = ex.to(units[0])
        
            # velocities
            v = v.to(units[1])
            ev = np.ones(np.shape(v))*errors[1]
            ev = ev.to(units[1])
        
            self.obs = np.concatenate([x,v]).value
            self.err = np.concatenate([ex,ev]).value
            
        elif mode=='equatorial':
            # assumes coordinates in the following order:
            # ra, dec, distance, vrad, mualpha, mudelta
            if len(units)!=6:
                units = [u.deg, u.deg, u.kpc, u.km/u.s, u.mas/u.yr, u.mas/u.yr]
            
            if len(errors)!=6:
                errors = [0.2*u.deg, 0.2*u.deg, 0.5*u.kpc, 1*u.km/u.s, 0.2*u.mas/u.yr, 0.2*u.mas/u.yr]
            
            # define reference frame
            xgal = coord.Galactocentric(x, **observer)
            #frame = coord.Galactocentric(**observer)
            
            # convert
            xeq = xgal.transform_to(coord.ICRS)
            veq = gc.vgal_to_hel(xeq, v, **vobs)
            
            # store coordinates
            ra, dec, dist = [xeq.ra.to(units[0]), xeq.dec.to(units[1]), xeq.distance.to(units[2])]
            vr, mua, mud = [veq[2].to(units[3]), veq[0].to(units[4]), veq[1].to(units[5])]
            
            obs = np.hstack([ra, dec, dist, vr, mua, mud]).value
            obs = np.reshape(obs,(6,-1))
            
            if footprint=='sdss':
                infoot = dec > -2.5*u.deg
                obs = obs[:,infoot]
            
            if np.all(rotmatrix)!=None:
                xi, eta  = myutils.rotate_angles(obs[0], obs[1], rotmatrix)
                obs[0] = xi
                obs[1] = eta
            
            self.obs = obs
            
            # store errors
            err = np.ones(np.shape(self.obs))
            if logerr:
                for i in range(6):
                    err[i] *= np.exp(errors[i].to(units[i]).value)
            else:
                for i in range(6):
                    err[i] *= errors[i].to(units[i]).value
            self.err = err
    
            

        self.obsunit = units
        self.obserror = errors
        
        # randomly select nstars from the stream
        if nstars>-1:
            if sequential:
                select = np.linspace(0, np.shape(self.obs)[1], nstars, endpoint=False, dtype=int)
            else:
                select = np.random.randint(low=0, high=np.shape(self.obs)[1], size=nstars)
            self.obs = self.obs[:,select]
            self.err = self.err[:,select]
        
        # include only designated dimensions
        if len(present)>0:
            self.obs = self.obs[present]
            self.err = self.err[present]

            self.obsunit = [ self.obsunit[x] for x in present ]
            self.obserror = [ self.obserror[x] for x in present ]
    
    def prog_orbit(self):
        """Generate progenitor orbital history"""
        orbit = streakline.orbit(self.st_params[0], self.st_params[1], self.st_params[2], self.st_params[4], self.st_params[5], self.st_params[6], self.st_params[11], -1)
        
        self.orbit = {}
        self.orbit['x'] = orbit[:3]*u.m
        self.orbit['v'] = orbit[3:]*u.m/u.s
    
    def project(self, name, N=1000, nbatch=-1):
        """Project the stream from observed to native coordinates"""
        
        poly = np.loadtxt("../data/{0:s}_all.txt".format(name))
        self.streak = np.poly1d(poly)
        
        self.streak_x = np.linspace(np.min(self.obs[0])-2, np.max(self.obs[0])+2, N)
        self.streak_y = np.polyval(self.streak, self.streak_x)
        
        self.streak_b = np.zeros(N)
        self.streak_l = np.zeros(N)
        pdot = np.polyder(poly)
    
        for i in range(N):
            length = scipy.integrate.quad(self._delta_path, self.streak_x[0], self.streak_x[i], args=(pdot,))
            self.streak_l[i] = length[0]
        
        XB = np.transpose(np.vstack([self.streak_x, self.streak_y]))
        
        n = np.shape(self.obs)[1]
        
        if nbatch<0:
            nstep = 0
            nbatch = -1
        else:
            nstep = np.int(n/nbatch)
        
        i1 = 0
        i2 = nbatch

        for i in range(nstep):
            XA = np.transpose(np.vstack([np.array(self.obs[0][i1:i2]), np.array(self.obs[1][i1:i2])]))
            self.emdist(XA, XB, i1=i1, i2=i2)
            
            i1 += nbatch
            i2 += nbatch

        XA = np.transpose(np.vstack([np.array(self.catalog['ra'][i1:]), np.array(self.catalog['dec'][i1:])]))
        self.emdist(XA, XB, i1=i1, i2=n)
        
        #self.catalog.write("../data/{0:s}_footprint_catalog.txt".format(self.name), format='ascii.commented_header')
    
    def emdist(self, XA, XB, i1=0, i2=-1):
        """"""
        
        distances = scipy.spatial.distance.cdist(XA, XB)
        
        self.catalog['b'][i1:i2] = np.min(distances, axis=1)
        imin = np.argmin(distances, axis=1)
        self.catalog['b'][i1:i2][self.catalog['dec'][i1:i2]<self.streak_y[imin]] *= -1
        self.catalog['l'][i1:i2] = self.streak_l[imin]
        
    def _delta_path(self, x, pdot):
        """Return integrand for calculating length of a path along a polynomial"""
        
        return np.sqrt(1 + np.polyval(pdot, x)**2)
    
    def plot(self, mode='native', fig=None, color='k', **kwargs):
        """Plot stream"""
        
        # Plotting
        if fig==None:
            plt.close()
            plt.figure()
            ax = plt.axes([0.12,0.1,0.8,0.8])
        
        if mode=='native':
            # Color setup
            cindices = np.arange(self.setup['nstars'])            # colors of stream particles
            nor = mpl.colors.Normalize(vmin=0, vmax=self.setup['nstars'])    # colormap normalization
            
            plt.plot(self.setup['x0'][0].to(u.kpc).value, self.setup['x0'][2].to(u.kpc).value, 'wo', ms=10, mew=2, zorder=3)
            plt.scatter(self.trailing['x'][0].to(u.kpc).value, self.trailing['x'][2].to(u.kpc).value, s=30, c=cindices, cmap='winter', norm=nor, marker='o', edgecolor='none', lw=0, alpha=0.1)
            plt.scatter(self.leading['x'][0].to(u.kpc).value, self.leading['x'][2].to(u.kpc).value, s=30, c=cindices, cmap='autumn', norm=nor, marker='o', edgecolor='none', lw=0, alpha=0.1)
            
            plt.xlabel("X (kpc)")
            plt.ylabel("Z (kpc)")
            
        elif mode=='observed':
            plt.subplot(221)
            plt.plot(self.obs[0], self.obs[1], 'o', color=color, **kwargs)
            plt.xlabel("RA")
            plt.ylabel("Dec")
            
            plt.subplot(223)
            plt.plot(self.obs[0], self.obs[2], 'o', color=color, **kwargs)
            plt.xlabel("RA")
            plt.ylabel("Distance")
            
            plt.subplot(222)
            plt.plot(self.obs[3], self.obs[4], 'o', color=color, **kwargs)
            plt.xlabel("V$_r$")
            plt.ylabel("$\mu\\alpha$")
            
            plt.subplot(224)
            plt.plot(self.obs[3], self.obs[5], 'o', color=color, **kwargs)
            plt.xlabel("V$_r$")
            plt.ylabel("$\mu\delta$")
            
        
        plt.tight_layout()
        #plt.minorticks_on()

    def read(self, fname, units={'x': u.kpc, 'v': u.km/u.s}):
        """Read stream star positions from a file"""
        
        t = np.loadtxt(fname).T
        n = np.shape(t)[1]
        ns = int((n-1)/2)
        self.setup['nstars'] = ns
        
        # progenitor
        self.setup['x0'] = t[:3,0] * units['x']
        self.setup['v0'] = t[3:,0] * units['v']
        
        # leading tail
        self.leading = {}
        self.leading['x'] = t[:3,1:ns+1] * units['x']
        self.leading['v'] = t[3:,1:ns+1] * units['v']
        
        # trailing tail
        self.trailing = {}
        self.trailing['x'] = t[:3,ns+1:] * units['x']
        self.trailing['v'] = t[3:,ns+1:] * units['v']
    
    def save(self, fname):
        """Save stream star positions to a file"""
        
        # define table
        t = Table(names=('x', 'y', 'z', 'vx', 'vy', 'vz'))
        
        # add progenitor info
        t.add_row(np.ravel([self.setup['x0'].to(u.kpc).value, self.setup['v0'].to(u.km/u.s).value]))
        
        # add leading tail info
        tt = Table(np.concatenate((self.leading['x'].to(u.kpc).value, self.leading['v'].to(u.km/u.s).value)).T, names=('x', 'y', 'z', 'vx', 'vy', 'vz'))
        t = astropy.table.vstack([t,tt])
        
        # add trailing tail info
        tt = Table(np.concatenate((self.trailing['x'].to(u.kpc).value, self.trailing['v'].to(u.km/u.s).value)).T, names=('x', 'y', 'z', 'vx', 'vy', 'vz'))
        t = astropy.table.vstack([t,tt])
        
        # save to file
        t.write(fname, format='ascii.commented_header')



MASK = -9999

mw_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_coord': coord.SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg, frame='icrs')}
vsun = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}
vsun0 = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

gc_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 0.1*u.kpc, 'roll': 0*u.deg, 'galcen_coord': coord.SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg, frame='icrs')}
vgc = {'vcirc': 0*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}
vgc0 = {'vcirc': 0*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

pparams_fid = [0.5*u.Msun, 0.7*u.kpc, 6.8*u.Msun, 3*u.kpc, 0.28*u.kpc, 430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 0.*u.pc/u.Myr**2, 0.*u.pc/u.Myr**2, 0.*u.pc/u.Myr**2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0*u.deg, 0*u.deg, 0*u.kpc, 0*u.km/u.s, 0*u.mas/u.yr, 0*u.mas/u.yr]

import streakline
import emcee
import ffwd
from scipy import stats
import time
import pickle

def load_stream(name, obserr=[2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc]):
    """Load a stream and return a Stream object
    Assumes only positions known
    Parameters:
    name - string identifying the stream
    obserr - list with observational uncertainties (optional)"""
    
    t = Table.read('/home/ana/projects/python/galstreams/footprints/galstreams.footprint.ALL.dat', format='ascii.commented_header')
    ind = t['IDst']==name
    t = t[ind]
    if len(t)>100:
        t = t[::10]
    obs = np.array([t['RA_deg'], t['DEC_deg'], t['Rhel_kpc'] + np.random.randn(len(t))*2])
    obsunit = [u.deg, u.deg, u.kpc]
    err = np.ones_like(obs)*np.array([x.value for x in obserr])[:,np.newaxis]

    # store into stream object
    observed = Stream()
    observed.obs = obs
    observed.obsunit = obsunit
    observed.err = err
    observed.obserror = obserr
    
    return observed

def vcirc_potential(r, pparams=pparams_fid):
    """Return circular velocity in a gravitational potential of a disk, bulge and NFW halo, in the disk plane, at a distance r from the galactic center
    Parameters:
    r - distance from center
    pparams - list with potential parameters (optional)"""
    
    # nfw halo
    mr = pparams[5]**2 * pparams[6] / G * (np.log(1 + r/pparams[6]) - r/(r + pparams[6]))
    vch2 = G*mr/r

    # hernquist bulge
    vcb2 = G * pparams[0] * r * (r + pparams[1])**-2

    # miyamoto-nagai disk
    vcd2 = G * pparams[2] * r**2 * (r**2 + (pparams[3] + pparams[4])**2)**-1.5
    
    vcirc = np.sqrt(vch2 + vcb2 + vcd2)
    
    return vcirc

def find_progenitor(name='Sty', test=False, verbose=False):
    """"""
    obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc]
    potential = 'gal'
    pparams = pparams_fid[:]
    dt = 1*u.Myr
    observer = mw_observer
    vobs = vsun
    obsmode = 'equatorial'
    footprint = None
    np.random.seed(58)
    
    # load stream
    observed = load_stream(name, obserr=obserr)
    
    # adjust circular velocity in this halo
    vobs['vcirc'] = vcirc_potential(observer['galcen_distance'], pparams=pparams)
    
    # plot observed stream
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    for i in range(2):
        plt.sca(ax[i])
        plt.plot(observed.obs[0], observed.obs[i+1], 'ko')
    
    plt.tight_layout()
    
    # initialize progenitor properties
    x0_obs, v0_obs = get_progenitor(observed, observer=mw_observer, pparams=pparams)
    plist = [i.value for i in x0_obs] + [i.value for i in v0_obs] + [4, -1, 3]
    pinit = np.array(plist)
    
    if test:
        print(lnprob_prog(pinit, potential, pparams, dt, obsmode, observer, vobs, footprint, observed))
        pbest = pinit
    
    else:
        extension = ''
        dname = '../data/chains/progenitor_{}'.format(name)
        
        # Define a sampler
        mpi = False
        nth = 4
        nwalkers = 30
        nfree = 9
        pool = get_pool(mpi=mpi, threads=nth)
        sampler = emcee.EnsembleSampler(nwalkers, nfree, lnprob_prog, pool=pool, args=[potential, pparams, dt, obsmode, observer, vobs, footprint, observed])
        
        # initialize run
        seeds = [905, 63]
        nstep = 100
        prng = np.random.RandomState(seeds[1])
        genstate = np.random.get_state()
        
        # initialize walkers
        np.random.seed(seeds[0])
        p = (np.random.rand(nfree * nwalkers).reshape((nwalkers, nfree)))
        for i in range(nfree):
            p[:,i] = (p[:,i]-0.5)*1e-2 + pinit[i]
        
        # Sample
        t1 = time.time()
        pos, prob, state = sampler.run_mcmc(p, nstep, rstate0=genstate)
        t2 = time.time()

        # Save chains and likelihoods
        np.savez('{}{}.npz'.format(dname, extension), lnp=sampler.flatlnprobability, chain=sampler.flatchain)
        
        # Save random generator state
        rgstate = {'state': state}
        pickle.dump(rgstate, open('{}.state{:s}'.format(dname, extension), 'wb'))

        if verbose:
            print("Time: ", t2 - t1)
            idmax = np.argmax(sampler.flatlnprobability)
            print("Best fit: ", sampler.flatchain[idmax])
            print("Acceptance fraction: ", np.mean(sampler.acceptance_fraction))
        
        pbest = sampler.flatchain[idmax]

        # Terminate walkers
        if((mpi==False) & (nth>1)):
            sampler.pool.terminate()
        elif(mpi==True):
            sampler.pool.close()
    
    x0 = [pbest[0]*u.deg, pbest[1]*u.deg, pbest[2]*u.kpc]
    v0 = [pbest[3]*u.km/u.s, pbest[4]*u.mas/u.yr, pbest[5]*u.mas/u.yr]
    mi = 10**pbest[6]*u.Msun
    mf = 10**pbest[7]*u.Msun
    age = pbest[8]*u.Gyr
    
    # stream model parameters
    params = {'generate': {'x0': x0, 'v0': v0, 'progenitor': {'coords': 'equatorial', 'observer': observer, 'pm_polar': False}, 'potential': potential, 'pparams': pparams, 'minit': mi, 'mfinal': mf, 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': dt, 'age': age, 'nstars': 100, 'integrator': 'lf'}, 'observe': {'mode': obsmode, 'nstars':-1, 'sequential':True, 'errors': [0.5*u.deg, 0.5*u.deg, 1*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,], 'observer': observer, 'vobs': vobs, 'footprint': footprint, 'rotmatrix': None}}
    
    model = Stream(**params['generate'])
    model.generate()
    model.observe(**params['observe'])
    
    for i in range(2):
        plt.sca(ax[i])
        plt.plot(model.obs[0], model.obs[i+1], 'ro')

def analyze_chains(name='Sty'):
    """"""
    
    extension = ''
    dname = '../data/chains/progenitor_{}'.format(name)
    d = np.load('{}{}.npz'.format(dname, extension))
    chain = d['chain']
    lnp = d['lnp']
    
    nwalkers = 30
    nstep, ndim = np.shape(chain)
    nstep = int(nstep/nwalkers)
    
    nx = 2
    ny = int((ndim+2)/2)
    dx = 15
    dy = dx*nx/ny
    
    plt.close()
    fig, ax = plt.subplots(nx, ny, figsize=(dx, dy))

    for i in range(ndim):
        plt.sca(ax[int(i/ny)][i%ny])
        plt.plot(np.arange(nstep), chain[:,i].reshape(nwalkers,nstep).T, '-');
        plt.xlabel('Step')

    plt.sca(ax[nx-1][ny-1])
    plt.plot(np.arange(nstep), lnp.reshape(nwalkers,nstep).T, '-');
    plt.xlabel('Step')
    plt.ylabel('ln(p)')

    plt.tight_layout()

def bestfit(name='Sty'):
    """"""
    
    extension = ''
    dname = '../data/chains/progenitor_{}'.format(name)
    d = np.load('{}{}.npz'.format(dname, extension))
    chain = d['chain']
    lnp = d['lnp']
    
    idmax = np.argmax(lnp)
    pbest = chain[idmax]
    
    x0 = [pbest[0]*u.deg, pbest[1]*u.deg, pbest[2]*u.kpc]
    v0 = [pbest[3]*u.km/u.s, pbest[4]*u.mas/u.yr, pbest[5]*u.mas/u.yr]
    mi = 10**pbest[6]*u.Msun
    mf = 10**pbest[7]*u.Msun
    age = pbest[8]*u.Gyr
    
    obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc]
    potential = 'gal'
    pparams = pparams_fid[:]
    dt = 0.2*u.Myr
    observer = mw_observer
    vobs = vsun
    obsmode = 'equatorial'
    footprint = None
    #np.random.seed(58)
    
    # adjust circular velocity in this halo
    vobs['vcirc'] = vcirc_potential(observer['galcen_distance'], pparams=pparams)
    
    # stream model parameters
    params = {'generate': {'x0': x0, 'v0': v0, 'progenitor': {'coords': 'equatorial', 'observer': observer, 'pm_polar': False}, 'potential': potential, 'pparams': pparams, 'minit': mi, 'mfinal': mf, 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': dt, 'age': age, 'nstars': 100, 'integrator': 'lf'}, 'observe': {'mode': obsmode, 'nstars':-1, 'sequential':True, 'errors': [0.5*u.deg, 0.5*u.deg, 1*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,3,4,5], 'observer': observer, 'vobs': vobs, 'footprint': footprint, 'rotmatrix': None}}
    
    model = Stream(**params['generate'])
    model.generate()
    model.observe(**params['observe'])
    
    # load stream
    observed = load_stream(name)
    
    # plot observed stream
    plt.close()
    fig, ax = plt.subplots(1,5, figsize=(15,3), sharex=True)
    
    for i in range(2):
        plt.sca(ax[i])
        plt.plot(observed.obs[0], observed.obs[i+1], 'ko')
    
    for i in range(5):
        plt.sca(ax[i])
        plt.plot(model.obs[0], model.obs[i+1], 'ro')
    
    plt.gca().invert_xaxis()

    plt.tight_layout()
    

def lnprob_prog(x, potential, pparams, dt, obsmode, observer, vobs, footprint, observed):
    """"""
    
    lnprior = lnprior_prog(x)
    
    if np.isfinite(lnprior):
        x0 = [x[0]*u.deg, x[1]*u.deg, x[2]*u.kpc]
        v0 = [x[3]*u.km/u.s, x[4]*u.mas/u.yr, x[5]*u.mas/u.yr]
        mi = 10**x[6]*u.Msun
        mf = 10**x[7]*u.Msun
        age = x[8]*u.Gyr
        
        # stream model parameters
        params = {'generate': {'x0': x0, 'v0': v0, 'progenitor': {'coords': 'equatorial', 'observer': observer, 'pm_polar': False}, 'potential': potential, 'pparams': pparams, 'minit': mi, 'mfinal': mf, 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': dt, 'age': age, 'nstars': 100, 'integrator': 'lf'}, 'observe': {'mode': obsmode, 'nstars':-1, 'sequential':True, 'errors': [0.5*u.deg, 0.5*u.deg, 1*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,], 'observer': observer, 'vobs': vobs, 'footprint': footprint, 'rotmatrix': None}}
        
        model = Stream(**params['generate'])
        model.generate()
        model.observe(**params['observe'])
        
        lnp = point_smooth_comparison(observed.obs, model.obs, observed.err, model.err)
        return lnp + lnprior
    
    else:
        return -np.inf

def lnprior_prog(x):
    """"""
    ranges = np.array([[0, 360], [-90, 90], [0,100], [-300, 300], [-5, 5], [-5, 5], [-2,7], [-2,7], [1,6]])
    npar = np.size(x)
    
    outbounds = [(x[i]<ranges[i][0]) | (x[i]>ranges[i][1]) for i in range(npar)]

    if np.any(~np.isfinite(outbounds)):
        return -np.inf
    else:
        return 0

def get_pool(mpi=False, threads=None, verbose=False):
    """ Get a pool object to pass to emcee for parallel processing.
    If mpi is False and threads is None, pool is None.
    Parameters
    ----------
    mpi : bool
    Use MPI or not. If specified, ignores the threads kwarg.
    threads : int (optional)
    If mpi is False and threads is specified, use a Python
    multiprocessing pool with the specified number of threads.
    """
    
    if mpi:
        from emcee.utils import MPIPool
        pool = MPIPool()
    # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        if verbose: print("Running with MPI...")
    elif threads > 1:
        import multiprocessing
        if verbose: print("Running with multiprocessing on %d cores..."%threads)
        pool = multiprocessing.Pool(threads)
    else:
        if verbose: print("Running serial...")
        pool = None
    
    return pool

def stream_scale(stream, verbose=False):
    """"""

    ndim, nobs = np.shape(stream)
    wmod = np.empty(ndim)

    for i in range(ndim):
        measured = stream[i]!=MASK
        nobs_ = np.sum(measured)
        data = stream[i][measured].reshape(nobs_, 1)
        
        wmod[i] = np.std(data) + 1e-10
    
    wmod = wmod[np.newaxis].T
    
    return 1./wmod

def get_mostdense_point(X, Y):
    """Return index of the densest point"""
    
    positions = np.vstack([X, Y])
    values = np.vstack([X, Y])
    kernel = stats.gaussian_kde(values)
    rhomax = np.argmax(kernel(positions))
    
    return rhomax

def get_progenitor(stream, **kwargs):
    """Return a guess for the phase space coordinates of the progenitor"""
    
    #for k in kwargs:
        #print(k, kwargs[k])
    
    pparams = kwargs['pparams']
    observer = kwargs['observer']
    
    # guess position
    # get point with maximal density
    X = stream.obs[0]
    Y = stream.obs[1]
    rhomax = get_mostdense_point(X, Y)
    
    # use this point as a guess for progenitor position
    px = [x*y for x, y in zip(stream.obs[:3,rhomax], stream.obsunit[:3])]
    xeq = coord.SkyCoord(px[0], px[1], px[2], **observer)
    xgal = xeq.transform_to(coord.Galactocentric)
    x0_u = np.array([xgal.x.to(u.kpc).value, xgal.y.to(u.kpc).value, xgal.z.to(u.kpc).value])*u.kpc
    x0 = np.array([xgal.x.to(u.kpc).value, xgal.y.to(u.kpc).value, xgal.z.to(u.kpc).value])
    
    # guess velocity
    # assume circular velocity at apocenter
    r = np.linalg.norm(x0)*u.kpc
    mr = pparams[5]**2 * pparams[6] / G * (np.log(1 + r/pparams[6]) - r/(r + pparams[6]))
    vtot = np.sqrt(G*mr/r)
    
    dp = np.array([-x0[0], -x0[1], (x0[0]**2 + x0[1]**2)/x0[2]])
    progdv = dp/np.linalg.norm(dp) * vtot
    veq = gc.vgal_to_hel(xeq, progdv)
    
    pv = [veq[2].to(u.km/u.s), veq[0].to(u.mas/u.yr), veq[1].to(u.mas/u.yr)]

    return (px, pv)



def point_point_fast(x_obs, x_mod, err_obs, err_mod):
    """Compare two pointlike datasets"""
    
    # setup
    N = np.shape(x_obs)[1]
    k, K = np.shape(x_mod)
    xm = x_mod.T
    log_pdfs = np.zeros(N)
    
    xi_all = np.resize(x_obs, (K, k, N))
    xm_all = np.resize(xm, (N, K, k)).transpose(1,2,0)
    delta_all = xi_all - xm_all
    #print(x_obs)
    #print(x_mod)
    #print(delta_all)
    
    ei_all = np.resize(err_obs, (K, k, N))
    em_all = np.resize(err_mod.T, (N, K, k)).transpose(1,2,0)
    diag_all = ei_all**2 + em_all**2
    
    mask = err_obs>0
    
    sum_aux = 0
    sum_aux2 = 0
    size_aux = 0
    
    # calculate model likelihood for each observed star
    for i in range(N):
        # difference in observables
        delta = delta_all[:,mask[:,i],i]
        
        # covariance matrix
        diag = diag_all[0,mask[:,i],i]
        sigma = np.diag(diag)
        sigma_inv = np.diag(1/diag)
        sign, logdet = np.linalg.slogdet(sigma)
        #print(diag)
        
        # log-likelihood
        aux = delta.dot(sigma_inv.dot(delta.T))
        deltap = np.sum(np.exp(-0.5*np.diag(aux)))
        aux2 = np.exp(-0.5*np.diag(aux))
        #print(deltap)
        deltalnp = np.log(deltap + 1e-100)
        log_pdfs[i] = deltalnp - 0.5*logdet + np.log(1/K * (2*np.pi)**(-np.sum(mask[:,i])/2))

    # remove outliers
    limit = np.median(log_pdfs) - 3*np.std(log_pdfs)
    to_keep = log_pdfs >= limit
    log_pdf = np.sum(log_pdfs[to_keep])
    
    return log_pdf

def point_point_comparison(x_obs, x_mod, err_obs, err_mod):
    """Compare two pointlike datasets"""
    
    # setup
    lnp = 0
    N = np.shape(x_obs)[1]
    k, K = np.shape(x_mod)
    xm = x_mod.T
    
    # calculate model likelihood for each observed star
    for i in range(N):
        # difference in observables
        xi = np.repeat([x_obs[:,i]], K, axis=0)
        delta = xi - xm
        
        # covariance matrix
        diag = err_obs[:,i]**2 + err_mod[:,0]**2
        sigma = np.diag(diag)
        sigma_inv = np.diag(1/diag)
        sign, logdet = np.linalg.slogdet(sigma)
        
        # log-likelihood
        aux = delta.dot(sigma_inv.dot(delta.T))
        deltap = np.sum(np.exp(-0.5*np.diag(aux)))
        #print(np.diag(aux))
        #print(deltap)
        
        deltap += 1e-100
        if deltap>0:
            deltalnp = np.log(deltap)
        else:
            deltalnp = -1e3
        #print(deltalnp)
        
        lnp = lnp + deltalnp - 0.5*logdet + np.log(1/K * (2*np.pi)**(-k/2))
    
    return lnp

def point_smooth_comparison(x_obs, x_mod, err_obs, err_mod, bandwidth=0.1, scale=1.):
    """Compare two distributions of points with associated uncertainties"""
    
    # rescale the coordinates to 0-1 range
    #mx_obs_ = x_obs[:3,:] * scale
    #mx_mod_ = x_mod[:3,:] * scale
    #me_obs = err_obs[:3,:] * scale
    ##me_mod = np.transpose((err_mod[:3,:].T*0 + 1)) * np.array(bandwidth)[np.newaxis].T
    #me_mod = np.ones_like(mx_mod_) * bandwidth
    
    mx_obs_ = x_obs
    mx_mod_ = x_mod
    me_obs = err_obs
    me_mod = err_mod
    
    log_pdf = 0
    nmodes = 1
    i1 = [0, ]
    i2 = [3, ]
    ind = [x_obs[1,:]!=MASK, ]
    
    for i in range(nmodes):
        if np.sum(ind[i])>0:
            if np.any(ind[i]==False):
                x_obs_ = mx_obs_[i1[i]:i2[i]][:,ind[i]]
                e_obs = me_obs[i1[i]:i2[i]][:,ind[i]]
            else:
                x_obs_ = mx_obs_[i1[i]:i2[i]]
                e_obs = me_obs[i1[i]:i2[i]]
            x_mod_ = mx_mod_[i1[i]:i2[i]]
            e_mod = me_mod[i1[i]:i2[i]]
    
            N = np.shape(x_obs_)[1]
            k = np.shape(x_mod_)[0]
            K = np.shape(x_mod_)[1]
        
            # find covariance matrix
            ei_all = np.resize(e_obs, (K, k, N))
            em_all = np.resize(e_mod.T, (N, K, k)).transpose(1,2,0)
            diag_all = ei_all**2 + em_all**2
        
            logdet = np.zeros(N)
            kp = np.zeros(N, dtype=int)
            sigma_inv = np.zeros((N,k,k))
        
            for i in range(N):
                kp[i] = k - np.sum(e_obs[:,i]<0)
                # covariance matrix
                diag = diag_all[0,:,i]
                sigma = np.diag(diag)
                sigma_inv[i] = np.diag(1/diag)
                sign, logdet[i] = np.linalg.slogdet(sigma[:kp[i],:kp[i]])
        
            logdet -= 2*np.log(1./K * (2*np.pi)**(-kp/2.))
        
            dp = ffwd.compare(np.ravel(x_obs_), np.ravel(x_mod_), np.ravel(e_obs), np.ravel(sigma_inv), logdet, N, k, K)
            log_pdf += dp

    
    return log_pdf
