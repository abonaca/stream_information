from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import streakline
#import streakline2
import myutils
import ffwd

from streams import load_stream, vcirc_potential, store_progparams, wrap_angles, progenitor_prior
#import streams

import astropy
import astropy.units as u
from astropy.constants import G
from astropy.table import Table
import astropy.coordinates as coord
import gala.coordinates as gc

import scipy.linalg as la
import scipy.interpolate
import scipy.optimize
import zscale
import itertools

import copy
import pickle

# observers
# defaults taken as in astropy v2.0 icrs
mw_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_coord': coord.SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg, frame='icrs')}
vsun = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}
vsun0 = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

gc_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 0.1*u.kpc, 'roll': 0*u.deg, 'galcen_coord': coord.SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg, frame='icrs')}
vgc = {'vcirc': 0*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}
vgc0 = {'vcirc': 0*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

MASK = -9999
pparams_fid = [np.log10(0.5e10)*u.Msun, 0.7*u.kpc, np.log10(6.8e10)*u.Msun, 3*u.kpc, 0.28*u.kpc, 430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 0.*u.pc/u.Myr**2, 0.*u.pc/u.Myr**2, 0.*u.pc/u.Myr**2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2*u.kpc**-1, 0.*u.Gyr**-2*u.kpc**-1, 0.*u.Gyr**-2*u.kpc**-1, 0.*u.Gyr**-2*u.kpc**-1, 0.*u.Gyr**-2*u.kpc**-1, 0.*u.Gyr**-2*u.kpc**-1, 0.*u.Gyr**-2*u.kpc**-1, 0*u.deg, 0*u.deg, 0*u.kpc, 0*u.km/u.s, 0*u.mas/u.yr, 0*u.mas/u.yr]
#pparams_fid = [0.5e-5*u.Msun, 0.7*u.kpc, 6.8e-5*u.Msun, 3*u.kpc, 0.28*u.kpc, 430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 0.*u.pc/u.Myr**2, 0.*u.pc/u.Myr**2, 0.*u.pc/u.Myr**2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0*u.deg, 0*u.deg, 0*u.kpc, 0*u.km/u.s, 0*u.mas/u.yr, 0*u.mas/u.yr]


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
        elif self.setup['potential']=='octu':
            self.setup_aux['paux'] = 10
            
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
    
    def observe(self, mode='cartesian', wangle=0*u.deg, units=[], errors=[], nstars=-1, sequential=False, present=[], logerr=False, observer={'z_sun': 0.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_ra': 300*u.deg, 'galcen_dec': 20*u.deg}, vobs={'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}, footprint='none', rotmatrix=None):
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
            ra, dec, dist = [xeq.ra.to(units[0]).wrap_at(wangle), xeq.dec.to(units[1]), xeq.distance.to(units[2])]
            vr, mua, mud = [veq[2].to(units[3]), veq[0].to(units[4]), veq[1].to(units[5])]
            
            obs = np.hstack([ra, dec, dist, vr, mua, mud]).value
            obs = np.reshape(obs,(6,-1))
            
            if footprint=='sdss':
                infoot = dec > -2.5*u.deg
                obs = obs[:,infoot]
            
            if np.allclose(rotmatrix, np.eye(3))!=1:
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
        
        # add leading tail infoobsmode
        tt = Table(np.concatenate((self.leading['x'].to(u.kpc).value, self.leading['v'].to(u.km/u.s).value)).T, names=('x', 'y', 'z', 'vx', 'vy', 'vz'))
        t = astropy.table.vstack([t,tt])
        
        # add trailing tail info
        tt = Table(np.concatenate((self.trailing['x'].to(u.kpc).value, self.trailing['v'].to(u.km/u.s).value)).T, names=('x', 'y', 'z', 'vx', 'vy', 'vz'))
        t = astropy.table.vstack([t,tt])
        
        # save to file
        t.write(fname, format='ascii.commented_header')


# make a streakline model of a stream

def stream_model(name='gd1', pparams0=pparams_fid, dt=0.2*u.Myr, rotmatrix=np.eye(3), graph=False, graphsave=False, observer=mw_observer, vobs=vsun, footprint='', obsmode='equatorial'):
    """Create a streakline model of a stream
    baryonic component as in kupper+2015: 3.4e10*u.Msun, 0.7*u.kpc, 1e11*u.Msun, 6.5*u.kpc, 0.26*u.kpc"""
    
    # vary progenitor parameters
    mock = pickle.load(open('../data/mock_{}.params'.format(name), 'rb'))
    for i in range(3):
        mock['x0'][i] += pparams0[26+i]
        mock['v0'][i] += pparams0[29+i]
    
    # vary potential parameters
    potential = 'octu'
    pparams = pparams0[:26]
    #print(pparams[0])
    pparams[0] = (10**pparams0[0].value)*pparams0[0].unit
    pparams[2] = (10**pparams0[2].value)*pparams0[2].unit
    #pparams[0] = pparams0[0]*1e15
    #pparams[2] = pparams0[2]*1e15
    #print(pparams[0])
    
    # adjust circular velocity in this halo
    vobs['vcirc'] = vcirc_potential(observer['galcen_distance'], pparams=pparams)

    # create a model stream with these parameters
    params = {'generate': {'x0': mock['x0'], 'v0': mock['v0'], 'progenitor': {'coords': 'equatorial', 'observer': mock['observer'], 'pm_polar': False}, 'potential': potential, 'pparams': pparams, 'minit': mock['mi'], 'mfinal': mock['mf'], 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': dt, 'age': mock['age'], 'nstars': 400, 'integrator': 'lf'}, 'observe': {'mode': mock['obsmode'], 'wangle': mock['wangle'], 'nstars':-1, 'sequential':True, 'errors': [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,3,4,5], 'observer': mock['observer'], 'vobs': mock['vobs'], 'footprint': mock['footprint'], 'rotmatrix': rotmatrix}}
    
    stream = Stream(**params['generate'])
    stream.generate()
    stream.observe(**params['observe'])
    
    ################################
    # Plot observed stream and model
    
    if graph:
        observed = load_stream(name)
        Ndim = np.shape(observed.obs)[0]
    
        modcol = 'k'
        obscol = 'orange'
        ylabel = ['Dec (deg)', 'Distance (kpc)', 'Radial velocity (km/s)']

        plt.close()
        fig, ax = plt.subplots(1, 3, figsize=(12,4))
        
        for i in range(3):
            plt.sca(ax[i])
            
            plt.gca().invert_xaxis()
            plt.xlabel('R.A. (deg)')
            plt.ylabel(ylabel[i])
            
            plt.plot(observed.obs[0], observed.obs[i+1], 's', color=obscol, mec='none', ms=8, label='Observed stream')
            plt.plot(stream.obs[0], stream.obs[i+1], 'o', color=modcol, mec='none', ms=4, label='Fiducial model')
            
            if i==0:
                plt.legend(frameon=False, handlelength=0.5, fontsize='small')
        
        plt.tight_layout()
        if graphsave:
            plt.savefig('../plots/mock_observables_{}_p{}.png'.format(name, potential), dpi=150)
    
    return stream

def progenitor_params(n):
    """Return progenitor parameters for a given stream"""
    
    if n==-1:
        age = 1.6*u.Gyr
        mi = 1e4*u.Msun
        mf = 2e-1*u.Msun
        x0, v0 = gd1_coordinates(observer=mw_observer)
    elif n==-2:
        age = 2.7*u.Gyr
        mi = 1e5*u.Msun
        mf = 2e4*u.Msun
        x0, v0 = pal5_coordinates(observer=mw_observer, vobs=vsun0)
    elif n==-3:
        age = 3.5*u.Gyr
        mi = 5e4*u.Msun
        mf = 2e-1*u.Msun
        x0, v0 = tri_coordinates(observer=mw_observer)
    elif n==-4:
        age = 2*u.Gyr
        mi = 2e4*u.Msun
        mf = 2e-1*u.Msun
        x0, v0 = atlas_coordinates(observer=mw_observer)
    
    out = {'x0': x0, 'v0': v0, 'age': age, 'mi': mi, 'mf': mf}
    
    return out

def gal2eq(x, v, observer=mw_observer, vobs=vsun0):
    """"""
    # define reference frame
    xgal = coord.Galactocentric(np.array(x)[:,np.newaxis]*u.kpc, **observer)
    
    # convert
    xeq = xgal.transform_to(coord.ICRS)
    veq = gc.vgal_to_hel(xeq, np.array(v)[:,np.newaxis]*u.km/u.s, **vobs)
    
    # store coordinates
    units = [u.deg, u.deg, u.kpc, u.km/u.s, u.mas/u.yr, u.mas/u.yr]
    xobs = [xeq.ra.to(units[0]), xeq.dec.to(units[1]), xeq.distance.to(units[2])]
    vobs = [veq[2].to(units[3]), veq[0].to(units[4]), veq[1].to(units[5])]
    
    return(xobs, vobs)

def gd1_coordinates(observer=mw_observer):
    """Approximate GD-1 progenitor coordinates"""
    
    x = coord.SkyCoord(ra=154.377*u.deg, dec=41.5309*u.deg, distance=8.2*u.kpc, **observer)
    x_ = x.galactocentric
    x0 = [x_.x.value, x_.y.value, x_.z.value]
    v0 = [-90, -250, -120]
    
    return (x0, v0)

def pal5_coordinates(observer=mw_observer, vobs=vsun0):
    """Pal5 coordinates"""
    
    # sdss
    ra = 229.0128*u.deg
    dec = -0.1082*u.deg
    # bob's rrlyrae
    d = 21.7*u.kpc
    # harris
    #d = 23.2*u.kpc
    # odenkirchen 2002
    vr = -58.7*u.km/u.s
    # fritz & kallivayalil 2015
    mua = -2.296*u.mas/u.yr
    mud = -2.257*u.mas/u.yr
    d = 24*u.kpc
    
    x = coord.SkyCoord(ra=ra, dec=dec, distance=d, **observer)
    x0 = x.galactocentric
    v0 = gc.vhel_to_gal(x.icrs, rv=vr, pm=[mua, mud], **vobs).to(u.km/u.s)

    return ([x0.x.value, x0.y.value, x0.z.value], v0.value.tolist())

def tri_coordinates(observer=mw_observer):
    """Approximate Triangulum progenitor coordinates"""
    
    x = coord.SkyCoord(ra=22.38*u.deg, dec=30.26*u.deg, distance=33*u.kpc, **observer)
    x_ = x.galactocentric
    x0 = [x_.x.value, x_.y.value, x_.z.value]
    v0 = [-40, 155, 155]
    
    return (x0, v0)

def atlas_coordinates(observer=mw_observer):
    """Approximate ATLAS progenitor coordinates"""
    
    x = coord.SkyCoord(ra=20*u.deg, dec=-27*u.deg, distance=20*u.kpc, **observer)
    x_ = x.galactocentric
    x0 = [x_.x.value, x_.y.value, x_.z.value]
    v0 = [40, 150, -120]
    
    return (x0, v0)


# great circle orientation

def find_greatcircle(stream=None, name='gd1', pparams=pparams_fid, dt=0.2*u.Myr, save=True, graph=True):
    """Save rotation matrix for a stream model"""
    
    if stream==None:
        stream = stream_model(name, pparams0=pparams, dt=dt)
    
    # find the pole
    ra = np.radians(stream.obs[0])
    dec = np.radians(stream.obs[1])
    
    rx = np.cos(ra) * np.cos(dec)
    ry = np.sin(ra) * np.cos(dec)
    rz = np.sin(dec)
    r = np.column_stack((rx, ry, rz))

    # fit the plane
    x0 = np.array([0, 1, 0])
    lsq = scipy.optimize.minimize(wfit_plane, x0, args=(r,))
    x0 = lsq.x/np.linalg.norm(lsq.x)
    ra0 = np.arctan2(x0[1], x0[0])
    dec0 = np.arcsin(x0[2])
    
    ra0 += np.pi
    dec0 = np.pi/2 - dec0

    # euler rotations
    R0 = myutils.rotmatrix(np.degrees(-ra0), 2)
    R1 = myutils.rotmatrix(np.degrees(dec0), 1)
    R2 = myutils.rotmatrix(0, 2)
    R = np.dot(R2, np.matmul(R1, R0))
    
    xi, eta = myutils.rotate_angles(stream.obs[0], stream.obs[1], R)
    
    # put xi = 50 at the beginning of the stream
    xi[xi>180] -= 360
    xi += 360
    xi0 = np.min(xi) - 50
    R2 = myutils.rotmatrix(-xi0, 2)
    R = np.dot(R2, np.matmul(R1, R0))
    xi, eta = myutils.rotate_angles(stream.obs[0], stream.obs[1], R)
    
    if save:
        np.save('../data/rotmatrix_{}'.format(name), R)
        
        f = open('../data/mock_{}.params'.format(name), 'rb')
        mock = pickle.load(f)
        mock['rotmatrix'] = R
        f.close()
        
        f = open('../data/mock_{}.params'.format(name), 'wb')
        pickle.dump(mock, f)
        f.close()
    
    if graph:
        plt.close()
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        
        plt.sca(ax[0])
        plt.plot(stream.obs[0], stream.obs[1], 'ko')
        
        plt.xlabel('R.A. (deg)')
        plt.ylabel('Dec (deg)')
        
        plt.sca(ax[1])
        plt.plot(xi, eta, 'ko')
        
        plt.xlabel('$\\xi$ (deg)')
        plt.ylabel('$\\eta$ (deg)')
        plt.ylim(-5, 5)
        
        plt.tight_layout()
        plt.savefig('../plots/gc_orientation_{}.png'.format(name))
    
    return R

def wfit_plane(x, r, p=None):
    """Fit a plane to a set of 3d points"""
    
    Np = np.shape(r)[0]
    if np.any(p)==None:
        p = np.ones(Np)
    
    Q = np.zeros((3,3))
    
    for i in range(Np):
        Q += p[i]**2 * np.outer(r[i], r[i])
    
    x = x/np.linalg.norm(x)
    lsq = np.inner(x, np.inner(Q, x))
    
    return lsq


# observed streams

#def load_stream(n):
    #"""Load stream observations"""
    
    #if n==-1:
        #observed = load_gd1(present=[0,1,2,3])
    #elif n==-2:
        #observed = load_pal5(present=[0,1,2,3])
    #elif n==-3:
        #observed = load_tri(present=[0,1,2,3])
    #elif n==-4:
        #observed = load_atlas(present=[0,1,2,3])
    
    #return observed

def endpoints(name):
    """"""
    stream = load_stream(name)
    
    # find endpoints
    amin = np.argmin(stream.obs[0])
    amax = np.argmax(stream.obs[0])
    ra = np.array([stream.obs[0][i] for i in [amin, amax]])
    dec = np.array([stream.obs[1][i] for i in [amin, amax]])
    
    f = open('../data/mock_{}.params'.format(name), 'rb')
    mock = pickle.load(f)
    
    # rotate endpoints
    R = mock['rotmatrix']
    xi, eta  = myutils.rotate_angles(ra, dec, R)
    #xi, eta  = myutils.rotate_angles(stream.obs[0], stream.obs[1], R)
    mock['ra_range'] = ra
    mock['xi_range'] = xi #np.percentile(xi, [10,90])
    f.close()
    
    f = open('../data/mock_{}.params'.format(name), 'wb')
    pickle.dump(mock, f)
    f.close()

def load_pal5(present, nobs=50, potential='gal'):
    """"""
    
    if len(present)==2:
        t = Table.read('../data/pal5_members.txt', format='ascii.commented_header')
        dist = 21.7
        deltadist = 0.7
        np.random.seed(34)
        t = t[np.random.randint(0, high=len(t), size=nobs)]
        nobs = len(t)
        d = np.random.randn(nobs)*deltadist + dist
        
        obs = np.array([t['ra'], t['dec'], d])
        obsunit = [u.deg, u.deg, u.kpc]
        err = np.repeat( np.array([2e-4, 2e-4, 0.7]), nobs ).reshape(3, -1)
        obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc]
    
    if len(present)==3:
        #t = Table.read('../data/pal5_kinematic.txt', format='ascii.commented_header')
        t = Table.read('../data/pal5_allmembers.txt', format='ascii.commented_header')
        obs = np.array([t['ra'], t['dec'], t['d']])
        obsunit = [u.deg, u.deg, u.kpc]
        err = np.array([t['err_ra'], t['err_dec'], t['err_d']])
        obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc]
    
    if len(present)==4:
        #t = Table.read('../data/pal5_kinematic.txt', format='ascii.commented_header')
        t = Table.read('../data/pal5_allmembers.txt', format='ascii.commented_header')
        obs = np.array([t['ra'], t['dec'], t['d'], t['vr']])
        obsunit = [u.deg, u.deg, u.kpc, u.km/u.s]
        err = np.array([t['err_ra'], t['err_dec'], t['err_d'], t['err_vr']])
        obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s]
        
        
    observed = Stream(potential=potential)
    observed.obs = obs
    observed.obsunit = obsunit
    observed.err = err
    observed.obserror = obserr
    
    return observed

def load_gd1(present, nobs=50, potential='gal'):
    """"""
    if len(present)==3:
        t = Table.read('../data/gd1_members.txt', format='ascii.commented_header')
        dist = 0
        deltadist = 0.5
        np.random.seed(34)
        t = t[np.random.randint(0, high=len(t), size=nobs)]
        nobs = len(t)
        d = np.random.randn(nobs)*deltadist + dist
        d += t['l']*0.04836 + 9.86
        
        obs = np.array([t['ra'], t['dec'], d])
        obsunit = [u.deg, u.deg, u.kpc]
        err = np.repeat( np.array([2e-4, 2e-4, 0.5]), nobs ).reshape(3, -1)
        obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc]
    
    if len(present)==4:
        #t = Table.read('../data/gd1_kinematic.txt', format='ascii.commented_header')
        t = Table.read('../data/gd1_allmembers.txt', format='ascii.commented_header')
        obs = np.array([t['ra'], t['dec'], t['d'], t['vr']])
        obsunit = [u.deg, u.deg, u.kpc, u.km/u.s]
        err = np.array([t['err_ra'], t['err_dec'], t['err_d'], t['err_vr']])
        obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s]
        
    ind = np.all(obs!=MASK, axis=0)
    
    observed = Stream(potential=potential)
    observed.obs = obs#[np.array(present)]
    observed.obsunit = obsunit
    observed.err = err#[np.array(present)]
    observed.obserror = obserr
    
    return observed

def load_tri(present, nobs=50, potential='gal'):
    """"""
    
    if len(present)==4:
        t = Table.read('../data/tri_allmembers.txt', format='ascii.commented_header')
        obs = np.array([t['ra'], t['dec'], t['d'], t['vr']])
        obsunit = [u.deg, u.deg, u.kpc, u.km/u.s]
        err = np.array([t['err_ra'], t['err_dec'], t['err_d'], t['err_vr']])
        obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s]
        
    if len(present)==3:
        t = Table.read('../data/tri_allmembers.txt', format='ascii.commented_header')
        obs = np.array([t['ra'], t['dec'], t['d']])
        obsunit = [u.deg, u.deg, u.kpc]
        err = np.array([t['err_ra'], t['err_dec'], t['err_d']])
        obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc]
        
    ind = np.all(obs!=MASK, axis=0)
    
    observed = Stream(potential=potential)
    observed.obs = obs
    observed.obsunit = obsunit
    observed.err = err
    observed.obserror = obserr
    
    return observed

def load_atlas(present, nobs=50, potential='gal'):
    """"""
    ra, dec = atlas_track()
    n = np.size(ra)
    d = np.random.randn(n)*2 + 20
    
    obs = np.array([ra, dec, d])
    obsunit = [u.deg, u.deg, u.kpc]
    err = np.array([np.ones(n)*0.05, np.ones(n)*0.05, np.ones(n)*2])
    obserr = [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s]
    
    observed = Stream(potential=potential)
    observed.obs = obs
    observed.obsunit = obsunit
    observed.err = err
    observed.obserror = obserr
    
    return observed

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

def fancy_name(n):
    """Return nicely formatted stream name"""
    names = {-1: 'GD-1', -2: 'Palomar 5', -3: 'Triangulum', -4: 'ATLAS'}
    
    return names[n]


# model parameters

def get_varied_pars(vary):
    """Return indices and steps for a preset of varied parameters, and a label for varied parameters
    Parameters:
    vary - string setting the parameter combination to be varied, options: 'potential', 'progenitor', 'halo', or a list thereof"""
    
    if type(vary) is not list:
        vary = [vary]
    
    Nt = len(vary)
    vlabel = '_'.join(vary)
    
    pid = []
    dp = []
    
    for v in vary:
        o1, o2 = get_varied_bytype(v)
        pid += o1
        dp += o2
    
    return (pid, dp, vlabel)

def get_varied_bytype(vary):
    """Get varied parameter of a particular type"""
    if vary=='potential':
        pid = [5,6,8,10,11]
        dp = [20*u.km/u.s, 2*u.kpc, 0.05*u.Unit(1), 0.05*u.Unit(1), 0.4e11*u.Msun]
    elif vary=='bary':
        pid = [0,1,2,3,4]
        # gd1
        dp = [1e-1*u.Msun, 0.005*u.kpc, 1e-1*u.Msun, 0.002*u.kpc, 0.002*u.kpc]
        ## atlas & triangulum
        #dp = [0.4e5*u.Msun, 0.0005*u.kpc, 0.5e6*u.Msun, 0.0002*u.kpc, 0.002*u.kpc]
        # pal5
        dp = [1e-2*u.Msun, 0.000005*u.kpc, 1e-2*u.Msun, 0.000002*u.kpc, 0.00002*u.kpc]
        dp = [1e-7*u.Msun, 0.5*u.kpc, 1e-7*u.Msun, 0.5*u.kpc, 0.5*u.kpc]
        dp = [1e-2*u.Msun, 0.5*u.kpc, 1e-2*u.Msun, 0.5*u.kpc, 0.5*u.kpc]
    elif vary=='halo':
        pid = [5,6,8,10]
        dp = [20*u.km/u.s, 2*u.kpc, 0.05*u.Unit(1), 0.05*u.Unit(1)]
        dp = [35*u.km/u.s, 2.9*u.kpc, 0.05*u.Unit(1), 0.05*u.Unit(1)]
    elif vary=='progenitor':
        pid = [26,27,28,29,30,31]
        dp = [1*u.deg, 1*u.deg, 0.5*u.kpc, 20*u.km/u.s, 0.3*u.mas/u.yr, 0.3*u.mas/u.yr]
    elif vary=='dipole':
        pid = [11,12,13]
        #dp = [1e-11*u.Unit(1), 1e-11*u.Unit(1), 1e-11*u.Unit(1)]
        dp = [0.05*u.pc/u.Myr**2, 0.05*u.pc/u.Myr**2, 0.05*u.pc/u.Myr**2]
    elif vary=='quad':
        pid = [14,15,16,17,18]
        dp = [0.5*u.Gyr**-2 for x in range(5)]
    elif vary=='octu':
        pid = [19,20,21,22,23,24,25]
        dp = [0.001*u.Gyr**-2*u.kpc**-1 for x in range(7)]
    else:
        pid = []
        dp = []
    
    return (pid, dp)

def get_parlabel(pid):
    """Return label for a list of parameter ids
    Parameter:
    pid - list of parameter ids"""
    
    master = ['log $M_b$', '$a_b$', 'log $M_d$', '$a_d$', '$b_d$', '$V_h$', '$R_h$', '$\phi$', '$q_x$', '$q_y$', '$q_z$', '$a_{1,-1}$', '$a_{1,0}$', '$a_{1,1}$', '$a_{2,-2}$', '$a_{2,-1}$', '$a_{2,0}$', '$a_{2,1}$', '$a_{2,2}$', '$a_{3,-3}$', '$a_{3,-2}$', '$a_{3,-1}$', '$a_{3,0}$', '$a_{3,1}$', '$a_{3,2}$', '$a_{3,3}$', '$RA_p$', '$Dec_p$', '$d_p$', '$V_{r_p}$', '$\mu_{\\alpha_p}$', '$\mu_{\delta_p}$', ]
    master_units = ['dex', 'kpc', 'dex', 'kpc', 'kpc', 'km/s', 'kpc', 'rad', '', '', '', 'pc/Myr$^2$', 'pc/Myr$^2$', 'pc/Myr$^2$', 'Gyr$^{-2}$', 'Gyr$^{-2}$', 'Gyr$^{-2}$', 'Gyr$^{-2}$', 'Gyr$^{-2}$', 'Gyr$^{-2}$ kpc$^{-1}$', 'Gyr$^{-2}$ kpc$^{-1}$', 'Gyr$^{-2}$ kpc$^{-1}$', 'Gyr$^{-2}$ kpc$^{-1}$', 'Gyr$^{-2}$ kpc$^{-1}$', 'Gyr$^{-2}$ kpc$^{-1}$', 'Gyr$^{-2}$ kpc$^{-1}$', 'deg', 'deg', 'kpc', 'km/s', 'mas/yr', 'mas/yr', ]
    
    if type(pid) is list:
        labels = []
        units = []
        
        for i in pid:
            labels += [master[i]]
            units += [master_units[i]]
    else:
        labels = master[pid]
        units = master_units[pid]
    
    return (labels, units)

def get_steps(Nstep=50, log=False):
    """Return deltax steps in both directions
    Paramerets:
    Nstep - number of steps in one direction (default: 50)
    log - if True, steps are logarithmically spaced (default: False)"""
    
    if log:
        step = np.logspace(-10, 1, Nstep)
    else:
        step = np.linspace(0.1, 10, Nstep)
    
    step = np.concatenate([-step[::-1], step])
    
    return (Nstep, step)

def lmc_position():
    """"""
    ra = 80.8939*u.deg
    dec = -69.7561*u.deg
    dm = 18.48
    d = 10**(1 + dm/5)*u.pc
    
    x = coord.SkyCoord(ra=ra, dec=dec, distance=d)
    xgal = [x.galactocentric.x.si, x.galactocentric.y.si, x.galactocentric.z.si]
    print(xgal)
    
def lmc_properties():
    """"""
    # penarrubia 2016
    mass = 2.5e11*u.Msun
    ra = 80.8939*u.deg
    dec = -69.7561*u.deg
    dm = 18.48
    d = 10**(1 + dm/5)*u.pc

    c1 = coord.SkyCoord(ra=ra, dec=dec, distance=d)
    cgal1 = c1.transform_to(coord.Galactocentric)
    xgal = np.array([cgal1.x.to(u.kpc).value, cgal1.y.to(u.kpc).value, cgal1.z.to(u.kpc).value])*u.kpc
    
    return (mass, xgal)


# fit bspline to a stream model

def fit_bspline(n, pparams=pparams_fid, dt=0.2*u.Myr, align=False, save='', graph=False, graphsave='', fiducial=False):
    """Fit bspline to a stream model and save to file"""
    Ndim = 6
    fits = [None]*(Ndim-1)
    
    if align:
        rotmatrix = np.load('../data/rotmatrix_{}.npy'.format(n))
    else:
        rotmatrix = None
    
    stream = stream_model(n, pparams0=pparams, dt=dt, rotmatrix=rotmatrix)
    
    Nobs = 10
    k = 3
    isort = np.argsort(stream.obs[0])
    ra = np.linspace(np.min(stream.obs[0])*1.05, np.max(stream.obs[0])*0.95, Nobs)
    t = np.r_[(stream.obs[0][isort][0],)*(k+1), ra, (stream.obs[0][isort][-1],)*(k+1)]
    
    for j in range(Ndim-1):
        fits[j] = scipy.interpolate.make_lsq_spline(stream.obs[0][isort], stream.obs[j+1][isort], t, k=k)
    
    if len(save)>0:
        np.savez('../data/{:s}'.format(save), fits=fits)
    
    if graph:
        xlims, ylims = get_stream_limits(n, align)
        ylabel = ['R.A. (deg)', 'Dec (deg)', 'd (kpc)', '$V_r$ (km/s)', '$\mu_\\alpha$ (mas/yr)', '$\mu_\delta$ (mas/yr)']
        if align:
            ylabel[:2] = ['$\\xi$ (deg)', '$\\eta$ (deg)']
        
        if fiducial:
            stream_fid = stream_model(n, pparams0=pparams_fid, dt=dt, rotmatrix=rotmatrix)
            fidsort = np.argsort(stream_fid.obs[0])
            ra = np.linspace(np.min(stream_fid.obs[0])*1.05, np.max(stream_fid.obs[0])*0.95, Nobs)
            tfid = np.r_[(stream_fid.obs[0][fidsort][0],)*(k+1), ra, (stream_fid.obs[0][fidsort][-1],)*(k+1)]
            llabel = 'b-spline fit'
        else:
            llabel = ''
        
        plt.close()
        fig, ax = plt.subplots(2,5,figsize=(20,5), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
        
        for i in range(Ndim-1):
            plt.sca(ax[0][i])
            plt.plot(stream.obs[0], stream.obs[i+1], 'ko')
            plt.plot(stream.obs[0][isort], fits[i](stream.obs[0][isort]), 'r-', lw=2, label=llabel)
            
            if fiducial:
                fits_fid = scipy.interpolate.make_lsq_spline(stream_fid.obs[0][fidsort], stream_fid.obs[i+1][fidsort], tfid, k=k)
                plt.plot(stream_fid.obs[0], stream_fid.obs[i+1], 'wo', mec='k', alpha=0.1)
                plt.plot(stream_fid.obs[0][fidsort], fits_fid(stream_fid.obs[0][fidsort]), 'b-', lw=2, label='Fiducial')
            
            plt.ylabel(ylabel[i+1])
            plt.xlim(xlims[0], xlims[1])
            plt.ylim(ylims[i][0], ylims[i][1])
            
            plt.sca(ax[1][i])
            if fiducial:
                yref = fits_fid(stream.obs[0])
                ycolor = 'b'
            else:
                yref = fits[i](stream.obs[0])
                ycolor = 'r'
            plt.axhline(0, color=ycolor, lw=2)

            if fiducial: plt.plot(stream.obs[0][isort], stream.obs[i+1][isort] - stream_fid.obs[i+1][fidsort], 'wo', mec='k', alpha=0.1)
            plt.plot(stream.obs[0], stream.obs[i+1] - yref, 'ko')
            
            if fiducial:
                fits_diff = scipy.interpolate.make_lsq_spline(stream.obs[0][isort], stream.obs[i+1][isort] - stream_fid.obs[i+1][fidsort], t, k=k)
                plt.plot(stream.obs[0][isort], fits_diff(stream.obs[0][isort]), 'r--')
            plt.plot(stream.obs[0][isort], fits[i](stream.obs[0][isort]) - yref[isort], 'r-', lw=2, label=llabel)
            
            plt.xlabel(ylabel[0])
            plt.ylabel('$\Delta$ {}'.format(ylabel[i+1].split(' ')[0]))
        
        if fiducial:
            plt.sca(ax[0][Ndim-2])
            plt.legend(fontsize='small')
        
        plt.tight_layout()
        if len(graphsave)>0:
            plt.savefig('../plots/{:s}.png'.format(graphsave))
    
def fitbyt_bspline(n, pparams=pparams_fid, dt=0.2*u.Myr, align=False, save='', graph=False, graphsave='', fiducial=False):
    """Fit each tail individually"""
    Ndim = 6
    fits = [None]*(Ndim-1)
    
    if align:
        rotmatrix = np.load('../data/rotmatrix_{}.npy'.format(n))
    else:
        rotmatrix = None
    
    stream = stream_model(n, pparams0=pparams, dt=dt, rotmatrix=rotmatrix)
    
    Nobs = 10
    k = 3
    isort = np.argsort(stream.obs[0])
    ra = np.linspace(np.min(stream.obs[0])*1.05, np.max(stream.obs[0])*0.95, Nobs)
    t = np.r_[(stream.obs[0][isort][0],)*(k+1), ra, (stream.obs[0][isort][-1],)*(k+1)]
    
    for j in range(Ndim-1):
        fits[j] = scipy.interpolate.make_lsq_spline(stream.obs[0][isort], stream.obs[j+1][isort], t, k=k)
    
    if len(save)>0:
        np.savez('../data/{:s}'.format(save), fits=fits)
    
    if graph:
        xlims, ylims = get_stream_limits(n, align)
        ylabel = ['R.A. (deg)', 'Dec (deg)', 'd (kpc)', '$V_r$ (km/s)', '$\mu_\\alpha$ (mas/yr)', '$\mu_\delta$ (mas/yr)']
        if align:
            ylabel[:2] = ['$\\xi$ (deg)', '$\\eta$ (deg)']
        
        if fiducial:
            stream_fid = stream_model(n, pparams0=pparams_fid, dt=dt, rotmatrix=rotmatrix)
        
        plt.close()
        fig, ax = plt.subplots(2,Ndim,figsize=(20,4), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
        
        for i in range(Ndim):
            plt.sca(ax[0][i])
            Nhalf = int(0.5*np.size(stream.obs[i]))
            plt.plot(stream.obs[i][:Nhalf], 'o')
            plt.plot(stream.obs[i][Nhalf:], 'o')

            if fiducial:
                plt.plot(stream_fid.obs[i][:Nhalf], 'wo', mec='k', mew=0.2, alpha=0.5)
                plt.plot(stream_fid.obs[i][Nhalf:], 'wo', mec='k', mew=0.2, alpha=0.5)

            plt.ylabel(ylabel[i])
            
            plt.sca(ax[1][i])
            if fiducial:
                plt.plot(stream.obs[i][:Nhalf] - stream_fid.obs[i][:Nhalf], 'o')
                plt.plot(stream.obs[i][Nhalf:] - stream_fid.obs[i][Nhalf:], 'o')
        
        if fiducial:
            plt.sca(ax[0][Ndim-1])
            plt.legend(fontsize='small')
        
        plt.tight_layout()
        if len(graphsave)>0:
            plt.savefig('../plots/{:s}.png'.format(graphsave))
        else:
            return fig

def get_stream_limits(n, align=False):
    """Return lists with limiting values in different dimensions"""
    if n==-1:
        xlims = [260, 100]
        ylims = [[-20, 70], [5, 15], [-400, 400], [-15,5], [-15, 5]]
    elif n==-2:
        xlims = [250, 210]
        ylims = [[-20, 15], [17, 27], [-80, -20], [-5,0], [-5, 0]]
    elif n==-3:
        xlims = [27, 17]
        ylims = [[10, 50], [34, 36], [-175, -50], [0.45, 1], [0.1, 0.7]]
    elif n==-4:
        xlims = [35, 10]
        ylims = [[-40, -20], [15, 25], [50, 200], [-0.5,0.5], [-1.5, -0.5]]
    
    if align:
        ylims[0] = [-5, 5]
        xup = [110, 110, 80, 80]
        xlims = [xup[np.abs(n)-1], 40]

    return (xlims, ylims)


# step sizes for derivatives

def iterate_steps(n):
    """Calculate derivatives for different parameter classes, and plot"""
    
    for vary in ['bary', 'halo', 'progenitor']:
        print(n, vary)
        step_convergence(n, Nstep=10, vary=vary)
        choose_step(n, Nstep=10, vary=vary)

def iterate_plotsteps(n):
    """Plot stream models for a variety of model parameters"""
    
    for vary in ['bary', 'halo', 'progenitor']:
        print(n, vary)
        pid, dp, vlabel = get_varied_pars(vary)
        for p in range(len(pid)):
            plot_steps(n, p=p, Nstep=5, vary=vary, log=False)

def plot_steps(n, p=0, Nstep=20, log=True, dt=0.2*u.Myr, vary='halo', verbose=False, align=True, observer=mw_observer, vobs=vsun):
    """Plot stream for different values of a potential parameter"""
    
    if align:
        rotmatrix = np.load('../data/rotmatrix_{}.npy'.format(n))
    else:
        rotmatrix = None
    
    pparams0 = pparams_fid
    pid, dp, vlabel = get_varied_pars(vary)
    plabel, punit = get_parlabel(pid[p])

    Nstep, step = get_steps(Nstep=Nstep, log=log)
    
    plt.close()
    fig, ax = plt.subplots(5,5,figsize=(20,10), sharex=True, gridspec_kw = {'height_ratios':[3, 1, 1, 1, 1]})
    
    # fiducial model
    stream0 = stream_model(n, pparams0=pparams0, dt=dt, rotmatrix=rotmatrix, observer=observer, vobs=vobs)
    
    Nobs = 10
    k = 3
    isort = np.argsort(stream0.obs[0])
    ra = np.linspace(np.min(stream0.obs[0])*1.05, np.max(stream0.obs[0])*0.95, Nobs)
    t = np.r_[(stream0.obs[0][isort][0],)*(k+1), ra, (stream0.obs[0][isort][-1],)*(k+1)]
    fits = [None]*5
    
    for j in range(5):
        fits[j] = scipy.interpolate.make_lsq_spline(stream0.obs[0][isort], stream0.obs[j+1][isort], t, k=k)
    
    # excursions
    stream_fits = [[None] * 5 for x in range(2 * Nstep)]
    
    for i, s in enumerate(step[:]):
        pparams = [x for x in pparams0]
        pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
        stream = stream_model(n, pparams0=pparams, dt=dt, rotmatrix=rotmatrix)
        color = mpl.cm.RdBu(i/(2*Nstep-1))
        #print(i, dp[p], pparams)
        
        # fits
        iexsort = np.argsort(stream.obs[0])
        raex = np.linspace(np.percentile(stream.obs[0], 10), np.percentile(stream.obs[0], 90), Nobs)
        tex = np.r_[(stream.obs[0][iexsort][0],)*(k+1), raex, (stream.obs[0][iexsort][-1],)*(k+1)]
        fits_ex = [None]*5
        
        for j in range(5):
            fits_ex[j] = scipy.interpolate.make_lsq_spline(stream.obs[0][iexsort], stream.obs[j+1][iexsort], tex, k=k)
            stream_fits[i][j] = fits_ex[j]
            
            plt.sca(ax[0][j])
            plt.plot(stream.obs[0], stream.obs[j+1], 'o', color=color, ms=2)
            
            plt.sca(ax[1][j])
            plt.plot(stream.obs[0], stream.obs[j+1] - fits[j](stream.obs[0]), 'o', color=color, ms=2)
            
            plt.sca(ax[2][j])
            plt.plot(stream.obs[0], fits_ex[j](stream.obs[0]) - fits[j](stream.obs[0]), 'o', color=color, ms=2)
            
            plt.sca(ax[3][j])
            plt.plot(stream.obs[0], (fits_ex[j](stream.obs[0]) - fits[j](stream.obs[0]))/(s*dp[p]), 'o', color=color, ms=2)
    
    # symmetric derivatives
    ra_der = np.linspace(np.min(stream0.obs[0])*1.05, np.max(stream0.obs[0])*0.95, 100)
    for i in range(Nstep):
        color = mpl.cm.Greys_r(i/Nstep)
        for j in range(5):
            dy = stream_fits[i][j](ra_der) - stream_fits[-i-1][j](ra_der)
            dydx = -dy / np.abs(2*step[i]*dp[p])
            
            plt.sca(ax[4][j])
            plt.plot(ra_der, dydx, '-', color=color, lw=2, zorder=Nstep-i)
    
    # labels, limits
    xlims, ylims = get_stream_limits(n, align)
    ylabel = ['R.A. (deg)', 'Dec (deg)', 'd (kpc)', '$V_r$ (km/s)', '$\mu_\\alpha$ (mas/yr)', '$\mu_\delta$ (mas/yr)']
    if align:
        ylabel[:2] = ['$\\xi$ (deg)', '$\\eta$ (deg)']
    
    for j in range(5):
        plt.sca(ax[0][j])
        plt.ylabel(ylabel[j+1])
        plt.xlim(xlims[0], xlims[1])
        plt.ylim(ylims[j][0], ylims[j][1])
        
        plt.sca(ax[1][j])
        plt.ylabel('$\Delta$ {}'.format(ylabel[j+1].split(' ')[0]))
        
        plt.sca(ax[2][j])
        plt.ylabel('$\Delta$ {}'.format(ylabel[j+1].split(' ')[0]))
        
        plt.sca(ax[3][j])
        plt.ylabel('$\Delta${}/$\Delta${}'.format(ylabel[j+1].split(' ')[0], plabel))
        
        plt.sca(ax[4][j])
        plt.xlabel(ylabel[0])
        plt.ylabel('$\langle$$\Delta${}/$\Delta${}$\\rangle$'.format(ylabel[j+1].split(' ')[0], plabel))
    
    #plt.suptitle('Varying {}'.format(plabel), fontsize='small')
    plt.tight_layout()
    plt.savefig('../plots/observable_steps_{:d}_{:s}_p{:d}_Ns{:d}.png'.format(n, vlabel, p, Nstep))

def step_convergence(name='gd1', Nstep=20, log=True, layer=1, dt=0.2*u.Myr, vary='halo', align=True, graph=False, verbose=False, Nobs=10, k=3, ra_der=np.nan, Nra=50):
    """Check deviations in numerical derivatives for consecutive step sizes"""
    
    mock = pickle.load(open('../data/mock_{}.params'.format(name),'rb'))
    if align:
        rotmatrix = mock['rotmatrix']
        xmm = mock['xi_range']
    else:
        rotmatrix = np.eye(3)
        xmm = mock['ra_range']
    
    # fiducial model
    pparams0 = pparams_fid
    stream0 = stream_model(name=name, pparams0=pparams0, dt=dt, rotmatrix=rotmatrix)

    if np.any(~np.isfinite(ra_der)):
        ra_der = np.linspace(xmm[0]*1.05, xmm[1]*0.95, Nra)
    Nra = np.size(ra_der)
    
    # parameters to vary
    pid, dp, vlabel = get_varied_pars(vary)
    Np = len(pid)
    dpvec = np.array([x.value for x in dp])

    Nstep, step = get_steps(Nstep=Nstep, log=log)
    dydx_all = np.empty((Np, Nstep, 5, Nra))
    dev_der = np.empty((Np, Nstep-2*layer))
    step_der = np.empty((Np, Nstep-2*layer))
    
    for p in range(Np):
        plabel = get_parlabel(pid[p])
        if verbose: print(p, plabel)
        
        # excursions
        stream_fits = [[None] * 5 for x in range(2 * Nstep)]
        
        for i, s in enumerate(step[:]):
            if verbose: print(i, s)
            pparams = [x for x in pparams0]
            pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
            stream = stream_model(name=name, pparams0=pparams, dt=dt, rotmatrix=rotmatrix)
            
            # fits
            iexsort = np.argsort(stream.obs[0])
            raex = np.linspace(np.percentile(stream.obs[0], 10), np.percentile(stream.obs[0], 90), Nobs)
            tex = np.r_[(stream.obs[0][iexsort][0],)*(k+1), raex, (stream.obs[0][iexsort][-1],)*(k+1)]
            fits_ex = [None]*5
            
            for j in range(5):
                fits_ex[j] = scipy.interpolate.make_lsq_spline(stream.obs[0][iexsort], stream.obs[j+1][iexsort], tex, k=k)
                stream_fits[i][j] = fits_ex[j]
        
        # symmetric derivatives
        dydx = np.empty((Nstep, 5, Nra))
        
        for i in range(Nstep):
            color = mpl.cm.Greys_r(i/Nstep)
            for j in range(5):
                dy = stream_fits[i][j](ra_der) - stream_fits[-i-1][j](ra_der)
                dydx[i][j] = -dy / np.abs(2*step[i]*dp[p])
        
        dydx_all[p] = dydx
        
        # deviations from adjacent steps
        step_der[p] = -step[layer:Nstep-layer] * dp[p]
        
        for i in range(layer, Nstep-layer):
            dev_der[p][i-layer] = 0
            for j in range(5):
                for l in range(layer):
                    dev_der[p][i-layer] += np.sum((dydx[i][j] - dydx[i-l-1][j])**2)
                    dev_der[p][i-layer] += np.sum((dydx[i][j] - dydx[i+l+1][j])**2)
    
    np.savez('../data/step_convergence_{}_{}_Ns{}_log{}_l{}'.format(name, vlabel, Nstep, log, layer), step=step_der, dev=dev_der, ders=dydx_all, steps_all=np.outer(dpvec,step[Nstep:]))
    
    if graph:
        plt.close()
        fig, ax = plt.subplots(1,Np,figsize=(4*Np,4))
        
        for p in range(Np):
            plt.sca(ax[p])
            plt.plot(step_der[p], dev_der[p], 'ko')
            
            #plabel = get_parlabel(pid[p])
            #plt.xlabel('$\Delta$ {}'.format(plabel))
            plt.ylabel('D')
            plt.gca().set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('../plots/step_convergence_{}_{}_Ns{}_log{}_l{}.png'.format(name, vlabel, Nstep, log, layer))

def choose_step(name='gd1', tolerance=2, Nstep=20, log=True, layer=1, vary='halo'):
    """"""
    
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
    
    # plot setup
    da = 4
    nrow = 2
    ncol = Np
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(da*ncol, da*1.3), squeeze=False, sharex='col', gridspec_kw = {'height_ratios':[1.2, 3]})
    
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
        
        plt.sca(ax[0][p])
        for i in range(5):
            for j in range(10):
                plt.plot(steps_all[p], np.tanh(dydx[p,:,i,np.int64(j*Nra/10)]), '-', color='{}'.format(i/5), lw=0.5, alpha=0.5)

        plt.axvline(opt_step, ls='-', color='r', lw=2)
        plt.ylim(-1,1)
        
        plt.ylabel('Derivative')
        plt.title('{}'.format(plabels[p])+'$_{best}$ = '+'{:2.2g}'.format(opt_step), fontsize='small')
        
        plt.sca(ax[1][p])
        plt.plot(step[p], dev[p], 'ko')
        
        plt.axvline(opt_step, ls='-', color='r', lw=2)
        plt.plot(step[p][opt_id], dev[p][opt_id], 'ro')
        
        plt.axhline(dtol, ls='-', color='orange', lw=1)
        y0, y1 = plt.gca().get_ylim()
        plt.axhspan(y0, dtol, color='orange', alpha=0.3, zorder=0)
        
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.xlabel('$\Delta$ {} {}'.format(plabels[p], punits[p]))
        plt.ylabel('Derivative deviation')
    
    np.save('../data/optimal_step_{}_{}'.format(name, vlabel), best)

    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/step_convergence_{}_{}_Ns{}_log{}_l{}.png'.format(name, vlabel, Nstep, log, layer))

def read_optimal_step(name, vary, equal=False):
    """Return optimal steps for a range of parameter types"""
    
    if type(vary) is not list:
        vary = [vary]
    
    dp = np.empty(0)
    
    for v in vary:
        dp_opt = np.load('../data/optimal_step_{}_{}.npy'.format(name, v))
        dp = np.concatenate([dp, dp_opt])
    
    if equal:
        dp = np.array([0.05, 0.05, 0.2, 1, 0.01, 0.01, 0.05, 0.1, 0.05, 0.1, 0.1, 10, 1, 0.01, 0.01])
    
    return dp

def visualize_optimal_steps(name='gd1', vary=['progenitor', 'bary', 'halo'], align=True, dt=0.2*u.Myr, Nobs=50, k=3):
    """"""
    mock = pickle.load(open('../data/mock_{}.params'.format(name),'rb'))
    if align:
        rotmatrix = mock['rotmatrix']
        xmm = mock['xi_range']
    else:
        rotmatrix = np.eye(3)
        xmm = mock['ra_range']
    
    # varied parameters
    pparams0 = pparams_fid
    pid, dp_fid, vlabel = get_varied_pars(vary)
    Np = len(pid)
    dp_opt = read_optimal_step(name, vary)
    dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
    
    fiducial = stream_model(name=name, pparams0=pparams0, dt=dt, rotmatrix=rotmatrix)
    iexsort = np.argsort(fiducial.obs[0])
    raex = np.linspace(np.percentile(fiducial.obs[0], 10), np.percentile(fiducial.obs[0], 90), Nobs)
    tex = np.r_[(fiducial.obs[0][iexsort][0],)*(k+1), raex, (fiducial.obs[0][iexsort][-1],)*(k+1)]
    fit = scipy.interpolate.make_lsq_spline(fiducial.obs[0][iexsort], fiducial.obs[1][iexsort], tex, k=k)
    
    nrow = 2
    ncol = np.int64((Np+1)/nrow)
    da = 4
    c = ['b', 'b', 'b', 'r', 'r', 'r']
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*da, nrow*da), squeeze=False)
    
    for p in range(Np):
        plt.sca(ax[p%2][int(p/2)])
        for i, s in enumerate([-1.1, -1, -0.9, 0.9, 1, 1.1]):
            pparams = [x for x in pparams0]
            pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
            stream = stream_model(name=name, pparams0=pparams, dt=dt, rotmatrix=rotmatrix)
            
            # bspline fits to stream centerline
            iexsort = np.argsort(stream.obs[0])
            raex = np.linspace(np.percentile(stream.obs[0], 10), np.percentile(stream.obs[0], 90), Nobs)
            tex = np.r_[(stream.obs[0][iexsort][0],)*(k+1), raex, (stream.obs[0][iexsort][-1],)*(k+1)]
            fitex = scipy.interpolate.make_lsq_spline(stream.obs[0][iexsort], stream.obs[1][iexsort], tex, k=k)
            
            plt.plot(raex, fitex(raex) - fit(raex), '-', color=c[i])
        
        plt.xlabel('R.A. (deg)')
        plt.ylabel('Dec (deg)')
        #print(get_parlabel(p))
        plt.title('$\Delta$ {} = {:.2g}'.format(get_parlabel(p)[0], dp[p]), fontsize='medium')
    
    plt.tight_layout()
    plt.savefig('../plots/{}_optimal_steps.png'.format(name), dpi=200)

# observing modes
def define_obsmodes():
    """Output a pickled dictionary with typical uncertainties and dimensionality of data for a number of observing modes"""
    
    obsmodes = {}
    
    obsmodes['fiducial'] = {'sig_obs': np.array([0.1, 2, 5, 0.1, 0.1]), 'Ndim': [3,4,6]}
    obsmodes['binospec'] = {'sig_obs': np.array([0.1, 2, 10, 0.1, 0.1]), 'Ndim': [3,4,6]}
    obsmodes['hectochelle'] = {'sig_obs': np.array([0.1, 2, 1, 0.1, 0.1]), 'Ndim': [3,4,6]}
    obsmodes['desi'] = {'sig_obs': np.array([0.1, 2, 10, np.nan, np.nan]), 'Ndim': [4,]}
    obsmodes['gaia'] = {'sig_obs': np.array([0.1, 0.2, 10, 0.2, 0.2]), 'Ndim': [6,]}
    obsmodes['exgal'] = {'sig_obs': np.array([0.5, np.nan, 20, np.nan, np.nan]), 'Ndim': [3,]}
    
    pickle.dump(obsmodes, open('../data/observing_modes.info','wb'))

def obsmode_name(mode):
    """Return full name of the observing mode"""
    if type(mode) is not list:
        mode = [mode]
    
    full_names = {'fiducial': 'Fiducial',
                  'binospec': 'Binospec',
                  'hectochelle': 'Hectochelle',
                  'desi': 'DESI-like',
                  'gaia': 'Gaia-like',
                  'exgal': 'Extragalactic'}
    keys = full_names.keys()
    
    names = []
    for m in mode:
        if m in keys:
            name = full_names[m]
        else:
            name = m
        names += [name]
    
    return names

# crbs using bspline

def calculate_crb(name='gd1', dt=0.2*u.Myr, vary=['progenitor', 'bary', 'halo'], ra=np.nan, dd=0.5, Nmin=15, verbose=False, align=True, scale=False, errmode='fiducial', k=3):
    """"""
    mock = pickle.load(open('../data/mock_{}.params'.format(name),'rb'))
    if align:
        rotmatrix = mock['rotmatrix']
        xmm = np.sort(mock['xi_range'])
    else:
        rotmatrix = np.eye(3)
        xmm = np.sort(mock['ra_range'])
        
    # typical uncertainties and data availability
    obsmodes = pickle.load(open('../data/observing_modes.info', 'rb'))
    if errmode not in obsmodes.keys():
        errmode = 'fiducial'
    sig_obs = obsmodes[errmode]['sig_obs']
    data_dim = obsmodes[errmode]['Ndim']
    
    # mock observations
    if np.any(~np.isfinite(ra)):
        if (np.int64((xmm[1]-xmm[0])/dd + 1) < Nmin):
            dd = (xmm[1]-xmm[0])/Nmin
        ra = np.arange(xmm[0], xmm[1]+dd, dd)
        #ra = np.linspace(xmm[0]*1.05, xmm[1]*0.95, Nobs)
    #else:
    Nobs = np.size(ra)
    print(name, Nobs)
    err = np.tile(sig_obs, Nobs).reshape(Nobs,-1)

    # varied parameters
    pparams0 = pparams_fid
    pid, dp_fid, vlabel = get_varied_pars(vary)
    Np = len(pid)
    dp_opt = read_optimal_step(name, vary)
    dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
    fits_ex = [[[None]*5 for x in range(2)] for y in range(Np)]
    
    if scale:
        dp_unit = unity_scale(dp)
        dps = [x*y for x,y in zip(dp, dp_unit)]

    # calculate derivatives for all parameters
    for p in range(Np):
        for i, s in enumerate([-1, 1]):
            pparams = [x for x in pparams0]
            pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
            stream = stream_model(name=name, pparams0=pparams, dt=dt, rotmatrix=rotmatrix)
            
            # bspline fits to stream centerline
            iexsort = np.argsort(stream.obs[0])
            raex = np.linspace(np.percentile(stream.obs[0], 10), np.percentile(stream.obs[0], 90), Nobs)
            tex = np.r_[(stream.obs[0][iexsort][0],)*(k+1), raex, (stream.obs[0][iexsort][-1],)*(k+1)]
            
            for j in range(5):
                fits_ex[p][i][j] = scipy.interpolate.make_lsq_spline(stream.obs[0][iexsort], stream.obs[j+1][iexsort], tex, k=k)
    
    # populate matrix of derivatives and calculate CRB
    for Ndim in data_dim:
    #for Ndim in [6,]:
        Ndata = Nobs * (Ndim - 1)
        cyd = np.empty(Ndata)
        dydx = np.empty((Np, Ndata))
        dy2 = np.empty((2, Np, Ndata))
        
        for j in range(1, Ndim):
            for p in range(Np):
                dy = fits_ex[p][0][j-1](ra) - fits_ex[p][1][j-1](ra)
                dy2[0][p][(j-1)*Nobs:j*Nobs] = fits_ex[p][0][j-1](ra)
                dy2[1][p][(j-1)*Nobs:j*Nobs] = fits_ex[p][1][j-1](ra)
                #positive = np.abs(dy)>0
                #if verbose: print('{:d},{:d} {:s} min{:.1e} max{:1e} med{:.1e}'.format(j, p, get_parlabel(pid[p])[0], np.min(np.abs(dy[positive])), np.max(np.abs(dy)), np.median(np.abs(dy))))
                if scale:
                    dydx[p][(j-1)*Nobs:j*Nobs] = -dy / np.abs(2*dps[p].value)
                else:
                    dydx[p][(j-1)*Nobs:j*Nobs] = -dy / np.abs(2*dp[p].value)
                #if verbose: print('{:d},{:d} {:s} min{:.1e} max{:1e} med{:.1e}'.format(j, p, get_parlabel(pid[p])[0], np.min(np.abs(dydx[p][(j-1)*Nobs:j*Nobs][positive])), np.max(np.abs(dydx[p][(j-1)*Nobs:j*Nobs])), np.median(np.abs(dydx[p][(j-1)*Nobs:j*Nobs]))))
                #print(j, p, get_parlabel(pid[p])[0], dp[p], np.min(np.abs(dy)), np.max(np.abs(dy)), np.median(dydx[p][(j-1)*Nobs:j*Nobs]))
        
            cyd[(j-1)*Nobs:j*Nobs] = err[:,j-1]**2
        
        np.savez('../data/crb/components_{:s}{:1d}_{:s}_a{:1d}_{:s}'.format(errmode, Ndim, name, align, vlabel), dydx=dydx, y=dy2, cyd=cyd, dp=dp_opt)
        
        # data component of the Fisher matrix
        cy = np.diag(cyd)
        cyi = np.diag(1. / cyd)
        caux = np.matmul(cyi, dydx.T)
        dxi = np.matmul(dydx, caux)
        
        # component based on prior knowledge of model parameters
        pxi = priors(name, vary)
        
        # full Fisher matrix
        cxi = dxi + pxi
        
        if verbose:
            cx = np.linalg.inv(cxi)
            cx = np.matmul(np.linalg.inv(np.matmul(cx, cxi)), cx) # iteration to improve inverse at large cond numbers
            sx = np.sqrt(np.diag(cx))
            print('CRB', sx)
            print('condition {:g}'.format(np.linalg.cond(cxi)))
            print('standard inverse', np.allclose(cxi, cxi.T), np.allclose(cx, cx.T), np.allclose(np.matmul(cx,cxi), np.eye(np.shape(cx)[0])))
            
            cx = stable_inverse(cxi)
            print('stable inverse', np.allclose(cxi, cxi.T), np.allclose(cx, cx.T), np.allclose(np.matmul(cx,cxi), np.eye(np.shape(cx)[0])))

        np.savez('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}'.format(errmode, Ndim, name, align, vlabel), cxi=cxi, dxi=dxi, pxi=pxi)

def priors(name, vary):
    """Return covariance matrix with prior knowledge about parameters"""
    
    mock = pickle.load(open('../data/mock_{}.params'.format(name), 'rb'))
    
    cprog = mock['prog_prior']
    cbary = np.array([0.1*x.value for x in pparams_fid[:5]])**-2
    chalo = np.zeros(4)
    cdipole = np.zeros(3)
    cquad = np.zeros(5)
    coctu = np.zeros(7)
    
    priors = {'progenitor': cprog, 'bary': cbary, 'halo': chalo, 'dipole': cdipole, 'quad': cquad, 'octu': coctu}
    cprior = np.empty(0)
    for v in vary:
        cprior = np.concatenate([cprior, priors[v]])
    
    pxi = np.diag(cprior)
    
    return pxi

def scale2invert(name='gd1', Ndim=6, vary=['progenitor', 'bary', 'halo'], verbose=False, align=True, errmode='fiducial'):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    #dp = read_optimal_step(name, vary)
    
    d = np.load('../data/crb/components_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
    dydx = d['dydx']
    cyd = d['cyd']
    y = d['y']
    dp = d['dp']
    
    dy = (y[1,:,:] - y[0,:,:])
    dydx = (y[1,:,:] - y[0,:,:]) / (2*dp[:,np.newaxis])
    
    scaling_par = np.median(np.abs(dydx), axis=1)
    dydx = dydx / scaling_par[:,np.newaxis]
    
    dydx_ = np.reshape(dydx, (len(dp), Ndim-1, -1))
    scaling_dim = np.median(np.abs(dydx_), axis=(2,0))
    dydx_ = dydx_ / scaling_dim[np.newaxis,:,np.newaxis]
    
    cyd_ = np.reshape(cyd, (Ndim-1, -1))
    cyd_ = cyd_ / scaling_dim[:,np.newaxis]
    
    cyd = np.reshape(cyd_, (-1))
    dydx = np.reshape(dydx_, (len(dp), -1))
    
    mmin = np.min(np.abs(dy), axis=0)
    mmax = np.max(np.abs(dy), axis=0)
    mmed = np.median(np.abs(dydx), axis=1)
    dyn_range = mmax/mmin
    
    #print(dyn_range)
    print(np.min(dyn_range), np.max(dyn_range), np.std(dyn_range))
    
    cy = np.diag(cyd)
    cyi = np.diag(1. / cyd)
    caux = np.matmul(cyi, dydx.T)
    cxi = np.matmul(dydx, caux)
    
    print('condition {:e}'.format(np.linalg.cond(cxi)))

    cx = np.linalg.inv(cxi)
    cx = np.matmul(np.linalg.inv(np.matmul(cx, cxi)), cx) # iteration to improve inverse at large cond numbers
    print('standard inverse', np.allclose(cxi, cxi.T), np.allclose(cx, cx.T), np.allclose(np.matmul(cx,cxi), np.eye(np.shape(cx)[0])))
    
    cx = stable_inverse(cxi, maxiter=30)
    print('stable inverse', np.allclose(cxi, cxi.T), np.allclose(cx, cx.T), np.allclose(np.matmul(cx,cxi), np.eye(np.shape(cx)[0])))


def unity_scale(dp):
    """"""
    dim_scale = 10**np.array([2, 3, 3, 2, 4, 3, 7, 7, 5, 7, 7, 4, 4, 4, 4, 3, 3, 3, 4, 3, 4, 4, 4])
    dim_scale = 10**np.array([3, 2, 3, 4, 0, 2, 2, 3, 2, 2, 2, 4, 3, 2, 2, 3])
    #dim_scale = 10**np.array([2, 3, 3, 1, 3, 2, 5, 5, 3, 5, 5, 2, 2, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3])
    #dim_scale = 10**np.array([2, 3, 3, 1, 3, 2, 5, 5, 3, 5, 5, 2, 2, 4, 4, 3, 3, 3])
    dp_unit = [(dp[x].value*dim_scale[x])**-1 for x in range(len(dp))]
    
    return dp_unit

def test_inversion(name='gd1', Ndim=6, vary=['progenitor', 'bary', 'halo'], align=True, errmode='fiducial'):
    """"""
    pid, dp, vlabel = get_varied_pars(vary)

    d = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
    cxi = d['cxi']
    N = np.shape(cxi)[0]
    
    cx_ = np.linalg.inv(cxi)
    cx = stable_inverse(cxi, verbose=True, maxiter=100)
    #cx_ii = stable_inverse(cx, verbose=True, maxiter=50)
    
    print('condition {:g}'.format(np.linalg.cond(cxi)))
    print('linalg inverse', np.allclose(np.matmul(cx_,cxi), np.eye(N)))
    print('stable inverse', np.allclose(np.matmul(cx,cxi), np.eye(N)))
    #print(np.matmul(cx,cxi))
    #print('inverse inverse', np.allclose(cx_ii, cxi))

def stable_inverse(a, maxiter=20, verbose=False):
    """Invert a matrix with a bad condition number"""
    N = np.shape(a)[0]
    
    # guess
    q = np.linalg.inv(a)
    qa = np.matmul(q,a)
    
    # iterate
    for i in range(maxiter):
        if verbose: print(i, np.sqrt(np.sum((qa - np.eye(N))**2)), np.allclose(qa, np.eye(N)))
        if np.allclose(qa, np.eye(N)):
            return q
        qai = np.linalg.inv(qa)
        q = np.matmul(qai,q)
        qa = np.matmul(q,a)
    
    return q

def crb_triangle(n, vary, Ndim=6, align=True, plot='all', fast=False):
    """"""
    
    pid, dp, vlabel = get_varied_pars(vary)
    plabels, units = get_parlabel(pid)
    params = ['$\Delta$' + x + '({})'.format(y) for x,y in zip(plabels, units)]
    
    if align:
        alabel = '_align'
    else:
        alabel = ''
        
    fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
    cxi = fm['cxi']
    if fast:
        cx = np.linalg.inv(cxi)
    else:
        cx = stable_inverse(cxi)
    #print(cx[0][0])
    
    if plot=='halo':
        cx = cx[:4, :4]
        params = params[:4]
    elif plot=='bary':
        cx = cx[4:9, 4:9]
        params = params[4:9]
    elif plot=='progenitor':
        cx = cx[9:, 9:]
        params = params[9:]
    
    Nvar = len(params)
    
    plt.close()
    dax = 2
    fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(dax*Nvar, dax*Nvar), sharex='col', sharey='row')
    
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar):
            plt.sca(ax[j-1][i])
            cx_2d = np.array([[cx[i][i], cx[i][j]], [cx[j][i], cx[j][j]]])
            
            w, v = np.linalg.eig(cx_2d)
            if np.all(np.isreal(v)):
                theta = np.degrees(np.arccos(v[0][0]))
                width = np.sqrt(w[0])*2
                height = np.sqrt(w[1])*2
                
                e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=mpl.cm.bone(0.5), lw=2)
                plt.gca().add_patch(e)
            plt.gca().autoscale_view()
            
            #plt.xlim(-ylim[i],ylim[i])
            #plt.ylim(-ylim[j], ylim[j])
            
            if j==Nvar-1:
                plt.xlabel(params[i])
                
            if i==0:
                plt.ylabel(params[j])
    
    # turn off unused axes
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar-1):
            plt.sca(ax[i][j])
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../plots/crb_triangle_{:s}_{:d}_{:s}_{:d}_{:s}.pdf'.format(alabel, n, vlabel, Ndim, plot))

def crb_triangle_alldim(name='gd1', vary=['progenitor', 'bary', 'halo'], align=True, plot='all', fast=False, scale=False, errmode='fiducial'):
    """Show correlations in CRB between a chosen set of parameters in a triangle plot"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    dp_opt = read_optimal_step(name, vary)
    dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]

    plabels, units = get_parlabel(pid)
    punits = [' ({})'.format(x) if len(x) else '' for x in units]
    params = ['$\Delta$ {}{}'.format(x, y) for x,y in zip(plabels, punits)]
    
    if plot=='halo':
        i0 = 11
        i1 = 15
    elif plot=='bary':
        i0 = 6
        i1 = 11
    elif plot=='progenitor':
        i0 = 0
        i1 = 6
    elif plot=='dipole':
        i0 = 15
        i1 = len(params)
    else:
        i0 = 0
        i1 = len(params)
    
    Nvar = i1 - i0
    params = params[i0:i1]
    if scale:
        dp_unit = unity_scale(dp)
        #print(dp_unit)
        dp_unit = dp_unit[i0:i1]
        pid = pid[i0:i1]
    
    label = ['RA, Dec, d', 'RA, Dec, d, $V_r$', 'RA, Dec, d, $V_r$, $\mu_\\alpha$, $\mu_\delta$']

    plt.close()
    dax = 2
    fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(dax*Nvar, dax*Nvar), sharex='col', sharey='row')
    
    for l, Ndim in enumerate([3, 4, 6]):
        fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        cxi = fm['cxi']
        #cxi = np.load('../data/crb/bspline_cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npy'.format(errmode, Ndim, name, align, vlabel))
        if fast:
            cx = np.linalg.inv(cxi)
        else:
            cx = stable_inverse(cxi)
        cx = cx[i0:i1,i0:i1]
        
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar):
                plt.sca(ax[j-1][i])
                if scale:
                    cx_2d = np.array([[cx[i][i]/dp_unit[i]**2, cx[i][j]/(dp_unit[i]*dp_unit[j])], [cx[j][i]/(dp_unit[j]*dp_unit[i]), cx[j][j]/dp_unit[j]**2]])
                else:
                    cx_2d = np.array([[cx[i][i], cx[i][j]], [cx[j][i], cx[j][j]]])
                
                w, v = np.linalg.eig(cx_2d)
                if np.all(np.isreal(v)):
                    theta = np.degrees(np.arctan2(v[1][0], v[0][0]))
                    width = np.sqrt(w[0])*2
                    height = np.sqrt(w[1])*2
                    
                    e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=mpl.cm.bone(0.1+l/4), lw=2, label=label[l])
                    plt.gca().add_patch(e)
                
                if l==1:
                    plt.gca().autoscale_view()
                
                if j==Nvar-1:
                    plt.xlabel(params[i])
                    
                if i==0:
                    plt.ylabel(params[j])
        
        # turn off unused axes
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar-1):
                plt.sca(ax[i][j])
                plt.axis('off')
        
        plt.sca(ax[int(Nvar/2-1)][int(Nvar/2-1)])
        plt.legend(loc=2, bbox_to_anchor=(1,1))
    
    plt.tight_layout()
    plt.savefig('../plots/cxi_{:s}_{:s}_a{:1d}_{:s}_{:s}.pdf'.format(errmode, name, align, vlabel, plot))

def compare_optimal_steps():
    """"""
    vary = ['progenitor', 'bary', 'halo', 'dipole', 'quad']
    vary = ['progenitor', 'bary', 'halo']
    
    for name in ['gd1', 'tri']:
        print(name)
        print(read_optimal_step(name, vary))


def get_crb(name, Nstep=10, vary=['progenitor', 'bary', 'halo'], first=True):
    """"""
    
    if first:
        store_progparams(name)
        wrap_angles(name, save=True)
        progenitor_prior(name)
        
        find_greatcircle(name=name)
        endpoints(name)
        
        for v in vary:
            step_convergence(name=name, Nstep=Nstep, vary=v)
            choose_step(name=name, Nstep=Nstep, vary=v)
    
    calculate_crb(name=name, vary=vary, verbose=True)
    crb_triangle_alldim(name=name, vary=vary)


########################
# cartesian coordinates

# accelerations
def acc_kepler(x, p=1*u.Msun):
    """Keplerian acceleration"""
    r = np.linalg.norm(x)*u.kpc
    a = -G * p * 1e11 * r**-3 * x
    
    return a.to(u.pc*u.Myr**-2)

def acc_bulge(x, p=[pparams_fid[j] for j in range(2)]):
    """"""
    r = np.linalg.norm(x)*u.kpc
    a = -(G*p[0]*x/(r * (r + p[1])**2)).to(u.pc*u.Myr**-2)
    
    return a

def acc_disk(x, p=[pparams_fid[j] for j in range(2,5)]):
    """"""
    R = np.linalg.norm(x[:2])*u.kpc
    z = x[2]
    a = -(G*p[0]*x * (R**2 + (p[1] + np.sqrt(z**2 + p[2]**2))**2)**-1.5).to(u.pc*u.Myr**-2)
    a[2] *= (1 + p[2]/np.sqrt(z**2 + p[2]**2))
    
    return a

def acc_nfw(x, p=[pparams_fid[j] for j in [5,6,8,10]]):
    """"""
    r = np.linalg.norm(x)*u.kpc
    q = np.array([1*u.Unit(1), p[2], p[3]])
    a = (p[0]**2 * p[1] * r**-3 * (1/(1+p[1]/r) - np.log(1+r/p[1])) * x * q**-2).to(u.pc*u.Myr**-2)
    
    return a

def acc_dipole(x, p=[pparams_fid[j] for j in range(11,14)]):
    """Acceleration due to outside dipole perturbation"""
    
    pv = [x.value for x in p]
    a = np.sqrt(3/(4*np.pi)) * np.array([pv[2], pv[0], pv[1]])*u.pc*u.Myr**-2
    
    return a

def acc_quad(x, p=[pparams_fid[j] for j in range(14,19)]):
    """Acceleration due to outside quadrupole perturbation"""

    a = np.zeros(3)*u.pc*u.Myr**-2
    f = 0.5*np.sqrt(15/np.pi)
    
    a[0] = x[0]*(f*p[4] - f/np.sqrt(3)*p[2]) + x[1]*f*p[0] + x[2]*f*p[3]
    a[1] = x[0]*f*p[0] - x[1]*(f*p[4] + f/np.sqrt(3)*p[2]) + x[2]*f*p[1]
    a[2] = x[0]*f*p[3] + x[1]*f*p[1] + x[2]*2*f/np.sqrt(3)*p[2]
    
    return a.to(u.pc*u.Myr**-2)

def acc_octu(x, p=[pparams_fid[j] for j in range(19,26)]):
    """Acceleration due to outside octupole perturbation"""
    
    a = np.zeros(3)*u.pc*u.Myr**-2
    f = np.array([0.25*np.sqrt(35/(2*np.pi)), 0.5*np.sqrt(105/np.pi), 0.25*np.sqrt(21/(2*np.pi)), 0.25*np.sqrt(7/np.pi), 0.25*np.sqrt(21/(2*np.pi)), 0.25*np.sqrt(105/np.pi), 0.25*np.sqrt(35/(2*np.pi))])
    
    xu = x.unit
    pu = p[0].unit
    pvec = np.array([i.value for i in p]) * pu
    dmat = np.ones((3,7)) * f * pvec * xu**2
    x = np.array([i.value for i in x])

    dmat[0] *= np.array([6*x[0]*x[1], x[1]*x[2], -2*x[0]*x[1], -6*x[0]*x[2], 4*x[2]**2-x[1]**2-3*x[0]**2, 2*x[0]*x[2], 3*x[0]**2-3*x[1]**2])
    dmat[1] *= np.array([3*x[0]**2-3*x[1]**2, x[0]*x[2], 4*x[2]**2-x[0]**2-3*x[1]**2, -6*x[1]*x[2], -2*x[0]*x[1], -2*x[1]*x[2], -6*x[0]*x[1]])
    dmat[2] *= np.array([0, x[0]*x[1], 8*x[1]*x[2], 6*x[2]**2-3*x[0]**2-3*x[1]**2, 8*x[0]*x[2], x[0]**2-x[1]**2, 0])
    
    a = np.einsum('ij->i', dmat) * dmat.unit
    
    return a.to(u.pc*u.Myr**-2)

# derivatives
def der_kepler(x, p=1*u.Msun):
    """Derivative of Kepler potential parameters wrt cartesian components of the acceleration"""
    
    r = np.linalg.norm(x)*u.kpc
    
    dmat = np.zeros((3,1)) * u.pc**-1 * u.Myr**2 * u.Msun
    dmat[:,0] = (-r**3/(G*x)).to(u.pc**-1 * u.Myr**2 * u.Msun) * 1e-11
    
    return dmat.value

def pder_kepler(x, p=1*u.Msun):
    """Derivative of cartesian components of the acceleration wrt to Kepler potential parameter"""
    r = np.linalg.norm(x)*u.kpc
    
    dmat = np.zeros((3,1)) * u.pc * u.Myr**-2 * u.Msun**-1
    dmat[:,0] = (-G*x*r**-3).to(u.pc * u.Myr**-2 * u.Msun**-1) * 1e11
    
    return dmat.value

def pder_nfw(x, pu=[pparams_fid[j] for j in [5,6,8,10]]):
    """Calculate derivatives of cartesian components of the acceleration wrt halo potential parameters"""
    
    p = pu
    q = np.array([1, p[2], p[3]])
    
    # physical quantities
    r = np.linalg.norm(x)*u.kpc
    a = acc_nfw(x, p=pu)
    
    #  derivatives
    dmat = np.zeros((3, 4))
    
    # Vh
    dmat[:,0] = 2*a/p[0]
    
    # Rh
    dmat[:,1] = a/p[1] + p[0]**2 * p[1] * r**-3 * (1/(p[1]+p[1]**2/r) - 1/(r*(1+p[1]/r)**2)) * x * q**-2
    
    # qy, qz
    for i in [1,2]:
        dmat[i,i+1] = (-2*a[i]/q[i]).value
    
    return dmat

def pder_bulge(x, pu=[pparams_fid[j] for j in range(2)]):
    """Calculate derivarives of cartesian components of the acceleration wrt Hernquist bulge potential parameters"""
    
    # coordinates
    r = np.linalg.norm(x)*u.kpc
    
    # accelerations
    ab = acc_bulge(x, p=pu[:2])
    
    #  derivatives
    dmat = np.zeros((3, 2))
    
    # Mb
    dmat[:,0] = ab/pu[0]
    
    # ab
    dmat[:,1] = 2 * ab / (r + pu[1])
    
    return dmat

def pder_disk(x, pu=[pparams_fid[j] for j in range(2,5)]):
    """Calculate derivarives of cartesian components of the acceleration wrt Miyamoto-Nagai disk potential parameters"""
    
    # coordinates
    R = np.linalg.norm(x[:2])*u.kpc
    z = x[2]
    aux = np.sqrt(z**2 + pu[2]**2)
    
    # accelerations
    ad = acc_disk(x, p=pu)
    
    #  derivatives
    dmat = np.zeros((3, 3))
    
    # Md
    dmat[:,0] = ad / pu[0]
    
    # ad
    dmat[:,1] = 3 * ad * (pu[1] + aux) / (R**2 + (pu[1] + aux)**2)
    
    # bd
    dmat[:2,2] = 3 * ad[:2] * (pu[1] + aux) / (R**2 + (pu[1] + aux)**2) * pu[2] / aux
    dmat[2,2] = (3 * ad[2] * (pu[1] + aux) / (R**2 + (pu[1] + aux)**2) * pu[2] / aux - G * pu[0] * z * (R**2 + (pu[1] + aux)**2)**-1.5 * z**2 * (pu[2]**2 + z**2)**-1.5).value
    
    return dmat

def der_dipole(x, pu=[pparams_fid[j] for j in range(11,14)]):
    """Calculate derivatives of dipole potential parameters wrt (Cartesian) components of the acceleration vector a"""
    
    # shape: 3, Npar
    dmat = np.zeros((3,3))
    
    f = np.sqrt((4*np.pi)/3)
    
    dmat[0,2] = f
    dmat[1,0] = f
    dmat[2,1] = f
    
    return dmat

def pder_dipole(x, pu=[pparams_fid[j] for j in range(11,14)]):
    """Calculate derivatives of (Cartesian) components of the acceleration vector a wrt dipole potential parameters"""
    
    # shape: 3, Npar
    dmat = np.zeros((3,3))
    
    f = np.sqrt(3/(4*np.pi))
    
    dmat[0,2] = f
    dmat[1,0] = f
    dmat[2,1] = f
    
    return dmat

def der_quad(x, p=[pparams_fid[j] for j in range(14,19)]):
    """Caculate derivatives of quadrupole potential parameters wrt (Cartesian) components of the acceleration vector a"""
    
    f = 2/np.sqrt(15/np.pi)
    s = np.sqrt(3)
    x = [1e-3/i.value for i in x]
    
    dmat = np.ones((3,5)) * f
    
    dmat[0] = np.array([x[1], 0, -s*x[0], x[2], x[0]])
    dmat[1] = np.array([x[0], x[2], -s*x[1], 0, -x[1]])
    dmat[2] = np.array([0, x[1], 0.5*s*x[2], x[0], 0])
    
    return dmat

def pder_quad(x, p=[pparams_fid[j] for j in range(14,19)]):
    """Caculate derivatives of (Cartesian) components of the acceleration vector a wrt quadrupole potential parameters"""
    
    f = 0.5*np.sqrt(15/np.pi)
    s = 1/np.sqrt(3)
    x = [1e-3*i.value for i in x]
    
    dmat = np.ones((3,5)) * f
    
    dmat[0] *= np.array([x[1], 0, -s*x[0], x[2], x[0]])
    dmat[1] *= np.array([x[0], x[2], -s*x[1], 0, -x[1]])
    dmat[2] *= np.array([0, x[1], 2*s*x[2], x[0], 0])
    
    return dmat

def pder_octu(x, p=[pparams_fid[j] for j in range(19,26)]):
    """Caculate derivatives of (Cartesian) components of the acceleration vector a wrt octupole potential parameters"""
    
    f = np.array([0.25*np.sqrt(35/(2*np.pi)), 0.5*np.sqrt(105/np.pi), 0.25*np.sqrt(21/(2*np.pi)), 0.25*np.sqrt(7/np.pi), 0.25*np.sqrt(21/(2*np.pi)), 0.25*np.sqrt(105/np.pi), 0.25*np.sqrt(35/(2*np.pi))])
    x = [1e-3*i.value for i in x]
    
    dmat = np.ones((3,7)) * f
    
    dmat[0] *= np.array([6*x[0]*x[1], x[1]*x[2], -2*x[0]*x[1], -6*x[0]*x[2], 4*x[2]**2-x[1]**2-3*x[0]**2, 2*x[0]*x[2], 3*x[0]**2-3*x[1]**2])
    dmat[1] *= np.array([3*x[0]**2-3*x[1]**2, x[0]*x[2], 4*x[2]**2-x[0]**2-3*x[1]**2, -6*x[1]*x[2], -2*x[0]*x[1], -2*x[1]*x[2], -6*x[0]*x[1]])
    dmat[2] *= np.array([0, x[0]*x[1], 8*x[1]*x[2], 6*x[2]**2-3*x[0]**2-3*x[1]**2, 8*x[0]*x[2], x[0]**2-x[1]**2, 0])
    
    return dmat

def crb_ax(n, Ndim=6, vary=['halo', 'bary', 'progenitor'], align=True, fast=False):
    """Calculate CRB inverse matrix for 3D acceleration at position x in a halo potential"""
    
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    # read in full inverse CRB for stream modeling
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
    if fast:
        cx = np.linalg.inv(cxi)
    else:
        cx = stable_inverse(cxi)
    
    # subset halo parameters
    Nhalo = 4
    cq = cx[:Nhalo,:Nhalo]
    if fast:
        cqi = np.linalg.inv(cq)
    else:
        cqi = stable_inverse(cq)
    
    xi = np.array([-8.3, 0.1, 0.1])*u.kpc
    
    x0, v0 = gd1_coordinates()
    #xi = np.array(x0)*u.kpc
    d = 50
    Nb = 20
    x = np.linspace(x0[0]-d, x0[0]+d, Nb)
    y = np.linspace(x0[1]-d, x0[1]+d, Nb)
    x = np.linspace(-d, d, Nb)
    y = np.linspace(-d, d, Nb)
    xv, yv = np.meshgrid(x, y)
    
    xf = np.ravel(xv)
    yf = np.ravel(yv)
    af = np.empty((Nb**2, 3))
    
    plt.close()
    fig, ax = plt.subplots(3,3,figsize=(11,10))
    
    dimension = ['x', 'y', 'z']
    xlabel = ['y', 'x', 'x']
    ylabel = ['z', 'z', 'y']
    
    for j in range(3):
        if j==0:
            xin = np.array([np.repeat(x0[j], Nb**2), xf, yf]).T
        elif j==1:
            xin = np.array([xf, np.repeat(x0[j], Nb**2), yf]).T
        elif j==2:
            xin = np.array([xf, yf, np.repeat(x0[j], Nb**2)]).T
        for i in range(Nb**2):
            #xi = np.array([xf[i], yf[i], x0[2]])*u.kpc
            xi = xin[i]*u.kpc
            a = acc_nfw(xi)
            
            dqda = halo_accelerations(xi)
            
            cai = np.matmul(dqda, np.matmul(cqi, dqda.T))
            if fast:
                ca = np.linalg.inv(cai)
            else:
                ca = stable_inverse(cai)
            a_crb = (np.sqrt(np.diag(ca)) * u.km**2 * u.kpc**-1 * u.s**-2).to(u.pc*u.Myr**-2)
            af[i] = np.abs(a_crb/a)
            af[i] = a_crb

        for i in range(3):
            plt.sca(ax[j][i])
            im = plt.imshow(af[:,i].reshape(Nb,Nb), extent=[-d, d, -d, d], cmap=mpl.cm.gray) #, norm=mpl.colors.LogNorm(), vmin=1e-2, vmax=0.1)
            
            plt.xlabel(xlabel[j]+' (kpc)')
            plt.ylabel(ylabel[j]+' (kpc)')
            
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("top", size="4%", pad=0.05)
            plt.colorbar(im, cax=cax, orientation='horizontal')
            
            plt.gca().xaxis.set_ticks_position('top')
            cax.tick_params(axis='x', labelsize='xx-small')
            if j==0:
                plt.title('a$_{}$'.format(dimension[i]), y=4)
        
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig('../plots/acc_{}_{}_{}.png'.format(n, vlabel, Ndim))

def acc_cart(x, components=['bary', 'halo', 'dipole']):
    """"""
    acart = np.zeros(3) * u.pc*u.Myr**-2
    dict_acc = {'bary': [acc_bulge, acc_disk], 'halo': [acc_nfw], 'dipole': [acc_dipole], 'quad': [acc_quad], 'octu': [acc_octu], 'point': [acc_kepler]}
    accelerations = []
    
    for c in components:
        accelerations += dict_acc[c]
    
    for acc in accelerations:
        a_ = acc(x)
        acart += a_
    
    return acart

def acc_rad(x, components=['bary', 'halo', 'dipole']):
    """Return radial acceleration"""
    
    r = np.linalg.norm(x) * x.unit
    theta = np.arccos(x[2].value/r.value)
    phi = np.arctan2(x[1].value, x[0].value)
    trans = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    
    a_cart = acc_cart(x, components=components)
    a_rad = np.dot(a_cart, trans)
    
    return a_rad

def ader_cart(x, components=['bary', 'halo', 'dipole']):
    """"""
    dacart = np.empty((3,0))
    dict_der = {'bary': [der_bulge, der_disk], 'halo': [der_nfw], 'dipole': [der_dipole], 'quad': [der_quad], 'point': [der_kepler]}
    derivatives = []
    
    for c in components:
        derivatives += dict_der[c]
    
    for ader in derivatives:
        da_ = ader(x)
        dacart = np.hstack((dacart, da_))
    
    return dacart

def apder_cart(x, components=['bary', 'halo', 'dipole']):
    """"""
    dacart = np.empty((3,0))
    dict_der = {'bary': [pder_bulge, pder_disk], 'halo': [pder_nfw], 'dipole': [pder_dipole], 'quad': [pder_quad], 'octu': [pder_octu], 'point': [pder_kepler]}
    derivatives = []
    
    for c in components:
        derivatives += dict_der[c]
    
    for ader in derivatives:
        da_ = ader(x)
        dacart = np.hstack((dacart, da_))
    
    return dacart

def apder_rad(x, components=['bary', 'halo', 'dipole']):
    """Return dar/dx_pot (radial acceleration/potential parameters) evaluated at vector x"""
    
    r = np.linalg.norm(x) * x.unit
    theta = np.arccos(x[2].value/r.value)
    phi = np.arctan2(x[1].value, x[0].value)
    trans = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    
    dadq_cart = apder_cart(x, components=components)
    dadq_rad = np.einsum('ij,i->j', dadq_cart, trans)
    
    return dadq_rad

def crb_acart(n, Ndim=6, vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], component='all', align=True, d=20, Nb=50, fast=False, scale=False, relative=True, progenitor=False, errmode='fiducial'):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    if relative:
        vmin = 1e-2
        vmax = 1
        rlabel = ' / a'
    else:
        vmin = 3e-1
        vmax = 1e1
        rlabel =  ' (pc Myr$^{-2}$)'
    
    # read in full inverse CRB for stream modeling
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, errmode, n, vlabel, Ndim))
    if fast:
        cx = np.linalg.inv(cxi)
    else:
        cx = stable_inverse(cxi)
    
    # choose the appropriate components:
    Nprog, Nbary, Nhalo, Ndipole, Npoint = [6, 5, 4, 3, 1]
    if 'progenitor' not in vary:
        Nprog = 0
    nstart = {'bary': Nprog, 'halo': Nprog + Nbary, 'dipole': Nprog + Nbary + Nhalo, 'all': Nprog, 'point': 0}
    nend = {'bary': Nprog + Nbary, 'halo': Nprog + Nbary + Nhalo, 'dipole': Nprog + Nbary + Nhalo + Ndipole, 'all': np.shape(cx)[0], 'point': 1}
    
    if 'progenitor' not in vary:
        nstart['dipole'] = Npoint
        nend['dipole'] = Npoint + Ndipole
    
    if component in ['bary', 'halo', 'dipole', 'point']:
        components = [component]
    else:
        components = [x for x in vary if x!='progenitor']
    cq = cx[nstart[component]:nend[component], nstart[component]:nend[component]]
    Npot = np.shape(cq)[0]
    
    if fast:
        cqi = np.linalg.inv(cq)
    else:
        cqi = stable_inverse(cq)
    
    if scale:
        dp_opt = read_optimal_step(n, vary)
        dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
        
        scale_vec = np.array([x.value for x in dp[nstart[component]:nend[component]]])
        scale_mat = np.outer(scale_vec, scale_vec)
        cqi *= scale_mat

    if progenitor:
        x0, v0 = gd1_coordinates()
    else:
        x0 = np.array([4, 4, 0])
    Rp = np.linalg.norm(x0[:2])
    zp = x0[2]
    
    R = np.linspace(-d, d, Nb)
    k = x0[1]/x0[0]
    x = R/np.sqrt(1+k**2)
    y = k * x
    
    z = np.linspace(-d, d, Nb)
    
    xv, zv = np.meshgrid(x, z)
    yv, zv = np.meshgrid(y, z)
    xin = np.array([np.ravel(xv), np.ravel(yv), np.ravel(zv)]).T

    Npix = np.size(xv)
    af = np.empty((Npix, 3))
    derf = np.empty((Npix, 3, Npot))
    
    for i in range(Npix):
        xi = xin[i]*u.kpc
        a = acc_cart(xi, components=components)
        
        dadq = apder_cart(xi, components=components)
        derf[i] = dadq
        
        ca = np.matmul(dadq, np.matmul(cq, dadq.T))
        a_crb = np.sqrt(np.diag(ca)) * u.pc * u.Myr**-2
        if relative:
            af[i] = np.abs(a_crb/a)
        else:
            af[i] = a_crb
        #print(xi, a_crb)
    
    # save
    np.savez('../data/crb_acart{:s}_{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}'.format(alabel, errmode, n, vlabel, component, Ndim, d, Nb, relative), acc=af, x=xin, der=derf)
    
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    label = ['$\Delta$ $a_X$', '$\Delta$ $a_Y$', '$\Delta$ $a_Z$']
    
    for i in range(3):
        plt.sca(ax[i])
        im = plt.imshow(af[:,i].reshape(Nb, Nb), origin='lower', extent=[-d, d, -d, d], cmap=mpl.cm.gray, vmin=vmin, vmax=vmax, norm=mpl.colors.LogNorm())
        if progenitor:
            plt.plot(Rp, zp, 'r*', ms=10)
        
        plt.xlabel('R (kpc)')
        plt.ylabel('Z (kpc)')
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        plt.ylabel(label[i] + rlabel)
        
    plt.tight_layout()
    plt.savefig('../plots/crb_acc_cart{:s}_{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}.png'.format(alabel, errmode, n, vlabel, component, Ndim, d, Nb, relative))

def crb_acart_cov(n, Ndim=6, vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], component='all', j=0, align=True, d=20, Nb=30, fast=False, scale=False, relative=True, progenitor=False, batch=False, errmode='fiducial'):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    if relative:
        vmin = 1e-2
        vmax = 1
        rlabel = ' / a'
    else:
        vmin = -0.005
        vmax = 0.005
        #vmin = 1e-2
        #vmax = 1e0
        rlabel =  ' (pc Myr$^{-2}$)'
    
    # read in full inverse CRB for stream modeling
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, errmode, n, vlabel, Ndim))
    if fast:
        cx = np.linalg.inv(cxi)
    else:
        cx = stable_inverse(cxi)
    
    # choose the appropriate components:
    Nprog, Nbary, Nhalo, Ndipole, Nquad, Npoint = [6, 5, 4, 3, 5, 1]
    if 'progenitor' not in vary:
        Nprog = 0
    nstart = {'bary': Nprog, 'halo': Nprog + Nbary, 'dipole': Nprog + Nbary + Nhalo, 'quad': Nprog + Nbary + Nhalo + Ndipole, 'all': Nprog, 'point': 0}
    nend = {'bary': Nprog + Nbary, 'halo': Nprog + Nbary + Nhalo, 'dipole': Nprog + Nbary + Nhalo + Ndipole, 'quad': Nprog + Nbary + Nhalo + Ndipole + Nquad, 'all': np.shape(cx)[0], 'point': 1}
    
    if 'progenitor' not in vary:
        nstart['dipole'] = Npoint
        nend['dipole'] = Npoint + Ndipole
    
    if component in ['bary', 'halo', 'dipole', 'quad', 'point']:
        components = [component]
    else:
        components = [x for x in vary if x!='progenitor']
    cq = cx[nstart[component]:nend[component], nstart[component]:nend[component]]
    Npot = np.shape(cq)[0]
    
    if fast:
        cqi = np.linalg.inv(cq)
    else:
        cqi = stable_inverse(cq)
    
    if scale:
        dp_opt = read_optimal_step(n, vary)
        dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
        
        scale_vec = np.array([x.value for x in dp[nstart[component]:nend[component]]])
        scale_mat = np.outer(scale_vec, scale_vec)
        cqi *= scale_mat

    if progenitor:
        prog_coords = {-1: gd1_coordinates(), -2: pal5_coordinates(), -3: tri_coordinates(), -4: atlas_coordinates()}
        x0, v0 = prog_coords[n]
        print(x0)
    else:
        x0 = np.array([4, 4, 0])
    Rp = np.linalg.norm(x0[:2])
    zp = x0[2]
    
    R = np.linspace(-d, d, Nb)
    k = x0[1]/x0[0]
    x = R/np.sqrt(1+k**2)
    y = k * x
    
    z = np.linspace(-d, d, Nb)
    
    xv, zv = np.meshgrid(x, z)
    yv, zv = np.meshgrid(y, z)
    xin = np.array([np.ravel(xv), np.ravel(yv), np.ravel(zv)]).T

    Npix = np.size(xv)
    af = np.empty((Npix, 3))
    derf = np.empty((Npix*3, Npot))
    
    for i in range(Npix):
        xi = xin[i]*u.kpc
        a = acc_cart(xi, components=components)
        
        dadq = apder_cart(xi, components=components)
        derf[i*3:(i+1)*3] = dadq
    
    ca = np.matmul(derf, np.matmul(cq, derf.T))
    
    Nx = Npot
    Nw = Npix*3
    vals, vecs = la.eigh(ca, eigvals=(Nw - Nx - 2, Nw - 1))

    ## check orthogonality:
    #for i in range(Npot-1):
        #for k in range(i+1, Npot):
            #print(i, k)
            #print(np.dot(vecs[:,i], vecs[:,k]))
            #print(np.dot(vecs[::3,i], vecs[::3,k]), np.dot(vecs[1::3,i], vecs[1::3,k]), np.dot(vecs[1::3,i], vecs[1::3,k]))
    
    # save
    np.savez('../data/crb_acart_cov{:s}_{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(alabel, errmode, n, vlabel, component, Ndim, d, Nb, relative, progenitor), x=xin, der=derf, c=ca)
    
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    if j==0:
        vcomb = np.sqrt(np.sum(vecs**2*vals, axis=1))
        label = ['($\Sigma$ Eigval $\\times$ Eigvec$^2$ $a_{}$'.format(x)+')$^{1/2}$' for x in ['X', 'Y', 'Z']]
        vmin = 1e-2
        vmax = 5e0
        norm = mpl.colors.LogNorm()
    else:
        vcomb = vecs[:,j]
        label = ['Eig {} $a_{}$'.format(np.abs(j), x) for x in ['X', 'Y', 'Z']]
        vmin = -0.025
        vmax = 0.025
        norm = None

    for i in range(3):
        plt.sca(ax[i])
        #im = plt.imshow(vecs[i::3,j].reshape(Nb, Nb), origin='lower', extent=[-d, d, -d, d], cmap=mpl.cm.gray, vmin=vmin, vmax=vmax)
        im = plt.imshow(vcomb[i::3].reshape(Nb, Nb), origin='lower', extent=[-d, d, -d, d], cmap=mpl.cm.gray, vmin=vmin, vmax=vmax, norm=norm)
        if progenitor:
            plt.plot(Rp, zp, 'r*', ms=10)
        
        plt.xlabel('R (kpc)')
        plt.ylabel('Z (kpc)')
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        plt.ylabel(label[i])
        
    plt.tight_layout()
    if batch:
        return fig
    else:
        plt.savefig('../plots/crb_acc_cart_cov{:s}_{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.png'.format(alabel, errmode, n, vlabel, component, np.abs(j), Ndim, d, Nb, relative, progenitor))


def a_vecfield(vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], component='all', d=20, Nb=10):
    """Plot acceleration field in R,z plane"""
    if component in ['bary', 'halo', 'dipole', 'quad', 'point']:
        components = [component]
    else:
        components = [x for x in vary if x!='progenitor']

    x0 = np.array([4, 4, 0])
    R = np.linspace(-d, d, Nb)
    k = x0[1]/x0[0]
    x = R/np.sqrt(1+k**2)
    y = k * x
    
    z = np.linspace(-d, d, Nb)
    
    xv, zv = np.meshgrid(x, z)
    yv, zv = np.meshgrid(y, z)
    xin = np.array([np.ravel(xv), np.ravel(yv), np.ravel(zv)]).T
    Rin = np.linalg.norm(xin[:,:2], axis=1) * np.sign(xin[:,0])
    zin = xin[:,2]

    Npix = np.size(xv)
    acart_pix = np.empty((Npix, 3))
    acyl_pix = np.empty((Npix, 2))
    
    for i in range(Npix):
        xi = xin[i]*u.kpc
        acart = acc_cart(xi, components=components)
        acart_pix[i] = acart
    
    acyl_pix[:,0] = np.linalg.norm(acart_pix[:,:2], axis=1) * -np.sign(xin[:,0])
    acyl_pix[:,1] = acart_pix[:,2]
    
    plt.close()
    plt.figure()
    
    plt.quiver(Rin, zin, acyl_pix[:,0], acyl_pix[:,1])
    
    plt.tight_layout()

def a_crbcov_vecfield(n, Ndim=6, vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], errmode='fiducial', component='all', j=0, align=True, d=20, Nb=10, fast=False, scale=True, relative=False, progenitor=False, batch=False):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    if relative:
        vmin = 1e-2
        vmax = 1
        rlabel = ' / a'
    else:
        vmin = -0.005
        vmax = 0.005
        #vmin = 1e-2
        #vmax = 1e0
        rlabel =  ' (pc Myr$^{-2}$)'
    
    # read in full inverse CRB for stream modeling
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, errmode, n, vlabel, Ndim))
    if fast:
        cx = np.linalg.inv(cxi)
    else:
        cx = stable_inverse(cxi)
    
    # choose the appropriate components:
    Nprog, Nbary, Nhalo, Ndipole, Nquad, Npoint = [6, 5, 4, 3, 5, 1]
    if 'progenitor' not in vary:
        Nprog = 0
    nstart = {'bary': Nprog, 'halo': Nprog + Nbary, 'dipole': Nprog + Nbary + Nhalo, 'quad': Nprog + Nbary + Nhalo + Ndipole, 'all': Nprog, 'point': 0}
    nend = {'bary': Nprog + Nbary, 'halo': Nprog + Nbary + Nhalo, 'dipole': Nprog + Nbary + Nhalo + Ndipole, 'quad': Nprog + Nbary + Nhalo + Ndipole + Nquad, 'all': np.shape(cx)[0], 'point': 1}
    
    if 'progenitor' not in vary:
        nstart['dipole'] = Npoint
        nend['dipole'] = Npoint + Ndipole
    
    if component in ['bary', 'halo', 'dipole', 'quad', 'point']:
        components = [component]
    else:
        components = [x for x in vary if x!='progenitor']
    cq = cx[nstart[component]:nend[component], nstart[component]:nend[component]]
    Npot = np.shape(cq)[0]
    
    if fast:
        cqi = np.linalg.inv(cq)
    else:
        cqi = stable_inverse(cq)
    
    if scale:
        dp_opt = read_optimal_step(n, vary)
        dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
        
        scale_vec = np.array([x.value for x in dp[nstart[component]:nend[component]]])
        scale_mat = np.outer(scale_vec, scale_vec)
        cqi *= scale_mat

    if progenitor:
        x0, v0 = gd1_coordinates()
    else:
        x0 = np.array([4, 4, 0])
    Rp = np.linalg.norm(x0[:2])
    zp = x0[2]
    
    R = np.linspace(-d, d, Nb)
    k = x0[1]/x0[0]
    x = R/np.sqrt(1+k**2)
    y = k * x
    
    z = np.linspace(-d, d, Nb)
    
    xv, zv = np.meshgrid(x, z)
    yv, zv = np.meshgrid(y, z)
    xin = np.array([np.ravel(xv), np.ravel(yv), np.ravel(zv)]).T
    Rin = np.linalg.norm(xin[:,:2], axis=1) * np.sign(xin[:,0])
    zin = xin[:,2]

    Npix = np.size(xv)
    acart_pix = np.empty((Npix, 3))
    acyl_pix = np.empty((Npix, 2))
    vcomb_pix = np.empty((Npix, 2))

    af = np.empty((Npix, 3))
    derf = np.empty((Npix*3, Npot))
    
    for i in range(Npix):
        xi = xin[i]*u.kpc
        a = acc_cart(xi, components=components)
        acart_pix[i] = a
        
        dadq = apder_cart(xi, components=components)
        derf[i*3:(i+1)*3] = dadq
    
    acyl_pix[:,0] = np.linalg.norm(acart_pix[:,:2], axis=1) * -np.sign(xin[:,0])
    acyl_pix[:,1] = acart_pix[:,2]
    
    ca = np.matmul(derf, np.matmul(cq, derf.T))
    
    Nx = Npot
    Nw = Npix*3
    vals, vecs = la.eigh(ca, eigvals=(Nw - Nx - 2, Nw - 1))

    if j==0:
        vcomb = np.sqrt(np.sum(vecs**2*vals, axis=1))
        label = ['($\Sigma$ Eigval $\\times$ Eigvec$^2$ $a_{}$'.format(x)+')$^{1/2}$' for x in ['X', 'Y', 'Z']]
        vmin = 1e-3
        vmax = 1e-1
        norm = mpl.colors.LogNorm()
    else:
        vcomb = vecs[:,j]*np.sqrt(vals[j])
        label = ['Eig {} $a_{}$'.format(np.abs(j), x) for x in ['X', 'Y', 'Z']]
        vmin = -0.025
        vmax = 0.025
        norm = None
    
    
    vcomb_pix[:,0] = np.sqrt(vcomb[0::3]**2 + vcomb[1::3]**2) * -np.sign(xin[:,0])
    #vcomb_pix[:,0] = np.sqrt(vcomb[0::3]**2 + vcomb[1::3]**2) * -np.sign(vcomb[0::3])
    vcomb_pix[:,1] = vcomb[2::3]
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    plt.quiver(Rin, zin, acyl_pix[:,0], acyl_pix[:,1], pivot='middle')
    
    plt.xlabel('R (kpc)')
    plt.ylabel('Z (kpc)')
    plt.title('Acceleration {}'.format(component), fontsize='medium')
    
    plt.sca(ax[1])
    plt.quiver(Rin, zin, vcomb_pix[:,0], vcomb_pix[:,1], pivot='middle', headwidth=0, headlength=0, headaxislength=0, scale=0.02, scale_units='xy')
    
    plt.xlabel('R (kpc)')
    plt.ylabel('Z (kpc)')
    plt.title('Eigenvector {}'.format(np.abs(j)), fontsize='medium')
    
    plt.tight_layout()
    if batch:
        return fig
    else:
        plt.savefig('../plots/afield_crbcov{:s}_{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}_{:d}.png'.format(alabel, errmode, n, vlabel, component, np.abs(j), Ndim, d, Nb, relative))


def summary(n, mode='scalar', vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], errmode='fiducial', component='all'):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    fn = {'scalar': crb_acart_cov, 'vector': a_crbcov_vecfield}
    bins = {'scalar': 30, 'vector': 10}
    
    Nprog, Nbary, Nhalo, Ndipole, Nquad, Npoint = [6, 5, 4, 3, 5, 1]
    Npars = {'bary': Nbary, 'halo': Nhalo, 'dipole': Ndipole, 'quad': Nquad, 'point': Npoint}
    
    if component in ['bary', 'halo', 'dipole', 'quad', 'point']:
        components = [component]
    else:
        components = [x for x in vary if x!='progenitor']
    
    Niter = [Npars[x] for x in components]
    Niter = sum(Niter) + 1
    
    pp = PdfPages('../plots/acceleration_{}_{}_{}_{}_{}.pdf'.format(n, errmode, vlabel, component, mode))
    
    for i in range(Niter):
        print(i, Niter)
        fig = fn[mode](-1, progenitor=True, batch=True, errmode=errmode, vary=vary, component=component, j=-i, d=20, Nb=bins[mode])
        pp.savefig(fig)
    
    pp.close()


#########
# Summary
def full_names():
    """"""
    full = {'gd1': 'GD-1', 'atlas': 'ATLAS', 'tri': 'Triangulum', 'ps1a': 'PS1A', 'ps1b': 'PS1B', 'ps1c': 'PS1C', 'ps1d': 'PS1D', 'ps1e': 'PS1E', 'ophiuchus': 'Ophiuchus', 'hermus': 'Hermus', 'kwando': 'Kwando', 'orinoco': 'Orinoco', 'sangarius': 'Sangarius', 'scamander': 'Scamander'}
    return full

def full_name(name):
    """"""
    full = full_names()
    
    return full[name]

def get_done(sort_length=False):
    """"""
    done = ['gd1', 'tri', 'atlas', 'ps1a', 'ps1c', 'ps1e', 'ophiuchus', 'kwando', 'orinoco', 'sangarius', 'hermus', 'ps1d']
    done = ['gd1', 'tri', 'atlas', 'ps1a', 'ps1c', 'ps1e', 'kwando', 'orinoco', 'sangarius', 'hermus', 'ps1d']
    
    # length
    if sort_length:
        tosort = []
        
        for name in done:
            mock = pickle.load(open('../data/mock_{}.params'.format(name), 'rb'))
            tosort += [np.max(mock['xi_range']) - np.min(mock['xi_range'])]

        done = [x for _,x in sorted(zip(tosort,done))]
    
    else:
        tosort = []
        
        vary = ['progenitor', 'bary', 'halo']
        Ndim = 6
        errmode = 'fiducial'
        align = True
        pid, dp_fid, vlabel = get_varied_pars(vary)
        pid_vh = myutils.wherein(np.array(pid), np.array([5]))
        
        for name in done:
            fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
            cxi = fm['cxi']
            cx = stable_inverse(cxi)
            
            crb = np.sqrt(np.diag(cx))
            tosort += [crb[pid_vh]]
        
        done = [x for _,x in sorted(zip(tosort,done))][::-1]
    
    return done

def store_mocks():
    """"""
    done = get_done()
    
    for name in done:
        stream = stream_model(name)
        np.save('../data/streams/mock_observed_{}'.format(name), stream.obs)

def period(name):
    """Return orbital period in units of stepsize and number of complete periods"""
    
    orbit = stream_orbit(name=name)
    r = np.linalg.norm(orbit['x'].to(u.kpc), axis=0)
    
    a = np.abs(np.fft.rfft(r))
    f = np.argmax(a[1:]) + 1
    p = np.size(a)/f
    
    return (p, f)

def extract_crbs(Ndim=6, vary=['progenitor', 'bary', 'halo'], component='halo', errmode='fiducial', j=0, align=True, fast=False, scale=False):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)

    names = get_done()
    
    tout = Table(names=('name', 'crb'))
    
    pparams0 = pparams_fid
    pid_comp, dp_fid2, vlabel2 = get_varied_pars(component)
    Np = len(pid_comp)
    pid_crb = myutils.wherein(np.array(pid), np.array(pid_comp))
    
    plt.close()
    fig, ax = plt.subplots(Np,1,figsize=(10,15), subplot_kw=dict(projection='mollweide'))
    
    for name in names[:]:
        fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        cxi = fm['cxi']
        if fast:
            cx = np.linalg.inv(cxi)
        else:
            cx = stable_inverse(cxi)
        
        crb = np.sqrt(np.diag(cx))
        #print([pparams0[pid_comp[i]] for i in range(Np)])
        crb_frac = [crb[pid_crb[i]]/pparams0[pid_comp[i]].value for i in range(Np)]
        print(name, crb_frac)
        
        stream = stream_model(name=name)
        
        for i in range(Np):
            plt.sca(ax[i])
            color_index = np.array(crb_frac[:])
            color_index[color_index>0.2] = 0.2
            color_index /= 0.2
            color = mpl.cm.viridis(color_index[i])
            
            plt.plot(np.radians(stream.obs[0]), np.radians(stream.obs[1]), 'o', color=color, ms=4)
    
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
    plt.savefig('../plots/crb_onsky_{}.png'.format(component))

def vhrh_correlation(Ndim=6, vary=['progenitor', 'bary', 'halo'], component='halo', errmode='fiducial', align=True):
    """"""
    names = get_done()
    t = Table.read('../data/crb/ar_orbital_summary.fits')
    N = len(names)
    p = np.empty(N)
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    pid_comp, dp_fid2, vlabel2 = get_varied_pars(component)
    i = pid_comp[0]
    j = pid_comp[1]
    
    for e, name in enumerate(names):
        fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        cxi = fm['cxi']
        cx = stable_inverse(cxi)
        
        p[e] = cx[i][j]/np.sqrt(cx[i][i]*cx[j][j])
    
    plt.close()
    plt.figure()
    
    plt.plot(t['rapo'], p, 'ko')

def allstream_2d(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True, relative=False):
    """Compare 2D constraints between all streams"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    names = get_done()
    N = len(names)
    
    # plot setup
    ncol = np.int64(np.ceil(np.sqrt(N)))
    nrow = np.int64(np.ceil(N/ncol))
    w_ = 8
    h_ = 1.1 * w_*nrow/ncol
    
    alpha = 1
    lw = 2
    frac = [0.8, 0.5, 0.2]
    
    # parameter pairs
    paramids = [8, 11, 12, 13, 14]
    all_comb = list(itertools.combinations(paramids, 2))
    comb = sorted(list(set(all_comb)))
    Ncomb = len(comb)
    #print(comb)

    pp = PdfPages('../plots/allstreams_2d_{}_a{:1d}_{}_r{:1d}.pdf'.format(errmode, align, vlabel, relative))
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

        plt.tight_layout(h_pad=0, w_pad=0)
        pp.savefig(fig)
    pp.close()
        

# circular velocity
def pder_vc(x, p=[pparams_fid[j] for j in [0,1,2,3,4,5,6,8,10]], components=['bary', 'halo']):
    """"""
    N = np.size(x)
    
    # components
    if 'bary' in components:
        bulge = np.array([G*x*(x+p[1])**-2, -2*G*p[0]*x*(x+p[1])**-3])
        aux = p[3] + p[4]
        disk = np.array([G*x**2*(x**2 + aux**2)**-1.5, -3*G*p[2]*x**2*aux*(x**2 + aux**2)**-2.5, -3*G*p[2]*x**2*aux*(x**2 + aux**2)**-2.5])
        nfw = np.array([2*p[5]*(p[6]/x*np.log(1+x.value/p[6].value) - (1+x.value/p[6].value)**-1), p[5]**2*(np.log(1+x.value/p[6].value)/x - (x+p[6])**-1 - x*(x+p[6])**-2), np.zeros(N), np.zeros(N)])

        pder = np.vstack([bulge, disk, nfw])
    else:
        pder = np.array([2*p[0]*(p[1]/x*np.log(1+x.value/p[1].value) - (1+x.value/p[1].value)**-1), p[0]**2*(np.log(1+x.value/p[1].value)/x - (x+p[1])**-1 - x*(x+p[1])**-2), np.zeros(N), np.zeros(N)])
        
    return pder

def delta_vc_vec(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', component='all', j=0, align=True, d=200, Nb=1000, fast=False, scale=False, ascale=False):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    
    names = get_done()
    labels = full_names()
    colors = {x: mpl.cm.bone(e/len(names)) for e, x in enumerate(names)}
    #colors = {'gd1': mpl.cm.bone(0), 'atlas': mpl.cm.bone(0.5), 'tri': mpl.cm.bone(0.8)}
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    for name in names:
        # read in full inverse CRB for stream modeling
        fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        cxi = fm['cxi']
        if fast:
            cx = np.linalg.inv(cxi)
        else:
            cx = stable_inverse(cxi)
        
        # choose the appropriate components:
        Nprog, Nbary, Nhalo, Ndipole, Nquad, Npoint = [6, 5, 4, 3, 5, 1]
        if 'progenitor' not in vary:
            Nprog = 0
        nstart = {'bary': Nprog, 'halo': Nprog + Nbary, 'dipole': Nprog + Nbary + Nhalo, 'quad': Nprog + Nbary + Nhalo + Ndipole, 'all': Nprog, 'point': 0}
        nend = {'bary': Nprog + Nbary, 'halo': Nprog + Nbary + Nhalo, 'dipole': Nprog + Nbary + Nhalo + Ndipole, 'quad': Nprog + Nbary + Nhalo + Ndipole + Nquad, 'all': np.shape(cx)[0], 'point': 1}
        
        if 'progenitor' not in vary:
            nstart['dipole'] = Npoint
            nend['dipole'] = Npoint + Ndipole
        
        if component in ['bary', 'halo', 'dipole', 'quad', 'point']:
            components = [component]
        else:
            components = [x for x in vary if x!='progenitor']
        cq = cx[nstart[component]:nend[component], nstart[component]:nend[component]]
        Npot = np.shape(cq)[0]
        
        if fast:
            cqi = np.linalg.inv(cq)
        else:
            cqi = stable_inverse(cq)
        
        if scale:
            dp_opt = read_optimal_step(name, vary)
            dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
            
            scale_vec = np.array([x.value for x in dp[nstart[component]:nend[component]]])
            scale_mat = np.outer(scale_vec, scale_vec)
            cqi *= scale_mat
        
        x = np.linspace(0.01, d, Nb)*u.kpc
        Npix = np.size(x)
        derf = np.transpose(pder_vc(x, components=components))
        
        ca = np.matmul(derf, np.matmul(cq, derf.T))
        
        Nx = Npot
        Nw = Nb
        vals, vecs = la.eigh(ca, eigvals=(Nw - Nx - 2, Nw - 1))

        if j==0:
            vcomb = np.sqrt(np.sum(vecs**2*vals, axis=1))
            #label = ['($\Sigma$ Eigval $\\times$ Eigvec$^2$ $a_{}$'.format(x)+')$^{1/2}$' for x in ['X', 'Y', 'Z']]
        else:
            vcomb = vecs[:,j]*np.sqrt(vals[j])
            #label = ['Eig {} $a_{}$'.format(np.abs(j), x) for x in ['X', 'Y', 'Z']]

        mcomb = (vcomb*u.km**2*u.s**-2 * x / G).to(u.Msun)
        vc_true = vcirc_potential(x, pparams=pparams_fid)
        
        # relate to orbit
        orbit = stream_orbit(name=name)
        r = np.linalg.norm(orbit['x'].to(u.kpc), axis=0)
        rmin = np.min(r)
        rmax = np.max(r)
        rcur = r[0]
        r0 = r[-1]
        print(name, rcur, r0)
        
        e = (rmax - rmin)/(rmax + rmin)
        l = np.cross(orbit['x'].to(u.kpc), orbit['v'].to(u.km/u.s), axisa=0, axisb=0)
        
        p, Np = period(name)
        
        np.savez('../data/crb/vcirc_{:s}{:1d}_{:s}_a{:1d}_{:s}'.format(errmode, Ndim, name, align, vlabel), dvc=np.sqrt(vcomb), vc=vc_true.value, r=x.value, rperi=rmin, rapo=rmax, rcur=rcur, r0=r0, ecc=e, l=l, p=p, Np=Np)

        if ascale:
            x = x * rmax**-1
            #x = x * rcur**-1
        
        # plot
        plt.sca(ax[0])
        plt.plot(x, np.sqrt(vcomb), '-', lw=3, color=colors[name], label=labels[name])
        #plt.plot(x, vc_true, 'r-')
        
        plt.sca(ax[1])
        plt.plot(x, np.sqrt(vcomb)/vc_true, '-', lw=3, color=colors[name], label=labels[name])
        #plt.plot(x, mcomb, '-', lw=3, color=colors[name], label=labels[name])
    
    plt.sca(ax[0])
    if ascale:
        plt.xlim(0,5)
        plt.xlabel('r/r$_{apo}$')
    else:
        plt.xlabel('r (kpc)')
    plt.ylabel('$\Delta$ $V_c$ (km s$^{-1}$)')
    #plt.ylim(0, 100)
    
    plt.sca(ax[1])
    plt.legend(loc=1, frameon=True, handlelength=1, fontsize='small')
    if ascale:
        plt.xlim(0,5)
        plt.xlabel('r/r$_{apo}$')
    else:
        plt.xlabel('r (kpc)')
    plt.ylabel('$\Delta$ $V_c$ / $V_c$')
    #plt.ylabel('$\Delta$ $M_{enc}$ ($M_\odot$)')
    #plt.ylim(0, 1e11)
    
    plt.tight_layout()
    plt.savefig('../plots/vc_r_summary_apo{:d}.pdf'.format(ascale))

def delta_vc_correlations(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', component='all', j=0, align=True, d=200, Nb=1000, r=False, fast=False, scale=False):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    elabel = ''
    ylabel = 'min ($\Delta$ $V_c$ / $V_c$)'
    if r:
        ylabel = 'r(min($\Delta$ $V_c$ / $V_c$)) (kpc)'
        elabel = 'r'
    
    names = get_done()
    labels = full_names()
    colors = {x: mpl.cm.bone(e/len(names)) for e, x in enumerate(names)}
    
    plt.close()
    fig, ax = plt.subplots(2,3,figsize=(15,9))
    
    for name in names:
        d = np.load('../data/crb/vcirc_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        rel_dvc = np.min(d['dvc'] / d['vc'])
        if r:
            idmin = np.argmin(d['dvc'] / d['vc'])
            rel_dvc = d['r'][idmin]
        
        mock = pickle.load(open('../data/mock_{}.params'.format(name), 'rb'))
        dlambda = np.max(mock['xi_range']) - np.min(mock['xi_range'])
        
        plt.sca(ax[0][0])
        if r:
            plt.plot(d['rapo'], d['rapo'], 'r.', zorder=0, lw=1.5)
        plt.plot(d['rapo'], rel_dvc, 'o', ms=10, color=colors[name], label=labels[name])
        
        plt.xlabel('$r_{apo}$ (kpc)')
        plt.ylabel(ylabel)
        
        plt.sca(ax[0][1])
        #plt.plot(d['rcur']/d['rapo'], rel_dvc, 'o', ms=10, color=colors[name])
        if r:
            plt.plot(d['rapo'], d['rapo'], 'r.', zorder=0, lw=1.5)
        plt.plot(d['rcur'], rel_dvc, 'o', ms=10, color=colors[name])
        #plt.plot(d['r0'], rel_dvc, 'ro')
        
        plt.xlabel('$r_{current}$')
        plt.ylabel(ylabel)
        
        plt.sca(ax[0][2])
        ecc = np.sqrt(1 - (d['rperi']/d['rapo'])**2)
        ecc = d['ecc']
        plt.plot(ecc, rel_dvc, 'o', ms=10, color=colors[name], label=labels[name])
        
        plt.xlabel('Eccentricity')
        plt.ylabel(ylabel)
        
        plt.sca(ax[1][0])
        plt.plot(np.median(np.abs(d['l'][:,2])/np.linalg.norm(d['l'], axis=1)), rel_dvc, 'o', ms=10, color=colors[name])
        
        plt.xlabel('|L_z|/|L|')
        plt.ylabel(ylabel)
        
        plt.sca(ax[1][1])
        plt.plot(d['Np'], rel_dvc, 'o', ms=10, color=colors[name])
        
        #plt.xlabel('$r_{peri}$ (kpc)')
        plt.xlabel('Completed periods')
        plt.ylabel(ylabel)

        plt.sca(ax[1][2])
        plt.plot(dlambda, rel_dvc, 'o', ms=10, color=colors[name])
        
        plt.xlabel('$\Delta$ $\\xi$ (deg)')
        plt.ylabel(ylabel)
    
    plt.sca(ax[0][2])
    plt.legend(fontsize='small', handlelength=0.1)
    
    plt.tight_layout()
    plt.savefig('../plots/delta_vc{}_correlations.pdf'.format(elabel))

def collate_orbit(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True):
    """Store all of the properties on streams"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    
    names = get_done()
    N = len(names)
    Nmax = len(max(names, key=len))
    
    tname = np.chararray(N, itemsize=Nmax)
    vcmin = np.empty(N)
    r_vcmin = np.empty(N)
    
    Labs = np.empty((N,3))
    lx = np.empty(N)
    ly = np.empty(N)
    lz = np.empty(N)
    Lmod = np.empty(N)
    
    period = np.empty(N)
    Nperiod = np.empty(N)
    ecc = np.empty(N)
    rperi = np.empty(N)
    rapo = np.empty(N)
    rcur = np.empty(N)
    length = np.empty(N)
    
    for e, name in enumerate(names[:]):
        d = np.load('../data/crb/vcirc_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        idmin = np.argmin(d['dvc'] / d['vc'])

        mock = pickle.load(open('../data/mock_{}.params'.format(name), 'rb'))
        dlambda = np.max(mock['xi_range']) - np.min(mock['xi_range'])
        
        tname[e] = name
        vcmin[e] = (d['dvc'] / d['vc'])[idmin]
        r_vcmin[e] = d['r'][idmin]
        
        if e==0:
            Nr = np.size(d['r'])
            dvc = np.empty((N, Nr))
            vc = np.empty((N, Nr))
            r = np.empty((N, Nr))
        
        dvc[e] = d['dvc']
        vc[e] = d['dvc'] / d['vc']
        r[e] = d['r']
        
        Labs[e] = np.median(np.abs(d['l']), axis=0)
        Lmod[e] = np.median(np.linalg.norm(d['l'], axis=1))
        lx[e] = np.abs(np.median(d['l'][:,0]/np.linalg.norm(d['l'], axis=1)))
        ly[e] = np.abs(np.median(d['l'][:,1]/np.linalg.norm(d['l'], axis=1)))
        lz[e] = np.abs(np.median(d['l'][:,2]/np.linalg.norm(d['l'], axis=1)))
        
        period[e] = d['p']
        Nperiod[e] = d['Np']
        ecc[e] = d['ecc']
        rperi[e] = d['rperi']
        rapo[e] = d['rapo']
        rcur[e] = d['rcur']
        length[e] = dlambda
    
    t = Table([tname, vcmin, r_vcmin, dvc, vc, r, Labs, Lmod, lx, ly, lz, period, Nperiod, length, ecc, rperi, rapo, rcur], names=('name', 'vcmin', 'rmin', 'dvc', 'vc', 'r', 'Labs', 'Lmod', 'lx', 'ly', 'lz', 'period', 'Nperiod', 'length', 'ecc', 'rperi', 'rapo', 'rcur'))
    t.pprint()
    t.write('../data/crb/vc_orbital_summary.fits', overwrite=True)

# radial acceleration
def ar_r(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True, Nsight=1, seed=39):
    """Calculate precision in radial acceleration as a function of galactocentric radius"""
    
    np.random.seed(seed)
    pid, dp_fid, vlabel = get_varied_pars(vary)
    components = [c for c in vary if c!='progenitor']
    
    names = get_done()
    N = len(names)
    Nmax = len(max(names, key=len))
    
    tname = np.chararray(N, itemsize=Nmax)
    armin = np.empty((N, Nsight))
    r_armin = np.empty((N, Nsight))
    
    Labs = np.empty((N,3))
    lx = np.empty(N)
    ly = np.empty(N)
    lz = np.empty(N)
    Lmod = np.empty(N)
    
    period_ = np.empty(N)
    Nperiod = np.empty(N)
    ecc = np.empty(N)
    rperi = np.empty(N)
    rapo = np.empty(N)
    rcur = np.empty(N)
    length = np.empty(N)
    
    Npix = 300
    r = np.linspace(0.1, 200, Npix)
    dar = np.empty((N, Nsight, Npix))
    ar = np.empty((N, Nsight, Npix))
    rall = np.empty((N, Nsight, Npix))

    plt.close()
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    
    for e, name in enumerate(names[:]):
        # read in full inverse CRB for stream modeling
        fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        cxi = fm['cxi']
        cx = stable_inverse(cxi)
        
        cq = cx[6:,6:]
        Npot = np.shape(cq)[0]
        
        # relate to orbit
        orbit = stream_orbit(name=name)
        ro = np.linalg.norm(orbit['x'].to(u.kpc), axis=0)
        rmin = np.min(ro)
        rmax = np.max(ro)
        rcur_ = ro[0]
        r0 = ro[-1]
        
        e_ = (rmax - rmin)/(rmax + rmin)
        l = np.cross(orbit['x'].to(u.kpc), orbit['v'].to(u.km/u.s), axisa=0, axisb=0)
        
        p, Np = period(name)
        mock = pickle.load(open('../data/mock_{}.params'.format(name), 'rb'))
        
        for s in range(Nsight):
            if Nsight==1:
            # single sightline
                x0 = mock['x0']
                xeq = coord.SkyCoord(ra=x0[0], dec=x0[1], distance=x0[2])
                xg = xeq.transform_to(coord.Galactocentric)

                rg = np.linalg.norm(np.array([xg.x.value, xg.y.value, xg.z.value]))
                theta = np.arccos(xg.z.value/rg)
                phi = np.arctan2(xg.y.value, xg.x.value)
            else:
                u_ = np.random.random(1)
                v_ = np.random.random(1)
                theta = np.arccos(2*u_ - 1)
                phi = 2 * np.pi * v_
            
            xin = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)]).T
            
            arad_pix = np.empty((Npix, 1))
            af = np.empty(Npix)
            derf = np.empty((Npix, Npot))
            
            for i in range(Npix):
                xi = xin[i]*u.kpc
                a = acc_rad(xi, components=components)
                af[i] = a
                
                dadq = apder_rad(xi, components=components)
                derf[i] = dadq
            
            ca = np.matmul(derf, np.matmul(cq, derf.T))
            
            Nx = Npot
            Nw = Npix
            vals, vecs = la.eigh(ca, eigvals=(Nw - Nx - 2, Nw - 1))
            vcomb = np.sqrt(np.sum(vecs**2*vals, axis=1))
            
            # store
            idmin = np.argmin(vcomb / np.abs(af))

            armin[e][s] = (vcomb / np.abs(af))[idmin]
            r_armin[e][s] = r[idmin]
            
            dar[e][s] = vcomb
            ar[e][s] = vcomb / np.abs(af)
            rall[e][s] = r
        
        dlambda = np.max(mock['xi_range']) - np.min(mock['xi_range'])
        tname[e] = name

        Labs[e] = np.median(np.abs(l), axis=0)
        Lmod[e] = np.median(np.linalg.norm(l, axis=1))
        lx[e] = np.abs(np.median(l[:,0]/np.linalg.norm(l, axis=1)))
        ly[e] = np.abs(np.median(l[:,1]/np.linalg.norm(l, axis=1)))
        lz[e] = np.abs(np.median(l[:,2]/np.linalg.norm(l, axis=1)))
        
        period_[e] = p
        Nperiod[e] = Np
        ecc[e] = e_
        rperi[e] = rmin
        rapo[e] = rmax
        rcur[e] = rcur_
        length[e] = dlambda
    
    t = Table([tname, armin, r_armin, dar, ar, rall, Labs, Lmod, lx, ly, lz, period_, Nperiod, length, ecc, rperi, rapo, rcur], names=('name', 'armin', 'rmin', 'dar', 'ar', 'r', 'Labs', 'Lmod', 'lx', 'ly', 'lz', 'period', 'Nperiod', 'length', 'ecc', 'rperi', 'rapo', 'rcur'))
    t.pprint()
    t.write('../data/crb/ar_orbital_summary_{}_sight{:d}.fits'.format(vlabel, Nsight), overwrite=True)
    
    plt.tight_layout()

def plot_ar(current=False, vary=['progenitor', 'bary', 'halo'], Nsight=1):
    """Explore constraints on radial acceleration, along the progenitor line"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    t = Table.read('../data/crb/ar_orbital_summary_{}_sight{:d}.fits'.format(vlabel, Nsight))
    N = len(t)
    fapo = t['rapo']/np.max(t['rapo'])
    fapo = t['rapo']/100
    flen = t['length']/(np.max(t['length']) + 10)
    fcolor = fapo
    
    plt.close()
    fig, ax = plt.subplots(1, 4, figsize=(20,5))
    
    for i in range(N):
        color = mpl.cm.bone(fcolor[i])
        lw = flen[i] * 5
        
        plt.sca(ax[0])
        plt.plot(t['r'][i][0], t['ar'][i][1], '-', color=color, lw=lw)
        
    plt.xlabel('R (kpc)')
    plt.ylabel('$\Delta$ $a_r$ / $a_r$')
    plt.ylim(0, 3.5)
    
    armin = np.median(t['armin'], axis=1)
    armin_err = 0.5 * (np.percentile(t['armin'], 84, axis=1) - np.percentile(t['armin'], 16, axis=1))
    rmin = np.median(t['rmin'], axis=1)
    rmin_err = 0.5 * (np.percentile(t['rmin'], 84, axis=1) - np.percentile(t['rmin'], 16, axis=1))
    plt.sca(ax[1])
    plt.scatter(t['length'], armin, c=fcolor, cmap='bone', vmin=0, vmax=1)
    plt.errorbar(t['length'], armin, yerr=armin_err, color='k', fmt='none', zorder=0)
    
    plt.xlabel('Length (deg)')
    plt.ylabel('min $\Delta$ $a_r$')
    plt.ylim(0, 3.5)
    
    plt.sca(ax[2])
    a = np.linspace(0,90,100)
    plt.plot(a, a, 'k-')
    #plt.plot(a, 2*a, 'k--')
    #plt.plot(a, 3*a, 'k:')
    plt.scatter(t['rcur'], rmin, c=fcolor, cmap='bone', vmin=0, vmax=1)
    plt.errorbar(t['rcur'], rmin, yerr=rmin_err, color='k', fmt='none', zorder=0)
    plt.xlabel('$R_{cur}$ (kpc)')
    plt.ylabel('$R_{min}$ (kpc)')
    
    #for i in range(len(t)):
        #plt.text(t['rcur'][i], rmin[i]+5, t['name'][i], fontsize='small')
    
    plt.xlim(0,90)
    plt.ylim(0,90)
    
    plt.sca(ax[3])
    a = np.linspace(0,90,100)
    plt.plot(a, a, 'k-')
    #plt.plot(a, 2*a, 'k--')
    #plt.plot(a, 3*a, 'k:')
    plt.scatter(t['rapo'], rmin, c=fcolor, cmap='bone', vmin=0, vmax=1)
    plt.errorbar(t['rapo'], rmin, yerr=rmin_err, color='k', fmt='none', zorder=0)
    plt.xlabel('$R_{apo}$ (kpc)')
    plt.ylabel('$R_{min}$ (kpc)')
    
    plt.xlim(0,90)
    plt.ylim(0,90)
    
    plt.tight_layout()
    plt.savefig('../plots/ar_crb_{}_sight{:d}.pdf'.format(vlabel, Nsight))
    
    # save stream constraints
    tout = Table([t['name'], t['rapo'], t['rcur'], t['length'], rmin, rmin_err, armin, armin_err], names=('name', 'rapo', 'rcur', 'length', 'rmin', 'rmin_err', 'armin', 'armin_err'))
    tout.write('../data/ar_constraints_{}_sight{}.fits'.format(vlabel, Nsight), overwrite=True)

def plot_all_ar(Nsight=50):
    """Explore constraints on radial acceleration, along the progenitor line"""

    alist = [0.2, 0.4, 0.7, 1]
    mslist = [11, 9, 7, 5]
    lwlist = [8, 6, 4, 2]
    fc = [0.8, 0.6, 0.4, 0.2]
    vlist = [['progenitor', 'bary', 'halo'], ['progenitor', 'bary', 'halo', 'dipole'], ['progenitor', 'bary', 'halo', 'dipole', 'quad'], ['progenitor', 'bary', 'halo', 'dipole', 'quad', 'octu']]
    labels = ['Fiducial Galaxy', '+ dipole', '++ quadrupole', '+++ octupole']
    
    alist = [0.2, 0.55, 1]
    #mslist = [11, 8, 5]
    mslist = [13, 10, 7]
    #lwlist = [8, 5, 2]
    lwlist = [9, 6, 3]
    fc = [0.8, 0.5, 0.2]
    vlist = [['progenitor', 'bary', 'halo'], ['progenitor', 'bary', 'halo', 'dipole', 'quad'], ['progenitor', 'bary', 'halo', 'dipole', 'quad', 'octu']]
    labels = ['Fiducial Galaxy', '++ quadrupole', '+++ octupole']
    
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(13.5,4.5))
    
    for e, vary in enumerate(vlist):
        pid, dp_fid, vlabel = get_varied_pars(vary)
        t = Table.read('../data/crb/ar_orbital_summary_{}_sight{:d}.fits'.format(vlabel, Nsight))
        N = len(t)
        
        color = mpl.cm.viridis(fc[e])
        lw = lwlist[e]
        ms = mslist[e]
        alpha = alist[e]
        
        plt.sca(ax[0])
        for i in range(0,5,4):
            plt.plot(t['r'][i][0], t['ar'][i][1], '-', color=color, lw=lw, alpha=alpha)
            
        plt.xlabel('r (kpc)')
        plt.ylabel('$\Delta$ $a_r$ / $a_r$')
        plt.ylim(0, 3.5)
        
        armin = np.median(t['armin'], axis=1)
        armin_err = 0.5 * (np.percentile(t['armin'], 84, axis=1) - np.percentile(t['armin'], 16, axis=1))
        rmin = np.median(t['rmin'], axis=1)
        rmin_err = 0.5 * (np.percentile(t['rmin'], 84, axis=1) - np.percentile(t['rmin'], 16, axis=1))
        
        # fit exponential
        p = np.polyfit(t['length'], np.log(armin), 1)
        print(1/p[0], np.exp(p[1]))
        poly = np.poly1d(p)
        x_ = np.linspace(np.min(t['length']), np.max(t['length']), 100)
        y_ = poly(x_)
        
        plt.sca(ax[1])
        plt.plot(x_, np.exp(y_), '-', color=color, alpha=alpha, lw=lw, label='')
        plt.plot(t['length'], armin, 'o', color=color, ms=ms, alpha=alpha, label=labels[e])
        plt.errorbar(t['length'], armin, yerr=armin_err, color=color, fmt='none', zorder=0, alpha=alpha)
        #plt.plot(t['length'], np.log(armin), 'o', color=color, ms=ms, alpha=alpha, label=labels[e])
        #plt.errorbar(t['length'], np.log(armin), yerr=np.log(armin_err), color=color, fmt='none', zorder=0, alpha=alpha)
        
        if e==len(vlist)-1:
            plt.legend(loc=1, fontsize='small', handlelength=0.5, frameon=False)
        plt.xlabel('Stream length (deg)')
        plt.ylabel('min $\Delta$ $a_r$')
        plt.ylim(0, 3.5)
        
        plt.sca(ax[2])
        a = np.linspace(0,90,100)
        plt.plot(a, a, 'k-', alpha=0.4)
        plt.plot(t['rcur'], rmin, 'o', color=color, ms=ms, alpha=alpha)
        plt.errorbar(t['rcur'], rmin, yerr=rmin_err, color=color, fmt='none', zorder=0, alpha=alpha)
        plt.xlabel('$R_{cur}$ (kpc)')
        plt.ylabel('$R_{min}$ (kpc)')

        plt.xlim(0,90)
        plt.ylim(0,90)
        
        #plt.sca(ax[3])
        #a = np.linspace(0,90,100)
        #plt.plot(a, a, 'k-')
        #plt.plot(t['rapo'], rmin, 'o', color=color, ms=ms, alpha=alpha)
        #plt.errorbar(t['rapo'], rmin, yerr=rmin_err, color=color, fmt='none', zorder=0, alpha=alpha)
        #plt.xlabel('$R_{apo}$ (kpc)')
        #plt.ylabel('$R_{min}$ (kpc)')
        
        #plt.xlim(0,90)
        #plt.ylim(0,90)
    
    plt.tight_layout()
    plt.savefig('../plots/ar_crb_all_sight{:d}.pdf'.format(Nsight))
    plt.savefig('../paper/ar_crb_all.pdf')

def ar_multi(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', align=True, Nsight=1, seed=39, verbose=True):
    """Calculate precision in radial acceleration as a function of galactocentric radius for multiple streams"""
    
    np.random.seed(seed)
    pid, dp_fid, vlabel = get_varied_pars(vary)
    components = [c for c in vary if c!='progenitor']
    Npar = len(pid)
    
    names = get_done()
    N = len(names)
    Nmax = len(max(names, key=len))
    
    armin = np.empty((N, Nsight))
    r_armin = np.empty((N, Nsight))
    
    Npix = 300
    r = np.linspace(0.1, 200, Npix)
    dar = np.empty((N, Nsight, Npix))
    ar = np.empty((N, Nsight, Npix))
    rall = np.empty((N, Nsight, Npix))

    plt.close()
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    plt.sca(ax)
    
    for k in range(N):
        names_in = [names[x] for x in range(k+1)]
        if verbose: print(k, names_in)
        cxi_all = np.zeros((Npar, Npar))
        for e, name in enumerate(names_in):
            # read in full inverse CRB for stream modeling
            fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
            cxi = fm['cxi']
            cxi_all = cxi_all + cxi
        
        cx_all = stable_inverse(cxi_all)
        cq = cx_all[6:,6:]
        Npot = np.shape(cq)[0]
        
        for s in range(Nsight):
            if Nsight==1:
                # single sightline
                mock = pickle.load(open('../data/mock_{}.params'.format('gd1'), 'rb'))
                x0 = mock['x0']
                xeq = coord.SkyCoord(ra=x0[0], dec=x0[1], distance=x0[2])
                xg = xeq.transform_to(coord.Galactocentric)

                rg = np.linalg.norm(np.array([xg.x.value, xg.y.value, xg.z.value]))
                theta = np.arccos(xg.z.value/rg)
                phi = np.arctan2(xg.y.value, xg.x.value)
            else:
                u_ = np.random.random(1)
                v_ = np.random.random(1)
                theta = np.arccos(2*u_ - 1)
                phi = 2 * np.pi * v_
            
            xin = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)]).T
            
            arad_pix = np.empty((Npix, 1))
            af = np.empty(Npix)
            derf = np.empty((Npix, Npot))
            
            for i in range(Npix):
                xi = xin[i]*u.kpc
                a = acc_rad(xi, components=components)
                af[i] = a
                
                dadq = apder_rad(xi, components=components)
                derf[i] = dadq
            
            ca = np.matmul(derf, np.matmul(cq, derf.T))
            
            Nx = Npot
            Nw = Npix
            vals, vecs = la.eigh(ca, eigvals=(Nw - Nx - 2, Nw - 1))
            vcomb = np.sqrt(np.sum(vecs**2*vals, axis=1))
            
            # store
            idmin = np.argmin(vcomb / np.abs(af))

            armin[k][s] = (vcomb / np.abs(af))[idmin]
            r_armin[k][s] = r[idmin]
            
            dar[k][s] = vcomb
            ar[k][s] = vcomb / np.abs(af)
            rall[k][s] = r
            
            plt.plot(rall[k][s], ar[k][s]*100, '-', color=mpl.cm.viridis_r(k/12.), lw=2)
    
    t = Table([armin, r_armin, dar, ar, rall], names=('armin', 'rmin', 'dar', 'ar', 'r'))
    t.pprint()
    t.write('../data/crb/ar_multistream{}_{}_sight{:d}.fits'.format(N, vlabel, Nsight), overwrite=True)
    
    plt.xlabel('r (kpc)')
    plt.ylabel('$\Delta$ $a_r$ / $a_r$ (%)')
    plt.ylim(0,100)
    
    # add custom colorbar
    sm = plt.cm.ScalarMappable(cmap=mpl.cm.viridis_r, norm=plt.Normalize(vmin=1, vmax=12))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='4%', pad=0.05)
    
    #cb = fig.colorbar(sm, ax=cax, pad=0.1, aspect=40, ticks=np.arange(1,13,3))
    cb = plt.colorbar(sm, cax=cax, ticks=np.arange(1,13,3))
    cb.set_label('Number of streams')
    
    plt.tight_layout()
    plt.savefig('../plots/ar_multistream{}_{}_sight{:d}.png'.format(N, vlabel, Nsight))

# flattening
def delta_q(q='x', Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', j=0, align=True, fast=False, scale=False):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    
    kq = {'x': 0, 'z': 2}
    iq = {'x': 2, 'z': 3}
    
    labelq = {'x': '$_x$', 'z': '$_z$'}
    
    component = 'halo'
    pparams0 = pparams_fid
    pid_comp, dp_fid2, vlabel2 = get_varied_pars(component)
    Np = len(pid_comp)
    pid_crb = myutils.wherein(np.array(pid), np.array(pid_comp))
    
    names = get_done()
    labels = full_names()
    colors = {x: mpl.cm.bone(e/len(names)) for e, x in enumerate(names)}
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    for name in names:
    #for n in [-1,]:
        # read in full inverse CRB for stream modeling
        fm = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, name, align, vlabel))
        cxi = fm['cxi']
        if fast:
            cx = np.linalg.inv(cxi)
        else:
            cx = stable_inverse(cxi)
        
        crb_all = np.sqrt(np.diag(cx))
        crb = [crb_all[pid_crb[i]] for i in range(Np)]
        crb_frac = [crb_all[pid_crb[i]]/pparams0[pid_comp[i]].value for i in range(Np)]
        
        delta_q = crb[iq[q]]
        
        ## choose the appropriate components:
        #Nprog, Nbary, Nhalo, Ndipole, Nquad, Npoint = [6, 5, 4, 3, 5, 1]
        #if 'progenitor' not in vary:
            #Nprog = 0
        #nstart = {'bary': Nprog, 'halo': Nprog + Nbary, 'dipole': Nprog + Nbary + Nhalo, 'quad': Nprog + Nbary + Nhalo + Ndipole, 'all': Nprog, 'point': 0}
        #nend = {'bary': Nprog + Nbary, 'halo': Nprog + Nbary + Nhalo, 'dipole': Nprog + Nbary + Nhalo + Ndipole, 'quad': Nprog + Nbary + Nhalo + Ndipole + Nquad, 'all': np.shape(cx)[0], 'point': 1}
        
        #if 'progenitor' not in vary:
            #nstart['dipole'] = Npoint
            #nend['dipole'] = Npoint + Ndipole
        
        #if component in ['bary', 'halo', 'dipole', 'quad', 'point']:
            #components = [component]
        #else:
            #components = [x for x in vary if x!='progenitor']
        #cq = cx[nstart[component]:nend[component], nstart[component]:nend[component]]
        #if ('progenitor' not in vary) & ('bary' not in vary):
            #cq = cx
        #Npot = np.shape(cq)[0]
        
        #if scale:
            #dp_opt = read_optimal_step(n, vary)
            #dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
            #dp_unit = unity_scale(dp)
            #scale_vec = np.array([x.value for x in dp_unit[nstart[component]:nend[component]]])
            #scale_mat = np.outer(scale_vec, scale_vec)
            #cqi /= scale_mat
        
        #delta_q = np.sqrt(cq[iq[q], iq[q]])

        # relate to orbit
        orbit = stream_orbit(name=name)
        r = np.linalg.norm(orbit['x'].to(u.kpc), axis=0)
        rmin = np.min(r)
        rmax = np.max(r)
        e = (rmax - rmin)/(rmax + rmin)
        e = rmin/rmax
        
        l = np.cross(orbit['x'].to(u.kpc), orbit['v'].to(u.km/u.s), axisa=0, axisb=0)
        ltheta = np.median(l[:,kq[q]]/np.linalg.norm(l, axis=1))
        langle = np.degrees(np.arccos(ltheta))
        sigltheta = np.std(l[:,kq[q]]/np.linalg.norm(l, axis=1))
        
        plt.sca(ax[0])
        plt.plot(e, delta_q, 'o', color=colors[name], label=labels[name])
        
        plt.sca(ax[1])
        plt.plot(sigltheta, delta_q, 'o', color=colors[name], label=labels[name])

        plt.sca(ax[2])
        plt.plot(np.abs(ltheta), delta_q, 'o', color=colors[name], label=labels[name])
    
    plt.sca(ax[0])
    plt.legend(frameon=False, handlelength=1, fontsize='small')
    plt.xlabel('Eccentricity')
    plt.ylabel('$\Delta$ q{}'.format(labelq[q]))
    plt.xlim(0,1)
    #plt.ylim(0, 1e11)
    
    plt.sca(ax[1])
    plt.xlabel('$\sigma$ L{}'.format(labelq[q]) + ' (kpc km s$^{-1}$)')
    plt.ylabel('$\Delta$ q{}'.format(labelq[q]))

    plt.sca(ax[2])
    plt.xlabel('|L{}| / |L|'.format(labelq[q]))
    plt.ylabel('$\Delta$ q{}'.format(labelq[q]))
    
    plt.tight_layout()
    plt.savefig('../plots/delta_q{}.pdf'.format(q))


###
# multiple streams
###

def pairs_pdf(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', component='halo', align=True, summary=False):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    
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
    punits = [' ({})'.format(x) if len(x) else '' for x in units]
    params = ['$\Delta$ {}{}'.format(x, y) for x,y in zip(plabels, punits)]
    
    done = get_done()
    N = len(done)
    pp = PdfPages('../plots/corner_pairs_{:s}{:1d}_a{:1d}_{:s}_{:s}_{:d}.pdf'.format(errmode, Ndim, align, vlabel, component, summary))
    fig = None
    ax = None
    
    for i in range(N):
        di = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, done[i], align, vlabel))
        cxi_i = di['cxi']
        for j in range(i+1,N):
            dj = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, done[j], align, vlabel))
            cxi_j = dj['cxi']
            
            cxi = cxi_i + cxi_j
            cx = stable_inverse(cxi)
            cx_i = stable_inverse(cxi_i)
            cx_j = stable_inverse(cxi_j)
            
            # select component of the parameter space
            cq = cx[nstart[component]:nend[component], nstart[component]:nend[component]]
            cq_i = cx_i[nstart[component]:nend[component], nstart[component]:nend[component]]
            cq_j = cx_j[nstart[component]:nend[component], nstart[component]:nend[component]]
            
            Nvar = np.shape(cq)[0]
            
            print(done[i], done[j])
            print(np.sqrt(np.diag(cq)))
            print(np.sqrt(np.diag(cq_i)))
            print(np.sqrt(np.diag(cq_j)))
            
            if summary==False:
                fig = None
                ax = None
                
                # plot ellipses
                fig, ax = corner_ellipses(cq, fig=fig, ax=ax)
                fig, ax = corner_ellipses(cq_i, alpha=0.5, fig=fig, ax=ax)
                fig, ax = corner_ellipses(cq_j, alpha=0.5, fig=fig, ax=ax)
                
                # labels
                plt.title('{} & {}'.format(done[i], done[j]))
                
                for k in range(Nvar-1):
                    plt.sca(ax[-1][k])
                    plt.xlabel(params[k])
                    
                    plt.sca(ax[k][0])
                    plt.ylabel(params[k+1])
                pp.savefig(fig)
            else:
                fig, ax = corner_ellipses(cq, fig=fig, ax=ax, alpha=0.5)
    
    if summary:
        # labels
        for k in range(Nvar-1):
            plt.sca(ax[-1][k])
            plt.xlabel(params[k])
            
            plt.sca(ax[k][0])
            plt.ylabel(params[k+1])
        pp.savefig(fig)
    pp.close()

def multi_pdf(Nmulti=3, Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', component='halo', align=True):
    """Create a pdf with each page containing a corner plot with constraints on a given component of the model from multiple streams"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    Ntot = len(pid)
    
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
    punits = [' ({})'.format(x) if len(x) else '' for x in units]
    params = ['$\Delta$ {}{}'.format(x, y) for x,y in zip(plabels, punits)]
    Nvar = len(pid_comp)
    
    pparams0 = pparams_fid
    pparams_comp = [pparams0[x] for x in pid_comp]
    pparams_arr = np.array([x.value for x in pparams_comp])
    
    pp = PdfPages('../plots/corner_multi{:d}_{:s}{:1d}_a{:1d}_{:s}_{:s}.pdf'.format(Nmulti, errmode, Ndim, align, vlabel, component))
    fig = None
    ax = None
    
    done = get_done()
    N = len(done)
    
    if Nmulti>N:
        Nmulti = N

    t = np.arange(N, dtype=np.int64).tolist()
    all_comb = list(itertools.combinations(t, Nmulti))
    comb = sorted(list(set(all_comb)))
    Ncomb = len(comb)
    
    comb_all = np.ones((Ncomb, N)) * np.nan
    cx_all = np.empty((Ncomb, Nvar, Nvar))
    p_all = np.empty((Ncomb, Nvar))
    prel_all = np.empty((Ncomb, Nvar))
    
    for i in range(Ncomb):
        print(i, [done[i_] for i_ in comb[i]])
        cxi = np.zeros((Ntot, Ntot))
        fig = None
        ax = None
        for j in range(Nmulti):
            ind = comb[i][j]
            #print('{} '.format(done[ind]), end='')
            
            dj = np.load('../data/crb/cxi_{:s}{:1d}_{:s}_a{:1d}_{:s}.npz'.format(errmode, Ndim, done[ind], align, vlabel))
            cxi_ = dj['dxi']
            cxi = cxi + cxi_
            
            # select component of the parameter space
            cx_ = stable_inverse(cxi_)
            cq_ = cx_[nstart[component]:nend[component], nstart[component]:nend[component]]
            if Ncomb==1:
                np.save('../data/crb/cx_multi1_{:s}{:1d}_{:s}_a{:1d}_{:s}_{:s}'.format(errmode, Ndim, done[ind], align, vlabel, component), cq_)
            
            print(np.sqrt(np.diag(cq_)))
            
            fig, ax = corner_ellipses(cq_, alpha=0.5, fig=fig, ax=ax)

        cx = stable_inverse(cxi + dj['pxi'])
        cq = cx[nstart[component]:nend[component], nstart[component]:nend[component]]
        print(np.sqrt(np.diag(cq)))
        
        #label = '.'.join([done[comb[i][i_]] for i_ in range(Nmulti)])
        #np.save('../data/crb/cx_multi{:d}_{:s}{:1d}_{:s}_a{:1d}_{:s}_{:s}'.format(Nmulti, errmode, Ndim, label, align, vlabel, component), cq)

        cx_all[i] = cq
        p_all[i] = np.sqrt(np.diag(cq))
        prel_all[i] = p_all[i]/pparams_arr
        comb_all[i][:Nmulti] = np.array(comb[i])

        fig, ax = corner_ellipses(cq, fig=fig, ax=ax)
        
        # labels
        title = ' + '.join([done[comb[i][i_]] for i_ in range(Nmulti)])
        plt.suptitle(title)
        
        for k in range(Nvar-1):
            plt.sca(ax[-1][k])
            plt.xlabel(params[k])
            
            plt.sca(ax[k][0])
            plt.ylabel(params[k+1])
        
        plt.tight_layout(rect=(0,0,1,0.95))
        pp.savefig(fig)

    np.savez('../data/crb/cx_collate_multi{:d}_{:s}{:1d}_a{:1d}_{:s}_{:s}'.format(Nmulti, errmode, Ndim, align, vlabel, component), comb=comb_all, cx=cx_all, p=p_all, p_rel=prel_all)
    pp.close()

def collate(Ndim=6, vary=['progenitor', 'bary', 'halo'], errmode='fiducial', component='halo', align=True, Nmax=None):
    """"""
    done = get_done()
    N = len(done)
    if Nmax==None:
        Nmax = N
    t = np.arange(N, dtype=np.int64).tolist()
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    Ntot = len(pid)
    
    pparams0 = pparams_fid
    pid_comp, dp_fid2, vlabel2 = get_varied_pars(component)
    Np = len(pid_comp)
    pid_crb = myutils.wherein(np.array(pid), np.array(pid_comp))
    
    pparams_comp = [pparams0[x] for x in pid_comp]
    pparams_arr = np.array([x.value for x in pparams_comp])
    
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
    punits = [' ({})'.format(x) if len(x) else '' for x in units]
    params = ['$\Delta$ {}{}'.format(x, y) for x,y in zip(plabels, punits)]
    Nvar = len(pid_comp)

    for i in range(1, Nmax+1):
        Nmulti = i
        all_comb = list(itertools.combinations(t, Nmulti))
        comb = sorted(list(set(all_comb)))
        Ncomb = len(comb)
        
        comb_all = np.ones((Ncomb, N)) * np.nan
        cx_all = np.empty((Ncomb, Nvar, Nvar))
        p_all = np.empty((Ncomb, Nvar))
        prel_all = np.empty((Ncomb, Nvar))
        
        for j in range(Ncomb):
            label = '.'.join([done[comb[j][i_]] for i_ in range(Nmulti)])
            cx = np.load('../data/crb/cx_multi{:d}_{:s}{:1d}_{:s}_a{:1d}_{:s}_{:s}.npy'.format(Nmulti, errmode, Ndim, label, align, vlabel, component))
            
            cx_all[j] = cx
            p_all[j] = np.sqrt(np.diag(cx))
            prel_all[j] = p_all[j]/pparams_arr
            comb_all[j][:Nmulti] = np.array(comb[j])
        
        np.savez('../data/crb/cx_collate_multi{:d}_{:s}{:1d}_a{:1d}_{:s}_{:s}'.format(Nmulti, errmode, Ndim, align, vlabel, component), comb=comb_all, cx=cx_all, p=p_all, p_rel=prel_all)

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
            #print(ids_min)
            #idmin = np.argmin(p_all[:,k])
            #print(k, [done[np.int64(i_)] for i_ in comb[idmin][:Nmulti]])
        
    for k in range(Nvar):
        plt.sca(ax[k%ncol][np.int64(k/ncol)])
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        if relative:
            plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        
        plt.ylabel(params[k])
        
        if k==0:
            plt.legend(frameon=False, fontsize='small', loc=1)
    
        if k%ncol==nrow-1:
            plt.xlabel('Number of streams in a combination')
    
    plt.tight_layout()
    plt.savefig('../plots/nstream_improvement_{:s}{:1d}_a{:1d}_{:s}_{:s}_{:1d}.pdf'.format(errmode, Ndim, align, vlabel, component, relative))

def corner_ellipses(cx, dax=2, color='k', alpha=1, lw=2, fig=None, ax=None, autoscale=True, correlate=False):
    """Corner plot with ellipses given by an input matrix"""
    
    # assert square matrix
    Nvar = np.shape(cx)[0]
    if correlate:
        Npair = np.int64(Nvar*(Nvar - 1)/2)
        pcc = np.empty((3,Npair))
        k = 0
    
    if (np.any(fig)==None) | (np.any(ax)==None):
        plt.close()
        fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(dax*Nvar, dax*Nvar), sharex='col', sharey='row')
    
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar):
            plt.sca(ax[j-1][i])

            cx_2d = np.array([[cx[i][i], cx[i][j]], [cx[j][i], cx[j][j]]])
            
            if correlate:
                pcc[0,k] = i
                pcc[1,k] = j
                pcc[2,k] = cx[i][j]/np.sqrt(cx[i][i]*cx[j][j])
                k += 1
            
            w, v = np.linalg.eig(cx_2d)
            if np.all(np.isreal(v)):
                theta = np.degrees(np.arctan2(v[1][0], v[0][0]))
                width = np.sqrt(w[0])*2
                height = np.sqrt(w[1])*2
                
                e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=color, alpha=alpha, lw=lw)
                plt.gca().add_patch(e)
            
            if autoscale:
                plt.gca().autoscale_view()
    
    # turn off unused axes
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar-1):
            plt.sca(ax[i][j])
            plt.axis('off')
    
    plt.tight_layout()
    
    if correlate:
        return(fig, ax, pcc)
    else:
        return (fig, ax)


###
# compare observing modes
###

def comp_errmodes_old(n, errmodes=['binospec', 'fiducial', 'hectochelle'], Ndim=4, vary=['progenitor', 'bary', 'halo'], plot='halo', align=True, fast=False, scale=False):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    dp_opt = read_optimal_step(n, vary)
    dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
    plabels, units = get_parlabel(pid)
    params = ['$\Delta$' + x + '({})'.format(y) for x,y in zip(plabels, units)]
    
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    if plot=='halo':
        i0 = 11
        i1 = 15
    elif plot=='bary':
        i0 = 6
        i1 = 11
    elif plot=='progenitor':
        i0 = 0
        i1 = 6
    elif plot=='dipole':
        i0 = 15
        i1 = len(params)
    else:
        i0 = 0
        i1 = len(params)
    
    Nvar = i1 - i0
    params = params[i0:i1]
    if scale:
        dp_unit = unity_scale(dp)
        #print(dp_unit)
        dp_unit = dp_unit[i0:i1]
        pid = pid[i0:i1]
    
    #print(params, dp_unit, Nvar, len(pid), len(dp_unit))
    #label = ['RA, Dec, d', 'RA, Dec, d, $V_r$', 'RA, Dec, d, $V_r$, $\mu_\\alpha$, $\mu_\delta$']
    label = errmodes

    plt.close()
    dax = 2
    fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(dax*Nvar, dax*Nvar), sharex='col', sharey='row')
    
    for l, errmode in enumerate(errmodes):
        cxi = np.load('../data/crb/bspline_cxi{:s}_{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, errmode, n, vlabel, Ndim))
        if fast:
            cx = np.linalg.inv(cxi)
        else:
            cx = stable_inverse(cxi)
        cx = cx[i0:i1,i0:i1]
        #print(np.sqrt(np.diag(cx)))
        
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar):
                plt.sca(ax[j-1][i])
                if scale:
                    cx_2d = np.array([[cx[i][i]/dp_unit[i]**2, cx[i][j]/(dp_unit[i]*dp_unit[j])], [cx[j][i]/(dp_unit[j]*dp_unit[i]), cx[j][j]/dp_unit[j]**2]])
                else:
                    cx_2d = np.array([[cx[i][i], cx[i][j]], [cx[j][i], cx[j][j]]])
                
                w, v = np.linalg.eig(cx_2d)
                if np.all(np.isreal(v)):
                    theta = np.degrees(np.arctan2(v[1][0], v[0][0]))
                    width = np.sqrt(w[0])*2
                    height = np.sqrt(w[1])*2
                    
                    e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=mpl.cm.bone(0.1+l/4), lw=2, label=label[l])
                    plt.gca().add_patch(e)
                
                if l==1:
                    plt.gca().autoscale_view()
                
                if j==Nvar-1:
                    plt.xlabel(params[i])
                    
                if i==0:
                    plt.ylabel(params[j])
        
        # turn off unused axes
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar-1):
                plt.sca(ax[i][j])
                plt.axis('off')
        
        plt.sca(ax[int(Nvar/2-1)][int(Nvar/2-1)])
        plt.legend(loc=2, bbox_to_anchor=(1,1))
    
    plt.tight_layout()
    plt.savefig('../plots/crb_triangle_alldim{:s}_comparison_{:d}_{:s}_{:s}.pdf'.format(alabel, n, vlabel, plot))

def comp_obsmodes(vary=['progenitor', 'bary', 'halo'], align=True, component='halo'):
    """Compare CRBs from different observing modes"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    
    pid_comp, dp_fid2, vlabel2 = get_varied_pars(component)
    Nvar = len(pid_comp)
    plabels, units = get_parlabel(pid_comp)
    punits = [' (%)' for x in units]
    params = ['$\Delta$ {}{}'.format(x, y) for x,y in zip(plabels, punits)]
    plainlabels = ['V_h', 'R_h', 'q_x', 'q_z']
    
    names = get_done()
    
    errmodes = ['fiducial', 'fiducial', 'fiducial', 'desi', 'gaia']
    Ndims = [ 3, 4, 6, 4, 6]
    Nmode = len(errmodes)
    
    # fiducial
    errmode = 'fiducial'
    Ndim = 6
    coll_fiducial = np.load('../data/crb/cx_collate_multi1_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(errmode, Ndim, align, vlabel, component))
    
    #errmodes = ['fiducial', 'gaia', 'desi']
    #Ndims = [6,6,4]

    labels = {'desi': 'DESI-like', 'gaia': 'Gaia-like', 'fiducial': 'Fiducial'}
    cfrac = {'desi': 0.8, 'gaia': 0.6, 'fiducial': 0.2}
    
    cmap = {'fiducial': mpl.cm.bone, 'desi': mpl.cm.pink, 'gaia': mpl.cm.pink}
    frac = [0.8, 0.5, 0.2, 0.5, 0.2]
    ls_all = ['-', '-', '-', '--', '--']
    a = 0.7
    
    da = 3
    ncol = 2
    nrow = np.int64(Nvar/ncol)
    w = 4 * da
    h = nrow * da * 1.3
    
    plt.close()
    fig, ax = plt.subplots(nrow+2, ncol, figsize=(w, h), sharex=True, gridspec_kw = {'height_ratios':[3, 1.2, 3, 1.2]})
    
    for i in range(Nmode):
        errmode = errmodes[i]
        Ndim = Ndims[i]
        coll = np.load('../data/crb/cx_collate_multi1_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(errmode, Ndim, align, vlabel, component))
        
        lw = np.sqrt(Ndims[i]) * 2
        ls = ls_all[i]
        #color = mpl.cm.bone(cfrac[errmodes[i]])
        color = cmap[errmode](frac[i])
        
        for j in range(Nvar):
            #plt.sca(ax[j])
            plt.sca(ax[j%ncol*2][np.int64(j/ncol)])
            if labels[errmode]=='Fiducial':
                label = '{} {}D'.format(labels[errmode], Ndims[i])
            else:
                label = '{} ({}D)'.format(labels[errmode], Ndims[i])
            plt.plot(coll['p_rel'][:,j]*100, '-', ls=ls, alpha=a, lw=lw, color=color, label=label)
            
            plt.sca(ax[j%ncol*2+1][np.int64(j/ncol)])
            plt.plot(coll['p_rel'][:,j]/coll_fiducial['p_rel'][:,j], '-', ls=ls, alpha=a, lw=lw, color=color)
            
            #print(errmode, j, np.median(coll['p_rel'][:,j]/coll_fiducial['p_rel'][:,j]), np.std(coll['p_rel'][:,j]/coll_fiducial['p_rel'][:,j]))
            
    
    for j in range(Nvar):
        plt.sca(ax[j%ncol*2][np.int64(j/ncol)])
        plt.ylabel(params[j])
        plt.gca().set_yscale('log')
        plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        
        plt.sca(ax[j%ncol*2+1][np.int64(j/ncol)])
        plt.ylabel('$\\frac{\Delta %s}{\Delta {%s}_{,\,Fid\,6D}}$'%(plainlabels[j], plainlabels[j]), fontsize='medium')
        plt.ylim(0.5, 10)
        plt.gca().set_yscale('log')
        plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
    
    plt.sca(ax[nrow][ncol-1])
    plt.legend(loc=0, fontsize='x-small', handlelength=0.8, frameon=True)
    
    # stream names
    for j in range(ncol):
        plt.sca(ax[0][j])
        y0, y1 = plt.gca().get_ylim()
        fp = 0.8
        yp = y0 + fp*(y1-y0)
        
        for e, name in enumerate(names):
            txt = plt.text(e, yp, name, ha='center', va='top', rotation=90, fontsize='x-small', color='0.2')
            txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))
    
    plt.tight_layout()
    plt.savefig('../plots/obsmode_comparison.pdf')

def vel_improvement(vary=['progenitor', 'bary', 'halo'], align=True, component='halo', errmode='fiducial'):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    
    pid_comp, dp_fid2, vlabel2 = get_varied_pars(component)
    Nvar = len(pid_comp)
    plabels, units = get_parlabel(pid_comp)
    punits = [' (%)' for x in units]
    params = ['$\Delta$ {}{}'.format(x, y) for x,y in zip(plabels, punits)]
    
    names = get_done()
    
    coll = []
    
    for Ndim in [3,4,6]:
        coll += [np.load('../data/crb/cx_collate_multi1_{:s}{:1d}_a{:1d}_{:s}_{:s}.npz'.format(errmode, Ndim, align, vlabel, component))]
    
    rv = coll[0]['p_rel'] / coll[1]['p_rel']
    pm = coll[1]['p_rel'] / coll[2]['p_rel']
    
    N = len(names)
    prog_rv = np.empty(N)
    prog_pm = np.empty(N)
    
    for i in range(N):
        mock = pickle.load(open('../data/mock_{}.params'.format(names[i]), 'rb'))
        pms = np.array([x.value for x in mock['v0'][1:]])
        prog_rv[i] = np.abs(mock['v0'][0].value)
        #prog_pm[i] = np.linalg.norm(pms)
        prog_pm[i] = max(np.abs(pms))
        
    da = 2
    
    plt.close()
    fig, ax = plt.subplots(Nvar, 3, figsize=(da*3, da*Nvar), sharex='col')
    
    for j in range(Nvar):
        plt.sca(ax[j][0])
        plt.plot(prog_rv, rv[:,j], 'ko')
        
        plt.sca(ax[j][1])
        plt.plot(prog_rv/prog_pm, pm[:,j], 'ko')
        
        plt.sca(ax[j][2])
        plt.plot(prog_pm, pm[:,j], 'ko')
    
    plt.tight_layout()


###
# Referee's report
###

def mass_age(name='atlas', pparams0=pparams_fid, dt=0.2*u.Myr, rotmatrix=np.eye(3), graph=False, graphsave=False, observer=mw_observer, vobs=vsun, footprint='', obsmode='equatorial'):
    """Create a streakline model of a stream
    baryonic component as in kupper+2015: 3.4e10*u.Msun, 0.7*u.kpc, 1e11*u.Msun, 6.5*u.kpc, 0.26*u.kpc"""
    
    # vary progenitor parameters
    mock = pickle.load(open('../data/mock_{}.params'.format(name), 'rb'))
    for i in range(3):
        mock['x0'][i] += pparams0[26+i]
        mock['v0'][i] += pparams0[29+i]
    
    # vary potential parameters
    potential = 'octu'
    pparams = pparams0[:26]
    #print(pparams[0])
    pparams[0] = (10**pparams0[0].value)*pparams0[0].unit
    pparams[2] = (10**pparams0[2].value)*pparams0[2].unit
    #pparams[0] = pparams0[0]*1e15
    #pparams[2] = pparams0[2]*1e15
    #print(pparams[0])
    
    # adjust circular velocity in this halo
    vobs['vcirc'] = vcirc_potential(observer['galcen_distance'], pparams=pparams)
    
    ylabel = ['Dec (deg)', 'd (kpc)', '$V_r$ (km/s)', '$\mu_\\alpha$ (mas yr$^{-1}$)', '$\mu_\delta$ (mas yr$^{-1}$)']
    
    plt.close()
    fig, ax = plt.subplots(2, 5, figsize=(20,7), sharex='col', sharey='col', squeeze=False)
    
    for e, f in enumerate(np.arange(0.8,1.21,0.1)[::-1]):
    
        # create a model stream with these parameters
        params = {'generate': {'x0': mock['x0'], 'v0': mock['v0'], 'progenitor': {'coords': 'equatorial', 'observer': mock['observer'], 'pm_polar': False}, 'potential': potential, 'pparams': pparams, 'minit': f*mock['mi'], 'mfinal': mock['mf'], 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': dt, 'age': mock['age'], 'nstars': 400, 'integrator': 'lf'}, 'observe': {'mode': mock['obsmode'], 'wangle': mock['wangle'], 'nstars':-1, 'sequential':True, 'errors': [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,3,4,5], 'observer': mock['observer'], 'vobs': mock['vobs'], 'footprint': mock['footprint'], 'rotmatrix': rotmatrix}}
        
        stream = Stream(**params['generate'])
        stream.generate()
        stream.observe(**params['observe'])
        
        for i in range(5):
            plt.sca(ax[0][i])
            
            plt.gca().invert_xaxis()
            #plt.xlabel('R.A. (deg)')
            plt.ylabel(ylabel[i])
            
            plt.plot(stream.obs[0], stream.obs[i+1], 'o', color=mpl.cm.viridis(e/5), mec='none', ms=4, label='{:.2g}$\\times$10$^3$ M$_\odot$'.format(f*mock['mi'].to(u.Msun).value*1e-3))
            
            if (i==0) & (e==4):
                plt.legend(frameon=True, handlelength=0.5, fontsize='small', markerscale=1.5)
            
            if i==2:
                plt.title('Age = {:.2g}'.format(mock['age'].to(u.Gyr)), fontsize='medium')
    
        params = {'generate': {'x0': mock['x0'], 'v0': mock['v0'], 'progenitor': {'coords': 'equatorial', 'observer': mock['observer'], 'pm_polar': False}, 'potential': potential, 'pparams': pparams, 'minit': mock['mi'], 'mfinal': mock['mf'], 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': dt, 'age': f*mock['age'], 'nstars': 400, 'integrator': 'lf'}, 'observe': {'mode': mock['obsmode'], 'wangle': mock['wangle'], 'nstars':-1, 'sequential':True, 'errors': [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,3,4,5], 'observer': mock['observer'], 'vobs': mock['vobs'], 'footprint': mock['footprint'], 'rotmatrix': rotmatrix}}
        
        stream = Stream(**params['generate'])
        stream.generate()
        stream.observe(**params['observe'])
        
        for i in range(5):
            plt.sca(ax[1][i])
            
            plt.gca().invert_xaxis()
            plt.xlabel('R.A. (deg)')
            plt.ylabel(ylabel[i])
            
            plt.plot(stream.obs[0], stream.obs[i+1], 'o', color=mpl.cm.viridis(e/5), mec='none', ms=4, label='{:.2g}'.format(f*mock['age'].to(u.Gyr)))
            
            if (i==0) & (e==4):
                plt.legend(frameon=True, handlelength=0.5, fontsize='small', markerscale=1.5)
            
            if i==2:
                plt.title('Initial mass = {:.2g}$\\times$10$^3$ M$_\odot$'.format(mock['mi'].to(u.Msun).value*1e-3), fontsize='medium')
        
    plt.tight_layout(w_pad=0)
    plt.savefig('../paper/age_mass_{}.png'.format(name))

# progenitor's orbit

def prog_orbit(n):
    """"""
    
    orbit = stream_orbit(n)

    R = np.linalg.norm(orbit['x'][:2,:].to(u.kpc), axis=0)[::-1]
    x = orbit['x'][0].to(u.kpc)[::-1]
    y = orbit['x'][1].to(u.kpc)[::-1]
    z = orbit['x'][2].to(u.kpc)[::-1]
    
    c = np.arange(np.size(z))[::-1]
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    plt.sca(ax[0])
    plt.scatter(x, y, c=c, cmap=mpl.cm.gray)
    
    plt.xlabel('X (kpc)')
    plt.ylabel('Y (kpc)')
    
    plt.sca(ax[1])
    plt.scatter(x, z, c=c, cmap=mpl.cm.gray)
    
    plt.xlabel('X (kpc)')
    plt.ylabel('Z (kpc)')
    
    plt.sca(ax[2])
    plt.scatter(y, z, c=c, cmap=mpl.cm.gray)
    
    plt.xlabel('Y (kpc)')
    plt.ylabel('Z (kpc)')
    
    plt.tight_layout()
    plt.savefig('../plots/orbit_cartesian_{}.png'.format(n))
    #plt.scatter(R[::-1], z[::-1], c=c[::-1], cmap=mpl.cm.gray)
    #plt.plot(Rp, zp, 'ko', ms=10)
    
    #plt.xlim(0,40)
    #plt.ylim(-20,20)

def prog_orbit3d(name, symmetry=False):
    """"""
    
    orbit = stream_orbit(name)

    R = np.linalg.norm(orbit['x'][:2,:].to(u.kpc), axis=0)[::-1]
    x = orbit['x'][0].to(u.kpc)[::-1].value
    y = orbit['x'][1].to(u.kpc)[::-1].value
    z = orbit['x'][2].to(u.kpc)[::-1].value
    
    c = np.arange(np.size(z))[::-1]
    
    plt.close()
    fig = plt.figure(figsize=(9,9))
    
    ax = fig.add_subplot(1,1,1, projection='3d')
    if symmetry:
        azimuth = {-1: 119, -2: -39, -3: -5, -4: -11}
        elevation = {-1: 49, -2: -117, -3: 49, -4: 60}
        ax.view_init(azim=azimuth[n], elev=elevation[n])
    else:
        ax.view_init(azim=-10, elev=30)
    ax.set_frame_on(False)
    
    ax.scatter(x, y, z, 'o', depthshade=False, c=c, cmap=mpl.cm.YlOrBr_r)
    
    ax.set_xlabel('X (kpc)')
    ax.set_ylabel('Y (kpc)')
    ax.set_zlabel('Z (kpc)')
    plt.title('{}'.format(name))
    
    plt.tight_layout()
    plt.savefig('../plots/orbit_3d_{}_{:d}.png'.format(name, symmetry))

def stream_orbit(name='gd1', pparams0=pparams_fid, dt=0.2*u.Myr, rotmatrix=np.eye(3), diagnostic=False, observer=mw_observer, vobs=vsun, footprint='', obsmode='equatorial'):
    """Create a streakline model of a stream
    baryonic component as in kupper+2015: 3.4e10*u.Msun, 0.7*u.kpc, 1e11*u.Msun, 6.5*u.kpc, 0.26*u.kpc"""
    
    # vary progenitor parameters
    mock = pickle.load(open('../data/mock_{}.params'.format(name), 'rb'))
    #for i in range(3):
        #mock['x0'][i] += pparams0[19+i]
        #mock['v0'][i] += pparams0[22+i]
    
    # vary potential parameters
    potential = 'quad'
    pparams = pparams0[:19]
    pparams[0] = pparams0[0]*1e10
    pparams[2] = pparams0[2]*1e10
    
    # adjust circular velocity in this halo
    vobs['vcirc'] = vcirc_potential(observer['galcen_distance'], pparams=pparams)

    # create a model stream with these parameters
    params = {'generate': {'x0': mock['x0'], 'v0': mock['v0'], 'progenitor': {'coords': 'equatorial', 'observer': mock['observer'], 'pm_polar': False}, 'potential': potential, 'pparams': pparams, 'minit': mock['mi'], 'mfinal': mock['mf'], 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': dt, 'age': mock['age'], 'nstars': 400, 'integrator': 'lf'}, 'observe': {'mode': mock['obsmode'], 'nstars':-1, 'sequential':True, 'errors': [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,3,4,5], 'observer': mock['observer'], 'vobs': mock['vobs'], 'footprint': mock['footprint'], 'rotmatrix': rotmatrix}}
    
    stream = Stream(**params['generate'])
    stream.prog_orbit()
    
    if diagnostic:
        r = np.linalg.norm(stream.orbit['x'].to(u.kpc), axis=0)
        rmin = np.min(r)
        rmax = np.max(r)
        e = (rmax - rmin)/(rmax + rmin)
        print(rmin, rmax, e)
    
    return stream.orbit

def check_rcur():
    """"""
    done = get_done()[::-1]
    N = len(done)
    
    t = Table.read('../data/crb/ar_orbital_summary.fits')

    for i, name in enumerate(done):
        mock = pickle.load(open('../data/mock_{}.params'.format(name), 'rb'))
        c = coord.ICRS(ra=mock['x0'][0], dec=mock['x0'][1], distance=mock['x0'][2])
        gal = c.transform_to(coord.Galactocentric)
        rcur = np.sqrt(gal.x**2 + gal.y**2 + gal.z**2).to(u.kpc)
        print(done[i], rcur, np.array(t[t['name']==name]['rcur']))

# summary of parameter constraints

def relative_crb(vary=['progenitor', 'bary', 'halo'], component='all', Ndim=6, align=True, fast=False, scale=False):
    """Plot crb_param/param for 3 streams"""
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    # choose the appropriate components:
    Nprog, Nbary, Nhalo, Ndipole, Nquad, Npoint = [6, 5, 4, 3, 5, 1]
    if 'progenitor' not in vary:
        Nprog = 0
    nstart = {'bary': Nprog, 'halo': Nprog + Nbary, 'dipole': Nprog + Nbary + Nhalo, 'quad': Nprog + Nbary + Nhalo + Ndipole, 'all': Nprog, 'point': 0}
    nend = {'bary': Nprog + Nbary, 'halo': Nprog + Nbary + Nhalo, 'dipole': Nprog + Nbary + Nhalo + Ndipole, 'quad': Nprog + Nbary + Nhalo + Ndipole + Nquad, 'all': len(pid), 'point': 1}
    
    if 'progenitor' not in vary:
        nstart['dipole'] = Npoint
        nend['dipole'] = Npoint + Ndipole
    
    if component in ['bary', 'halo', 'dipole', 'quad', 'point']:
        components = [component]
    else:
        components = [x for x in vary if x!='progenitor']
    
    plabels, units = get_parlabel(pid)
    #params = ['$\Delta$' + x + '({})'.format(y) for x,y in zip(plabels, units)]
    params = [x for x in plabels]
    params = params[nstart[component]:nend[component]]
    Nvar = len(params)
    xpos = np.arange(Nvar)
    
    params_fid = np.array([pparams_fid[x].value for x in pid[nstart[component]:nend[component]]])
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    for n in [-1,-2,-3]:
        cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
        if fast:
            cx = np.linalg.inv(cxi)
        else:
            cx = stable_inverse(cxi)
        
        cq = cx[nstart[component]:nend[component], nstart[component]:nend[component]]

        if scale:
            dp_opt = read_optimal_step(n, vary)
            dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
            
            scale_vec = np.array([x.value for x in dp[nstart[component]:nend[component]]])
            scale_mat = np.outer(scale_vec, scale_vec)
            cq /= scale_mat
        
        crb = np.sqrt(np.diag(cq))
        crb_rel = crb / params_fid
        
        print(fancy_name(n))
        #print(crb)
        print(crb_rel)
    
        plt.plot(xpos, crb_rel, 'o', label='{}'.format(fancy_name(n)))
    
    plt.legend(fontsize='small')
    plt.ylabel('Relative CRB')
    plt.xticks(xpos, params, rotation='horizontal', fontsize='medium')
    plt.xlabel('Parameter')
    
    plt.ylim(0, 0.2)
    #plt.gca().set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('../plots/relative_crb_{:s}_{:s}_{:d}.png'.format(vlabel, component, Ndim))
    
def relative_crb_sky(vary=['progenitor', 'bary', 'halo'], component='all', Ndim=6, align=True, fast=False, scale=False):
    """"""
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    # choose the appropriate components:
    Nprog, Nbary, Nhalo, Ndipole, Nquad, Npoint = [6, 5, 4, 3, 5, 1]
    if 'progenitor' not in vary:
        Nprog = 0
    nstart = {'bary': Nprog, 'halo': Nprog + Nbary, 'dipole': Nprog + Nbary + Nhalo, 'quad': Nprog + Nbary + Nhalo + Ndipole, 'all': Nprog, 'point': 0}
    nend = {'bary': Nprog + Nbary, 'halo': Nprog + Nbary + Nhalo, 'dipole': Nprog + Nbary + Nhalo + Ndipole, 'quad': Nprog + Nbary + Nhalo + Ndipole + Nquad, 'all': len(pid), 'point': 1}
    
    if 'progenitor' not in vary:
        nstart['dipole'] = Npoint
        nend['dipole'] = Npoint + Ndipole
    
    if component in ['bary', 'halo', 'dipole', 'quad', 'point']:
        components = [component]
    else:
        components = [x for x in vary if x!='progenitor']
    
    plabels, units = get_parlabel(pid)
    #params = ['$\Delta$' + x + '({})'.format(y) for x,y in zip(plabels, units)]
    params = [x for x in plabels]
    params = params[nstart[component]:nend[component]]
    Nvar = len(params)
    xpos = np.arange(Nvar)
    
    params_fid = np.array([pparams_fid[x].value for x in pid[nstart[component]:nend[component]]])
    
    dd = 5
    plt.close()
    fig, ax = plt.subplots(Nvar, 2, figsize=(dd, 0.5*dd*Nvar), sharex='col', sharey='col', gridspec_kw = {'width_ratios':[6, 1]})
    
    for n in [-1,-2,-3]:
        cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
        if fast:
            cx = np.linalg.inv(cxi)
        else:
            cx = stable_inverse(cxi)
        
        cq = cx[nstart[component]:nend[component], nstart[component]:nend[component]]

        if scale:
            dp_opt = read_optimal_step(n, vary)
            dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
            
            scale_vec = np.array([x.value for x in dp[nstart[component]:nend[component]]])
            scale_mat = np.outer(scale_vec, scale_vec)
            cq /= scale_mat
        
        crb = np.sqrt(np.diag(cq))
        crb_rel = crb / params_fid
        
        #print(fancy_name(n))
        ##print(crb)
        #print(crb_rel)
        
        stream = stream_model(n)
        for i in range(Nvar):
            vmin, vmax = -2, 2
            cind = (np.log10(crb_rel[i]) - vmin)/(vmax - vmin)
            color = mpl.cm.magma_r(cind)
            
            plt.sca(ax[i])
            plt.plot(stream.obs[0], stream.obs[1], 'o', color=color)
    
    for i in range(Nvar):
        plt.sca(ax[i])
        plt.gca().set_axis_bgcolor(mpl.cm.magma(0))
        plt.gca().invert_xaxis()

        plt.title(params[i], fontsize='medium')
        plt.ylabel('Dec (deg)')
        if i==Nvar-1:
            plt.xlabel('R.A. (deg)')
        
    #plt.legend(fontsize='small')
    #plt.ylabel('Relative CRB')
    #plt.xticks(xpos, params, rotation='horizontal', fontsize='medium')
    #plt.xlabel('Parameter')
    
    #plt.gca().set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('../plots/relative_crb_sky_{:s}_{:s}_{:d}.png'.format(vlabel, component, Ndim))
    
# toy problem: kepler + dipole

import sklearn.datasets
def create_fmi(n, Ndim=4, niter=20, alabel='_align', vlabel='point_dipole', Nobsdim=6):
    """"""
    state = n
    
    invertible = False
    cnt = 0
    
    for cnt in range(niter):
        cxi = sklearn.datasets.make_spd_matrix(Ndim, random_state=state)
        cx = stable_inverse(cxi)
        invertible = np.allclose(np.matmul(cxi, cx), np.eye(Ndim))
        if invertible:
            break
        else:
            state = np.random.get_state()
    
    np.save('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}'.format(alabel, n, vlabel, Nobsdim), cxi)
    
    cx[0,1:] = 0
    cx[1:,0] = 0
    cxi = stable_inverse(cx)
    
    np.save('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}'.format(alabel, n+1, vlabel, Nobsdim), cxi)

def basic_fmi(n=0, alabel='_align', vlabel='point_dipole', Nobsdim=6):
    """"""
    
    Ndim = 4
    cxi = np.diag([1.5, 3, 1, 1])
    
    np.save('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}'.format(alabel, n, vlabel, Nobsdim), cxi)

def crb_toy(n, alabel='_align', Nobsdim=6, vlabel='point_dipole'):
    """"""


def talk_crb_triangle(n=-1, vary=['progenitor', 'bary', 'halo'], plot='all', reveal=0, fast=False, scale=False):
    """Produce a triangle plot of 2D Cramer-Rao bounds for all model parameters using a given stream"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    dp_opt = read_optimal_step(n, vary)
    dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
    plabels, units = get_parlabel(pid)
    params = ['$\Delta$' + x + '({})'.format(y) for x,y in zip(plabels, units)]
    alabel='_align'
    
    if plot=='halo':
        i0 = 11
        i1 = 15
    elif plot=='bary':
        i0 = 6
        i1 = 11
    elif plot=='progenitor':
        i0 = 0
        i1 = 6
    elif plot=='dipole':
        i0 = 15
        i1 = len(params)
    else:
        i0 = 0
        i1 = len(params)
    
    Nvar = i1 - i0
    params = params[i0:i1]
    
    #label = ['GD-1', 'Pal 5']
    label = ['RA, Dec, d', 'RA, Dec, d, $V_r$', 'RA, Dec, d, $V_r$, $\mu_\\alpha$, $\mu_\delta$']
    #name = columns[int(np.abs(n)-1)]
    
    #labels = ['RA, Dec, d', 'RA, Dec, d,\n$V_r$', 'RA, Dec, d,\n$V_r$, $\mu_\\alpha$, $\mu_\\delta$']
    #params0 = ['$V_h$ (km/s)', '$R_h$ (kpc)', '$q_1$', '$q_z$', '$M_{LMC}$', '$X_p$', '$Y_p$', '$Z_p$', '$V_{xp}$', '$V_{yp}$', '$V_{zp}$']
    #params = ['$\Delta$ '+x for x in params0]
    ylim = [150, 20, 0.5, 0.5, 5e11]
    ylim = [20, 10, 0.1, 0.1]
    
    plt.close()
    fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(8,8), sharex='col', sharey='row')
    
    # plot 2d bounds in a triangle fashion
    Ndim = 3
    #labels = columns
    streams = np.array([-1,-2,-3,-4])
    slist = streams[:reveal+1]
    #for l, n in enumerate(slist):
    for l, Ndim in enumerate([3, 4, 6]):
        cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
        if fast:
            cx = np.linalg.inv(cxi)
        else:
            cx = stable_inverse(cxi)
        cx = cx[i0:i1,i0:i1]
        
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar):
                plt.sca(ax[j-1][i])
                if scale:
                    cx_2d = np.array([[cx[i][i]/dp_unit[i]**2, cx[i][j]/(dp_unit[i]*dp_unit[j])], [cx[j][i]/(dp_unit[j]*dp_unit[i]), cx[j][j]/dp_unit[j]**2]])
                else:
                    cx_2d = np.array([[cx[i][i], cx[i][j]], [cx[j][i], cx[j][j]]])
                
                w, v = np.linalg.eig(cx_2d)
                if np.all(np.isreal(v)):
                    theta = np.degrees(np.arccos(v[0][0]))
                    width = np.sqrt(w[0])*2
                    height = np.sqrt(w[1])*2
                    
                    e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=mpl.cm.PuBu((l+3)/6), lw=3, label=label[l])
                    plt.gca().add_patch(e)
                
                if l==1:
                    plt.gca().autoscale_view()
                
                if j==Nvar-1:
                    plt.xlabel(params[i])
                    
                if i==0:
                    plt.ylabel(params[j])
        
        # turn off unused axes
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar-1):
                plt.sca(ax[i][j])
                plt.axis('off')
        
        plt.sca(ax[int(Nvar/2-1)][int(Nvar/2-1)])
        plt.legend(loc=2, bbox_to_anchor=(1,1))
    
    #plt.title('Marginalized ')
    
    #plt.tight_layout()
    plt.tight_layout(h_pad=0.0, w_pad=0.0)
    plt.savefig('../plots/talk2/triangle_{}.png'.format(n))
    #plt.savefig('../plots/talk2/triangle_{}.png'.format(reveal))

def talk_stream_comp(n=-1, vary=['progenitor', 'bary', 'halo'], plot='all', reveal=0, fast=False, scale=False):
    """Produce a triangle plot of 2D Cramer-Rao bounds for all model parameters using a given stream"""
    
    pid, dp_fid, vlabel = get_varied_pars(vary)
    dp_opt = read_optimal_step(n, vary)
    dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
    plabels, units = get_parlabel(pid)
    params = ['$\Delta$' + x + '({})'.format(y) for x,y in zip(plabels, units)]
    alabel='_align'
    
    if plot=='halo':
        i0 = 11
        i1 = 15
    elif plot=='bary':
        i0 = 6
        i1 = 11
    elif plot=='progenitor':
        i0 = 0
        i1 = 6
    elif plot=='dipole':
        i0 = 15
        i1 = len(params)
    else:
        i0 = 0
        i1 = len(params)
    
    Nvar = i1 - i0
    params = params[i0:i1]
    
    label = ['GD-1', 'Pal 5', 'Triangulum']
    #label = ['RA, Dec, d', 'RA, Dec, d, $V_r$', 'RA, Dec, d, $V_r$, $\mu_\\alpha$, $\mu_\delta$']
    #name = columns[int(np.abs(n)-1)]
    
    #labels = ['RA, Dec, d', 'RA, Dec, d,\n$V_r$', 'RA, Dec, d,\n$V_r$, $\mu_\\alpha$, $\mu_\\delta$']
    #params0 = ['$V_h$ (km/s)', '$R_h$ (kpc)', '$q_1$', '$q_z$', '$M_{LMC}$', '$X_p$', '$Y_p$', '$Z_p$', '$V_{xp}$', '$V_{yp}$', '$V_{zp}$']
    #params = ['$\Delta$ '+x for x in params0]
    ylim = [150, 20, 0.5, 0.5, 5e11]
    ylim = [20, 10, 0.1, 0.1]
    
    plt.close()
    fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(8,8), sharex='col', sharey='row')
    
    # plot 2d bounds in a triangle fashion
    Ndim = 3
    #labels = columns
    streams = np.array([-1,-2,-3,-4])
    slist = streams[:reveal+1]
    for l, n in enumerate(slist):
    #for l, Ndim in enumerate([3, 4, 6]):
        cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
        if fast:
            cx = np.linalg.inv(cxi)
        else:
            cx = stable_inverse(cxi)
        cx = cx[i0:i1,i0:i1]
        
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar):
                plt.sca(ax[j-1][i])
                if scale:
                    cx_2d = np.array([[cx[i][i]/dp_unit[i]**2, cx[i][j]/(dp_unit[i]*dp_unit[j])], [cx[j][i]/(dp_unit[j]*dp_unit[i]), cx[j][j]/dp_unit[j]**2]])
                else:
                    cx_2d = np.array([[cx[i][i], cx[i][j]], [cx[j][i], cx[j][j]]])
                
                w, v = np.linalg.eig(cx_2d)
                if np.all(np.isreal(v)):
                    theta = np.degrees(np.arctan2(v[1][0], v[0][0]))
                    width = np.sqrt(w[0])*2
                    height = np.sqrt(w[1])*2
                    
                    e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=mpl.cm.YlOrBr((l+3)/6), lw=3, label=label[l])
                    plt.gca().add_patch(e)
                
                if l==0:
                    plt.gca().autoscale_view()
                
                if j==Nvar-1:
                    plt.xlabel(params[i])
                    
                if i==0:
                    plt.ylabel(params[j])
        
        # turn off unused axes
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar-1):
                plt.sca(ax[i][j])
                plt.axis('off')
        
        plt.sca(ax[int(Nvar/2-1)][int(Nvar/2-1)])
        plt.legend(loc=2, bbox_to_anchor=(1,1))
    
    #plt.title('Marginalized ')
    
    #plt.tight_layout()
    plt.tight_layout(h_pad=0.0, w_pad=0.0)
    plt.savefig('../plots/talk2/comparison_{}.png'.format(reveal))

def test_ellipse():
    """"""
    
    th = np.radians(60)
    v = np.array([[np.cos(th),np.sin(th)], [-np.sin(th),np.cos(th)]])
    w = np.array([2,1])
    
    plt.close()
    plt.figure()
    
    theta = np.degrees(np.arctan2(v[0][1], v[0][0]))
    print(theta, np.degrees(th))
    e = mpl.patches.Ellipse((0,0), width=w[0]*2, height=w[1]*2, angle=theta, fc='none', ec='k', lw=2)
    plt.gca().add_artist(e)
    
    plt.xlim(-5,5)
    plt.ylim(-5,5)

def test_ellipse2():
    """"""
    v1 = np.array([1.5, -0.05])
    v2 = np.array([0.01, 0.3])
    c = np.outer(v1, v1) + np.outer(v2, v2)
    w, v = np.linalg.eig(c)
    
    print(w)
    print(v)
    
    plt.close()
    plt.figure()
    
    theta = np.degrees(np.arctan2(v[1][0], v[0][0]))
    width = np.sqrt(w[0])*2
    height = np.sqrt(w[1])*2
    print(width/height)
    e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec='k', lw=2)
    plt.gca().add_artist(e)
    
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.savefig('../plots/test_ellipse.png')

def test_ellipse3():
    """"""
    v1 = np.array([-28., -8.])
    v2 = np.array([6., -21.])
    c = np.outer(v1, v1) + np.outer(v2, v2)
    w, v = np.linalg.eig(c)
    
    print(w)
    print(v)
    
    plt.close()
    plt.figure()
    
    theta = np.degrees(np.arctan2(v[1][0], v[0][0]))
    width = np.sqrt(w[0])*2
    height = np.sqrt(w[1])*2
    print(width, height, width/height)
    e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec='k', lw=2)
    
    plt.gca().add_artist(e)
    plt.gca().autoscale_view()
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.savefig('../plots/test_ellipse3.png')
