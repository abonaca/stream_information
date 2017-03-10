from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import streakline
#import halo_masses as hm
import myutils
import ffwd

import astropy
import astropy.units as u
from astropy.constants import G
from astropy.table import Table
#from astropy.io import fits
import astropy.coordinates as coord
#from astropy.coordinates import Angle
import gala.coordinates as gc

# observers
vl2_observer = {'z_sun': 0.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_ra': 300*u.deg, 'galcen_dec': 20*u.deg}

# defaults taken as in astropy v1.1 icrs
mw_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_ra': coord.Angle("17:45:37.224 hours"), 'galcen_dec': coord.Angle("-28:56:10.23 degrees")}
vsun = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}
vsun0 = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

MASK = -9999

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
        elif self.setup['potential']=='gal':
            self.setup_aux['paux'] = 4
        elif self.setup['potential']=='lmc':
            self.setup_aux['paux'] = 6
            
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
    
    def observe(self, mode='cartesian', units=[], errors=[], nstars=-1, sequential=False, present=[], logerr=False, observer = {'z_sun': 0.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_ra': 300*u.deg, 'galcen_dec': 20*u.deg}, footprint='none'):
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
            veq = gc.vgal_to_hel(xeq, v, **vsun)
            
            # store coordinates
            ra, dec, dist = [xeq.ra.to(units[0]), xeq.dec.to(units[1]), xeq.distance.to(units[2])]
            vr, mua, mud = [veq[2].to(units[3]), veq[0].to(units[4]), veq[1].to(units[5])]
            
            obs = np.hstack([ra, dec, dist, vr, mua, mud]).value
            obs = np.reshape(obs,(6,-1))
            
            if footprint=='sdss':
                infoot = dec > -2.5*u.deg
                obs = obs[:,infoot]
            
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

def gd1_coordinates(verbose=False):
    """Find best fitting GD1 coordinates from single-stream chain"""
    
    x0 = [-12.7, 0.1, 6.7]
    #v0 = [46, 73, -16]
    #v0 = [-60, -210, -120]
    v0 = [-100, -250, -100]
    #v0 = [-57, -173, -114]
    
    return (x0, v0)

def tri_coordinates():
    """"""
    
    #x0 = [-27, 19, -17]
    #v0 = [31, -135, -145]
    
    x = coord.SkyCoord(ra=22.38*u.deg, dec=30.26*u.deg, distance=35*u.kpc, **mw_observer)
    x_ = x.galactocentric
    x0 = [x_.x.value, x_.y.value, x_.z.value]
    v0 = [-31, 135, 145]
    
    return (x0, v0)

def pal5_coordinates(verbose=False):
    """Print pal5 coordinates in different systems"""
    
    # sdss
    ra = 229.0128*u.deg
    dec = -0.1082*u.deg
    # bob's rrlyrae
    d = 21.7*u.kpc
    #d = 23.2*u.kpc
    d = 24*u.kpc
    # odenkirchen 2002
    vr = -58.7*u.km/u.s
    # fritz & kallivayalil 2015
    mua = -2.296*u.mas/u.yr
    mud = -2.257*u.mas/u.yr
    
    x = coord.SkyCoord(ra=ra, dec=dec, distance=d, **mw_observer)
    v = gc.vhel_to_gal(x.icrs, rv=vr, pm=[mua, mud], **vsun0)
    
    if verbose: print(x.galactocentric, v)
    #v = np.array([ -32.98831945, -122.98597235,  -19.3962925 ])*u.km/u.s
    
    return (x.galactocentric, v)

def pal5_coordinates2():
    """Return pal5 cartesian coordinates in two lists"""

    x0, v0 = pal5_coordinates()
    
    return ([x0.x.value, x0.y.value, x0.z.value], v0.value.tolist())

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

def atlas_coordinates():
    """"""
    
    x = coord.SkyCoord(ra=20*u.deg, dec=-27*u.deg, distance=20*u.kpc, **mw_observer)
    x_ = x.galactocentric
    x0 = [x_.x.value, x_.y.value, x_.z.value]
    v0 = [40, 150, -120]
    
    return (x0, v0)

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

# effects of different params on stream shape

def plot_potstream2(n, pparams=[430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1)], potential='gal', age=None, nlabel=0):
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

# residuals
import scipy.interpolate
def plot_residuals(n, potential='gal'):
    """"""
    
    # Load streams
    if n==-1:
        observed = load_gd1(present=[0,1,2,3])
        xi = [140, 180]
        xlims = [[190, 130], [0, 350]]
        ylims = [[15, 65], [5, 10], [-250, 150], [0, 250]]
        loc = 2
        name = 'GD-1'
    elif n==-3:
        observed = load_tri(present=[0,1,2,3])
        xi = [20, 24]
        xlims = [[25, 19], [0, 350]]
        ylims = [[10, 50], [20, 45], [-175, -50], [0, 250]]
        loc = 1
        name = 'Triangulum'
    elif n==-4:
        observed = load_atlas(present=[0,1,2,3])
        xi = [15, 28]
        xlims = [[35, 10], [0, 350]]
        ylims = [[-40, -20], [15, 25], [50, 200], [0, 250]]
        loc = 3
        name = 'ATLAS'
    else:
        observed = load_pal5(present=[0,1,2,3])
        xi = [227, 242]
        xlims = [[245, 225], [0, 350]]
        ylims = [[-4, 10], [21, 27], [-80, -20], [0, 250]]
        loc = 3
        name = 'Pal 5'
    
    obscol = '0.5'
    modcols = [mpl.cm.bone(0.2), mpl.cm.bone(0.4), mpl.cm.bone(0.6), mpl.cm.bone(0.8)]
    mass = [0, 1, 2.5, 5]
    oblateness = [0.8,0.9,1,1.1]

    Ndim = 3
    Nmod = 4
    Nint = 100
    x = np.linspace(xi[0], xi[1], Nint)
    x = np.linspace(xlims[0][1], xlims[0][0], Nint)
    y = np.empty((Ndim, Nmod, Nint))
    #yfit = np.empty((Ndim, Nmod, Nint))
    
    plt.close()
    fig, ax = plt.subplots(3, 3, figsize=(12,8), sharex=True)
    
    plt.sca(ax[0][0])
    plt.plot(observed.obs[0], observed.obs[1], 's', color=obscol, mec='none', ms=8)
    
    plt.xlim(xlims[0][0], xlims[0][1])
    plt.ylim(ylims[0][0], ylims[0][1])
    #plt.xlabel("R.A. (deg)")
    plt.ylabel("Dec (deg)")
    
    plt.sca(ax[0][1])
    plt.plot(observed.obs[0], observed.obs[2], 's', color=obscol, mec='none', ms=8)
    
    plt.xlim(xlims[0][0], xlims[0][1])
    plt.ylim(ylims[1][0], ylims[1][1])
    #plt.xlabel("R.A. (deg)")
    plt.ylabel("Distance (kpc)")
    
    plt.sca(ax[0][2])
    if np.shape(observed.obs)[0]>3:
        rvsample = observed.obs[3]>MASK
        plt.plot(observed.obs[0][rvsample], observed.obs[3][rvsample], 's', color=obscol, mec='none', ms=8, label='Observed')
    
    plt.xlim(xlims[0][0], xlims[0][1])
    plt.ylim(ylims[2][0], ylims[2][1])
    #plt.xlabel("R.A. (deg)")
    plt.ylabel("Radial velocity (km/s)")

    ylabel = ['$\Delta$ Dec (deg)', '$\Delta$ d (kpc)', '$\Delta$ $V_r$ (km/s)']
    for k in range(3):
        plt.sca(ax[1][k])
        plt.ylabel(ylabel[k])
        
        plt.sca(ax[2][k])
        plt.ylabel(ylabel[k])
        plt.xlabel('R.A. (deg)')
    
    stream0 = np.load('../data/lmc/stream_{0:d}_{1:s}_{2:d}.npy'.format(n, potential, 0))
    
    for i in range(4):
        modcol = modcols[i]
        if potential=='gal':
            label = '$q_z$ = {:.1f}'.format(oblateness[i])
        elif potential=='lmc':
            label = 'M$_{LMC}$ = ' + '{:.1f}'.format(mass[i]) + '$\cdot 10^{11} M_\odot$'
        
        stream = np.load('../data/lmc/stream_{0:d}_{1:s}_{2:d}.npy'.format(n, potential, i))
        
        for k in range(Ndim):
            #f = scipy.interpolate.interp1d(stream[0], stream[1+k])
            f = np.poly1d(np.polyfit(stream[0], stream[1+k],3))
            y[k][i] = f(x)
            yfit = f(stream[0])
            
            plt.sca(ax[1][k])
            plt.plot(stream[0], yfit-stream[k+1], 'o', color=modcol, ms=4)
            if k==1:
                plt.title('Fit residual', fontsize=18)
            
            plt.sca(ax[2][k])
            plt.plot(x, y[k][i]-y[k][0], '-', color=modcol)
            if k==1:
                plt.title('Residual', fontsize=18)
        
        
        # Plot modeled streams
        plt.sca(ax[0][0])
        plt.plot(stream[0], stream[1], 'o', color=modcol, mec='none', ms=7)
        plt.plot(x, y[0][0], '-', color=modcols[0])
        
        plt.sca(ax[0][1])
        plt.plot(stream[0], stream[2], 'o', color=modcol, mec='none', ms=7)
        plt.plot(x, y[1][0], '-', color=modcols[0])
        
        plt.sca(ax[0][2])
        plt.plot(stream[0], stream[3], 'o', color=modcol, mec='none', ms=7, label=label)
        plt.plot(x, y[2][0], '-', color=modcols[0])
    
    plt.sca(ax[0][2])
    plt.legend(fontsize='xx-small', loc=loc, handlelength=0.2, frameon=False)
    plt.suptitle('{} stream'.format(name), fontsize='large')
    
    plt.tight_layout(h_pad=0.01, w_pad=0.02, rect=[0,0,1,0.97])
    plt.subplots_adjust(hspace=0.2, wspace=0.4)
    plt.savefig('../plots/lmc/residuals_{0:s}_{1:d}.png'.format(potential, n))

def crb(n, potential='gal', Ndim=6):
    """"""
    # Load streams
    if n==-1:
        observed = load_gd1(present=[0,1,2,3])
        xi = [140, 180]
        xlims = [[190, 130], [0, 350]]
        ylims = [[15, 65], [5, 10], [-250, 150], [0, 250]]
        loc = 2
        name = 'GD-1'
    elif n==-3:
        observed = load_tri(present=[0,1,2,3])
        xi = [20, 24]
        xlims = [[25, 19], [0, 350]]
        ylims = [[10, 50], [20, 45], [-175, -50], [0, 250]]
        loc = 1
        name = 'Triangulum'
    elif n==-4:
        observed = load_atlas(present=[0,1,2,3])
        xi = [15, 28]
        xlims = [[35, 10], [0, 350]]
        ylims = [[-40, -20], [15, 25], [50, 200], [0, 250]]
        loc = 3
        name = 'ATLAS'
    else:
        observed = load_pal5(present=[0,1,2,3])
        xi = [227, 242]
        xlims = [[245, 225], [0, 350]]
        ylims = [[-4, 10], [21, 27], [-80, -20], [0, 250]]
        loc = 3
        name = 'Pal 5'
        
    ifid = {'gal': 2, 'lmc': 2}
    dx = {'gal': 0.1, 'lmc': 2.5}
    
    #Ndim = np.shape(observed.obs)[0]
    
    #dydx = np.empty(0)
    #cyd = np.empty(0)
    
    # uncertainty 0.5 deg, instead of 1"
    observed.err[1]*= 1800
    
    # typical uncertainties
    sig_obs = np.array([0.5, 2, 5, 0.1, 0.1])
    
    # mock observations
    Nobs = 50
    ra = np.linspace(np.min(observed.obs[0]), np.max(observed.obs[0]), Nobs)
    err = np.tile(sig_obs, Nobs).reshape(Nobs,-1)
    #print(err, np.shape(err))
    
    params = ['gal', 'lmc']
    Npar = len(params)
    Ndata = Nobs * (Ndim - 1)
    dydx = np.empty((Npar, Ndata))
    cyd = np.empty(Ndata)
    
    # find derivatives, uncertainties
    for k in range(1,Ndim):
        fits = [None]*3
        for l, p in enumerate(params):
            
            for i, j in enumerate(range(ifid[p]-1,ifid[p]+2)):
                stream = np.load('../data/lmc/stream_{0:d}_{1:s}_{2:d}.npy'.format(n, p, j))
                fits[i] = np.poly1d(np.polyfit(stream[0], stream[k],3))
            
            #if k==3:
                #sample = observed.obs[3]>MASK
            #else:
                #sample = np.arange(np.shape(observed.obs)[1], dtype=int)
            
            #dydx_ = (fits[2](observed.obs[0][sample]) - fits[1](observed.obs[0][sample]))/dx[potential]
            #cyd_ = observed.err[k][sample]**2
            
            #dydx_ = (fits[2](ra) - fits[1](ra))/dx[p]
            #cyd_ = err[:,k-1]**2
            
            #dydx = np.hstack((dydx, dydx_))
            #cyd = np.hstack((cyd, cyd_))
            
            dydx[l][(k-1)*Nobs:k*Nobs] = (fits[2](ra) - fits[1](ra))/dx[p]
            cyd[(k-1)*Nobs:k*Nobs] = err[:,k-1]**2
    
    cy = np.diag(cyd)
    cyi = np.linalg.inv(cy)
    
    cxi = np.matmul(dydx, np.matmul(cyi, dydx.T))
    cx = np.linalg.inv(cxi)
    sx = np.sqrt(np.diag(cx))
    
    print(sx)

    #print('{:g}'.format(sx), Ndim)
    np.save('../data/lmc/crb_cxi_{:d}_{:d}'.format(n, Ndim), cxi)

def test_ellipse():
    """"""
    
    th = np.radians(60)
    v = np.array([[np.cos(th),np.sin(th)], [-np.sin(th),np.cos(th)]])
    w = np.array([2,1])
    
    plt.close()
    plt.figure()
    
    theta = np.degrees(np.arccos(v[0][0]))
    print(theta, np.degrees(th))
    e = mpl.patches.Ellipse((0,0), width=w[0]*2, height=w[1]*2, angle=theta, fc='none', ec='k', lw=2)
    plt.gca().add_artist(e)
    
    plt.xlim(-5,5)
    plt.ylim(-5,5)

def plot_crb_ind():
    """"""
    
    cmap = [mpl.cm.Reds, mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Greys]
    labels = ['GD-1', 'Pal 5', 'Triangulum', 'ATLAS']
    
    cxi_tot = np.empty((5,2,2))
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    
    for i, n in enumerate([-1,-2,-3,-4]):
        plt.plot([0,1], [-10,-10], '-', color=cmap[i](0.5), lw=2, label=labels[i])
        for Ndim in range(2,7):
            cxi = np.load('../data/lmc/crb_cxi_{:d}_{:d}.npy'.format(n, Ndim))
            cxi_tot[Ndim-2] += cxi
            cx = np.linalg.inv(cxi)
            
            if np.all(np.isfinite(cx)):
                w, v = np.linalg.eig(cx)
                theta = np.degrees(np.arccos(v[0][0]))
                e = mpl.patches.Ellipse((Ndim-2,0), width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2, angle=theta, fc='none', ec=cmap[i](Ndim/7), lw=2)
                plt.gca().add_artist(e)
    
    plt.plot([0,1], [-10,-10], '-', color=mpl.cm.Purples(0.5), lw=2, label='Combined')
    
    data_label = ['R.A., Dec', 'R.A., Dec, d', 'R.A., Dec, d, $V_r$', 'R.A., Dec, d,\n$V_r$, $\mu_{\\alpha}$', 'R.A., Dec, d,\n$V_r$, $\mu_{\\alpha}$, $\mu_{\delta}$']
    for i in range(5):
        cx = np.linalg.inv(cxi_tot[i])
        
        plt.text(i,3,data_label[i], ha='center', va='center', fontsize='small')
            
        if np.all(np.isfinite(cx)):
            w, v = np.linalg.eig(cx)
            theta = np.degrees(np.arccos(v[0][0]))
            e = mpl.patches.Ellipse((4.75,0), width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2, angle=theta, fc='none', ec=mpl.cm.Purples((i+1)/7), lw=2)
            plt.gca().add_artist(e)
    
    plt.minorticks_on()
    plt.xlabel('Oblateness')
    plt.ylabel('LMC mass ($\\times\;10^{11}\;M_\odot$)')
    plt.legend(frameon=False, loc=2, fontsize='small')
    plt.xlim(-1,5)
    plt.ylim(-4,4)
    
    plt.title('Cramer-Rao bounds for:', fontsize='medium')
    plt.tight_layout()
    plt.savefig('../plots/lmc/crb_summary.png', bbox_inches='tight')


