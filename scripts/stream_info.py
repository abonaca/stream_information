from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import streakline
import myutils
import ffwd

import astropy
import astropy.units as u
from astropy.constants import G
from astropy.table import Table
import astropy.coordinates as coord
import gala.coordinates as gc

import scipy.interpolate
import scipy.optimize

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
    
    def observe(self, mode='cartesian', units=[], errors=[], nstars=-1, sequential=False, present=[], logerr=False, observer = {'z_sun': 0.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_ra': 300*u.deg, 'galcen_dec': 20*u.deg}, footprint='none', rotmatrix=None):
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


# choose appropriate step size when calculating derivatives
def get_steps(Nstep=50, log=False):
    """Return deltax steps
    Paramerets:
    Nstep - 0.5 x number of steps to return (default: 50)
    log - if True, steps are logarithmically spaced (default: False)"""
    
    if log:
        step = np.logspace(-1, 1, Nstep)
    else:
        step = np.linspace(0.1, 10, Nstep)
    
    step = np.concatenate([-step[::-1], step])
    
    return (Nstep, step)

def explore_stepsize(n, p=0, vary='all', Nstep=50, log=False, Ndeg=3):
    """Create models with smoothly varying parameter p values and save polynomial stream tracks"""
    
    pparams0 = [430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s]
    pid, dp = get_varied_pars(vary)

    Nstep, step = get_steps(Nstep=Nstep, log=log)

    Ndim = 6
    fits = np.empty((2*Nstep+1, Ndim-1, Ndeg+1))
    
    # fiducial model
    stream = stream_model(n, pparams0=pparams0)
    for j in range(Ndim-1):
        fits[0][j] = np.poly1d(np.polyfit(stream.obs[0], stream.obs[j+1], Ndeg))
    
    # excursions
    for i, s in enumerate(step):
        pparams = [x for x in pparams0]
        pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
        #print(pparams)
        
        stream = stream_model(n, pparams0=pparams)
        for j in range(Ndim-1):
            fits[i+1][j] = np.poly1d(np.polyfit(stream.obs[0], stream.obs[j+1], Ndeg))
        
    #print(fits[:,0])
    
    np.savez('../data/stepsize_{:d}_{:d}_{:d}'.format(n, p, log), fits)

def get_all_stepsizes(streams=[-1, -2, -3, -4], Npar=11, Nstep=50, log=False, Ndeg=3):
    """"""
    
    for n in streams:
        for p in range(Npar):
            print(n, p)
            explore_stepsize(n, p=p, Nstep=Nstep, log=log)

def dydx_stepsize(n, Nobs=10, vary='all', log=False, ylabels=False, Nstep=50, fiducial=False):
    """Plot derivatives dy/dx as a function of parameter step delta x"""
    
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['font.size'] = 15

    # Load streams
    if n==-1:
        observed = load_gd1(present=[0,1,2,3])
        name = 'GD-1'
    elif n==-3:
        observed = load_tri(present=[0,1,2,3])
        name = 'Triangulum'
    elif n==-4:
        observed = load_atlas(present=[0,1,2,3])
        name = 'ATLAS'
    else:
        observed = load_pal5(present=[0,1,2,3])
        name = 'Pal 5'
    
    ra = np.linspace(np.min(observed.obs[0]), np.max(observed.obs[0]), Nobs)
    
    Ndim = 5
    dimensions = ['$\delta$', 'd', '$V_r$', '$\mu_\\alpha$', '$\mu_\delta$']
    
    Nstep, step = get_steps(log=log, Nstep=Nstep)
    pid, dp = get_varied_pars(vary)
    Npar = len(pid)

    h = 2
    plt.close()
    fig, ax = plt.subplots(Ndim, Npar, figsize=(Npar*h*1.2,Ndim*h), sharex='col')
    #fig, ax = plt.subplots(Npar, Ndim, figsize=(Ndim*h,Npar*h), sharex='col')

    for p in range(Npar):
        fin = np.load('../data/stepsize_{:d}_{:d}_{:d}.npz'.format(n, p, log))
        fits = fin['arr_0']
        
        parameter = get_parlabel(pid)[p]
        units = ['km/s', 'kpc', '', '', '$M_\odot$']
        
        dydx = np.empty((Ndim, 2*Nstep, Nobs))
        
        for i in range(Ndim):
            #for j, s in enumerate(step):
                #if (i==0) & (p==0):
                    #print(j, fits[j+1][i])
                #print((np.poly1d(fits[j+1][i])(ra) - np.poly1d(fits[0][i])(ra))/(dp[p].value*s))
                #dydx[i][j] = (np.poly1d(fits[j+1][i])(ra) - np.poly1d(fits[0][i])(ra))/(dp[p].value*s)
            for j in range(Nstep):
                dydx[i][j] = (np.poly1d(fits[j+1][i])(ra) - np.poly1d(fits[2*Nstep-j][i])(ra))/(dp[p].value*(step[j] - step[2*Nstep-j-1]))
                #if (np.abs(dydx[i][j][0])<1e-5):
                    #dydx[i][j][0] = np.nan
                #else:
                    #dydx[i][j][0] /= dp[p].value*(step[j] - step[2*Nstep-j-1])
                #print(i, j, dydx[i][j]) #, np.poly1d(fits[j+1][i])(ra), np.poly1d(fits[2*Nstep-j][i])(ra))
        
            plt.sca(ax[i][p])
            for k in range(Nobs):
                #plt.plot(step[Nstep:] * dp[p], dydx[i,Nstep:,k], '-', ms=2, color='{}'.format(k/Nobs), lw=1.5)
                plt.plot(step[Nstep:] * dp[p], dydx[i,:Nstep,k], '-', ms=2, color='{}'.format(k/Nobs), lw=1.5)

                #dsigma = 0.1/(dp[p].value*step[Nstep+1:])
                #plt.fill_between(step[Nstep+1:] * dp[p].value, dydx[i,Nstep+1:,k]+dsigma, y2=dydx[i,Nstep+1:,k]-dsigma, color='k', alpha=0.5)
            
            if fiducial:
                plt.axvline(dp[p].value, color='lightsteelblue', lw=2)
            
            if i==Ndim-1:
                if len(units[p]):
                    plt.xlabel('$\Delta$ {} ({})'.format(parameter, units[p]))
                else:
                    plt.xlabel('$\Delta$ {}'.format(parameter))
            if p==0:
                plt.ylabel('d{}/dx'.format(dimensions[i]))
            #if i==0:
                #plt.title('x = {}'.format(parameter), fontsize='medium')
            
            plt.setp(plt.gca().get_yticklabels(), visible=ylabels)
            #if log:
            plt.gca().set_xscale('log')
    
    plt.suptitle(name, fontsize='large')
    plt.tight_layout(h_pad=0.02, w_pad=0.02, rect=(0,0,1,0.95))
    plt.savefig('../plots/stepsize_{:d}_{:d}.pdf'.format(n, log))
    plt.savefig('../plots/stepsize_{:d}_{:d}_{:d}_{:d}.png'.format(n, log, Nobs, fiducial))

    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['font.size'] = 18

def adjacent_models(n, p=0, vary='all', dim=1):
    """"""
    nstep = 100 + np.int64(np.linspace(0,70,4))
    
    pparams0 = [430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s]
    pid, dp = get_varied_pars(vary)

    Nstep = 100
    log = True
    Nstep, step = get_steps(Nstep=Nstep, log=log)

    Ndim = 6
    Ndeg = 3
    c = ['b', 'r']
    ylabel = ['R.A. (deg)', 'Dec (deg)', 'd (kpc)', '$V_r$ (km/s)', '$\mu_\\alpha$ (mas/yr)', '$\mu_\delta$ (mas/yr)']
    
    # fiducial model
    stream_fid = stream_model(n, pparams0=pparams0)
    fit_fid = np.poly1d(np.polyfit(stream_fid.obs[0], stream_fid.obs[dim], Ndeg))
    #print(np.std(stream_fid.obs[dim] - fit_fid(stream_fid.obs[0])))
    
    plt.close()
    fig, ax = plt.subplots(2,5,figsize=(18,4), gridspec_kw = {'height_ratios':[3, 1]}, sharex='col')
    
    for i, s in enumerate(np.array(step)[nstep]):
        color = mpl.cm.magma(i/np.size(nstep))
        pparams = [x for x in pparams0]
        pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
        
        stream = stream_model(n, pparams0=pparams)
        #fit = np.poly1d(np.polyfit(stream.obs[0], stream.obs[dim], Ndeg))
        
        for dim in range(1,6):
            plt.sca(ax[0][dim-1])
            plt.plot(stream.obs[0], stream.obs[dim], 'o', color=color, ms=2, label='$\Delta$ {} = {:4.2g}'.format(get_parlabel([pid[p]])[0], s*dp[p]))
            
            plt.ylabel(ylabel[dim])
            if dim==1:
                plt.legend(fontsize='xx-small', frameon=False, handlelength=0.2)
            
            plt.sca(ax[1][dim-1])
            plt.plot(stream.obs[0], stream.obs[dim] - stream_fid.obs[dim], 'o', color=color, ms=2, label='$\Delta$ {} = {:4.2g}'.format(get_parlabel([pid[p]])[0], s*dp[p]))

            plt.xlabel('R.A. (deg)')
            plt.ylabel('$\Delta$')
    
    plt.tight_layout()
    plt.savefig('../plots/step_sizes_{}_{}.png'.format(n, p))

def adjacent_models_expanded(n, p=0, vary='all', dim=1):
    """"""
    #nstep = 100 + np.int64(np.linspace(30,90,4))
    nstep = 100 + np.int64(np.linspace(10,70,4))
    
    pparams0 = [430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s]
    pid, dp = get_varied_pars(vary)

    Nstep = 100
    log = True
    Nstep, step = get_steps(Nstep=Nstep, log=log)

    Ndim = 6
    Ndeg = 5
    Nobs = 15
    c = ['b', 'r']
    ylabel = ['$\Delta$ R.A. (deg)', '$\Delta$ Dec (deg)', '$\Delta$ d (kpc)', '$\Delta$ $V_r$ (km/s)', '$\Delta$ $\mu_\\alpha$ (mas/yr)', '$\Delta$ $\mu_\delta$ (mas/yr)']
    
    # fiducial model
    stream_fid = stream_model(n, pparams0=pparams0)
    poly_fid = np.poly1d(np.polyfit(stream_fid.obs[0], stream_fid.obs[dim], Ndeg))
    
    isort = np.argsort(stream_fid.obs[0])
    ra = np.linspace(np.min(stream_fid.obs[0])*1.05, np.max(stream_fid.obs[0])*0.95, Nobs)
    k = 3
    t = np.r_[(stream_fid.obs[0][isort][0],)*(k+1), ra, (stream_fid.obs[0][isort][-1],)*(k+1)]
    spline_fid = scipy.interpolate.make_lsq_spline(stream_fid.obs[0][isort], stream_fid.obs[dim][isort], t, k=k)
    #print(np.std(stream_fid.obs[dim] - fit_fid(stream_fid.obs[0])))
    
    plt.close()
    fig, ax = plt.subplots(2,4,figsize=(16,5), gridspec_kw = {'height_ratios':[3, 1]}, sharey='row', sharex='col')
    
    for i, s in enumerate(np.array(step)[nstep]):
        color = mpl.cm.magma(i/np.size(nstep))
        #color = 'k'
        pparams = [x for x in pparams0]
        pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
        
        stream = stream_model(n, pparams0=pparams)
        poly = np.poly1d(np.polyfit(stream.obs[0], stream.obs[dim], Ndeg))
        isort_ = np.argsort(stream.obs[0])
        ra_ = np.linspace(np.min(stream.obs[0])*1.05, np.max(stream.obs[0])*0.95, Nobs)
        k = 3
        t = np.r_[(stream.obs[0][isort_][0],)*(k+1), ra_, (stream.obs[0][isort_][-1],)*(k+1)]
        spline = scipy.interpolate.make_lsq_spline(stream.obs[0][isort_], stream.obs[dim][isort_], t, k=k)
        
        for j in range(4):
            plt.sca(ax[0][j])
            plt.plot(stream.obs[0], stream.obs[dim], 'o', color=color, ms=2)
        #plt.plot(stream.obs[0], stream.obs[dim], 'ko')
        #plt.plot(stream.obs[0], poly(stream.obs[0]), 'r.')
        #plt.plot(stream.obs[0], spline(stream.obs[0]), 'b.')
        #plt.plot(stream_fid.obs[0], stream_fid.obs[dim], 'wo', mec='k', ms=4, mew=0.5, alpha=0.1)

            #if j==0:
                #plt.title('$\Delta$ {} = {:4.2g}'.format(get_parlabel([pid[p]])[0], s*dp[p]), fontsize='medium')
        if i==0:
            plt.sca(ax[0][i])
            plt.ylabel(ylabel[dim])
        
        plt.sca(ax[1][i])
        plt.plot(stream.obs[0], stream.obs[dim] - stream_fid.obs[dim], 'ko')
        plt.plot(stream.obs[0], poly(stream.obs[0]) - poly_fid(stream.obs[0]), 'r.')
        plt.plot(stream.obs[0], spline(stream.obs[0]) - spline_fid(stream.obs[0]), 'b.')
        
        #plt.ylim(-10,10)
        plt.xlabel('R.A. (deg)')
        if i==0:
            plt.ylabel('$\Delta$')
    #plt.legend(fontsize='small', frameon=False)
    
    plt.tight_layout()
    plt.savefig('../plots/step_sizes_expanded_{}_{}.png'.format(n, dim))

def crb_stepsize(n, Nobs=10, vary='potential', log=False, ylabels=False, Nstep=50):
    """Plot derivatives dy/dx as a function of parameter step delta x"""
    
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['font.size'] = 15

    # Load streams
    if n==-1:
        observed = load_gd1(present=[0,1,2,3])
        name = 'GD-1'
    elif n==-3:
        observed = load_tri(present=[0,1,2,3])
        name = 'Triangulum'
    elif n==-4:
        observed = load_atlas(present=[0,1,2,3])
        name = 'ATLAS'
    else:
        observed = load_pal5(present=[0,1,2,3])
        name = 'Pal 5'
    
    # mock observations
    ra = np.linspace(np.min(observed.obs[0]), np.max(observed.obs[0]), Nobs)
    sig_obs = np.array([0.1, 2, 5, 0.1, 0.1])
    err = np.tile(sig_obs, Nobs).reshape(Nobs,-1)
    
    Ndim = 5
    dimensions = ['$\delta$', 'd', '$V_r$', '$\mu_\\alpha$', '$\mu_\delta$']
    
    Nstep, step = get_steps(log=log, Nstep=Nstep)
    Npar = 5
    
    h = 2
    plt.close()
    fig, ax = plt.subplots(Ndim, Npar, figsize=(Npar*h,Ndim*h), sharex='col')

    pid, dp = get_varied_pars(vary)
    Nvar = len(pid)

    Ndata = Nobs * (Ndim - 1)
    dydx = np.empty((Nvar, Ndata))
    cyd = np.empty(Ndata)
    
    #for js, s in enumerate(step):
    s = step[Nstep+2]
    
    # find derivatives, uncertainties
    for k in range(1,Ndim):
        fits = [None]*2
        for l, p in enumerate(pid):
            fin = np.load('../data/stepsize_{:d}_{:d}_{:d}.npz'.format(n, p, log))
            fits = fin['arr_0']
            
            #for i, j in enumerate(sorted([0,1*Nex*sign])):
                #stream = np.load('../data/models/stream_{0:d}_{1:d}_{2:d}.npy'.format(n, pid[l], j))
                #fits[i] = np.poly1d(np.polyfit(stream[0], stream[k],3))
            
            dydx[l][(k-1)*Nobs:k*Nobs] = (fits[1](ra) - fits[0](ra))/(dp[l].value*Nex)
            cyd[(k-1)*Nobs:k*Nobs] = err[:,k-1]**2

    cy = np.diag(cyd)
    cyi = np.linalg.inv(cy)
    
    cxi = np.matmul(dydx, np.matmul(cyi, dydx.T))

    cx = np.linalg.inv(cxi)
    sx = np.sqrt(np.diag(cx))

    for p in range(Npar):
        fin = np.load('../data/stepsize_{:d}_{:d}_{:d}.npz'.format(n, p, log))
        fits = fin['arr_0']
        
        pid, dp = get_varied_pars(vary)
        parameter = get_parlabel(pid)[p]
        units = ['km/s', 'kpc', '', '', '$M_\odot$']
        
        dydx = np.empty((Ndim, Nobs))
        
        for i in range(Ndim):
            dydx[i][j] = (np.poly1d(fits[j+1][i])(ra) - np.poly1d(fits[0][i])(ra))/(dp[p].value*s)
            
            
            dydx[l][(k-1)*Nobs:k*Nobs] = (fits[1](ra) - fits[0](ra))/(dp[l].value*Nex)
            cyd[(k-1)*Nobs:k*Nobs] = err[:,k-1]**2
        
            #plt.sca(ax[i][p])
            #for k in range(Nobs):
                #plt.plot(step[Nstep:] * dp[p], dydx[i,Nstep:,k], '-', ms=2, color='{}'.format(k/Nobs), lw=1.5)

            #if i==Ndim-1:
                #if len(units[p]):
                    #plt.xlabel('$\Delta$ {} ({})'.format(parameter, units[p]))
                #else:
                    #plt.xlabel('$\Delta$ {}'.format(parameter))
            #if p==0:
                #plt.ylabel('d{}/dx'.format(dimensions[i]))
            #if i==0:
                #plt.title('x = {}'.format(parameter), fontsize='medium')
            
            #plt.setp(plt.gca().get_yticklabels(), visible=ylabels)
            #plt.gca().set_xscale('log')
    
    plt.suptitle(name, fontsize='large')
    plt.tight_layout(h_pad=0.02, w_pad=0.02, rect=(0,0,1,0.95))
    plt.savefig('../plots/crb_stepsize_{:d}_{:d}.pdf'.format(n, log))

    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['font.size'] = 18

##############################
# b-spline based derivatives #

pparams_fid = [430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s]

# fit b-spline to a stream model
def fit_bspline(n, pparams=[430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s], dt=1*u.Myr, align=False, save='', graph=False, graphsave='', fiducial=False):
    """"""
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

            plt.plot(stream.obs[0][isort], stream.obs[i+1][isort] - stream_fid.obs[i+1][fidsort], 'wo', mec='k', alpha=0.1)
            plt.plot(stream.obs[0], stream.obs[i+1] - yref, 'ko')
            
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
    

def fitbyt_bspline(n, pparams=[430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s], dt=1*u.Myr, align=False, save='', graph=False, graphsave='', fiducial=False):
    """"""
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
        xup = [180, 110, 80, 80]
        xlims = [xup[np.abs(n)-1], 40]

    return (xlims, ylims)

# fit bspline for a range of integration time steps
def bspline_atdt(n):
    """"""
    dt = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1])*u.Myr
    
    for dt_ in dt:
        fout = 'bspline_n{:d}_dt{:.2f}'.format(n, dt_.value)
        fit_bspline(n, dt=dt_, save=fout)

def analyze_atdt(n):
    """"""
    stream = stream_model(n, dt=0.01*u.Myr)
    Ndim = np.shape(stream.obs)[0]

    ylabel = ['R.A. (deg)', 'Dec (deg)', 'd (kpc)', '$V_r$ (km/s)', '$\mu_\\alpha$ (mas/yr)', '$\mu_\delta$ (mas/yr)']
    Nobs = 10
    k = 3
    
    isort = np.argsort(stream.obs[0])
    ra = np.linspace(np.min(stream.obs[0])*1.05, np.max(stream.obs[0])*0.95, Nobs)
    t = np.r_[(stream.obs[0][isort][0],)*(k+1), ra, (stream.obs[0][isort][-1],)*(k+1)]
    
    fiducial = [None]*(Ndim-1)
    for j in range(Ndim-1):
        fiducial[j] = scipy.interpolate.make_lsq_spline(stream.obs[0][isort], stream.obs[j+1][isort], t, k=k)
    
    plt.close()
    fig, ax = plt.subplots(2,5,figsize=(16,5), sharex='col', gridspec_kw = {'height_ratios':[3, 1]})
    
    for i in range(Ndim-1):
        plt.sca(ax[0][i])
        plt.plot(stream.obs[0], stream.obs[i+1], 'wo', mec='k', mew=0.1)
        
        plt.ylabel(ylabel[i+1])
        
        plt.sca(ax[1][i])
        plt.plot(stream.obs[0][isort], stream.obs[i+1][isort] - fiducial[i](stream.obs[0][isort]), 'wo', mec='k', mew=0.1)
        plt.xlabel('R.A. (deg)')
        plt.ylabel('$\Delta$')
    
    dt = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1])*u.Myr
    for j, dt_ in enumerate(dt):
        fin = np.load('../data/bspline_n{:d}_dt{:.2f}.npz'.format(n, dt_.value))
        fits = fin['fits']
        
        for i in range(Ndim-1):
            plt.sca(ax[0][i])
            plt.plot(stream.obs[0][isort], fits[i](stream.obs[0][isort]), '-', color=mpl.cm.bone(j/np.size(dt)), lw=2, label='{}'.format(dt_))
            
            if i==0:
                plt.legend(frameon=True, framealpha=0.3, fontsize='xx-small', handlelength=0.5)
            
            plt.sca(ax[1][i])
            plt.plot(stream.obs[0][isort], fits[i](stream.obs[0][isort]) - fiducial[i](stream.obs[0][isort]), '-', color=mpl.cm.bone(j/np.size(dt)), lw=2)
    
    plt.tight_layout()
    plt.savefig('../plots/bspline_atdt_n{:d}.png'.format(n))

# fit bspline for a range of potential parameter steps
def bspline_atdx(n, p=0, Nstep=20, log=True, dt=0.2*u.Myr, vary='halo', verbose=False):
    """"""
    pparams0 = [430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s]
    pid, dp = get_varied_pars(vary)

    Nstep, step = get_steps(Nstep=Nstep, log=log)
    
    # excursions
    for i, s in enumerate(step):
        pparams = [x for x in pparams0]
        pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
        fout = 'bspline_n{:d}_p{:d}_dx{:d}_log{:d}_align'.format(n, p, i, log)
        if verbose: print(fout)
        fit_bspline(n, pparams=pparams, dt=dt, save=fout, graph=False, graphsave=fout, fiducial=True, align=True)

def analyze_atdx(n, Nobs=10, log=True, Nstep=20, vary='halo', ylabels=True, fiducial=False, align=False):
    """"""
    
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['font.size'] = 15

    stream = stream_model(n, dt=1*u.Myr)
    Ndim = np.shape(stream.obs)[0] - 1

    ylabel = ['R.A. (deg)', 'Dec (deg)', 'd (kpc)', '$V_r$ (km/s)', '$\mu_\\alpha$ (mas/yr)', '$\mu_\delta$ (mas/yr)']
    names = {'-1':'GD-1', '-2':'Pal 5', '-3':'Triangulum', '-4':'ATLAS'}
    name = names['{:d}'.format(n)]
    k = 3
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    isort = np.argsort(stream.obs[0])
    ra = np.linspace(np.min(stream.obs[0])*1.05, np.max(stream.obs[0])*0.95, Nobs)
    t = np.r_[(stream.obs[0][isort][0],)*(k+1), ra, (stream.obs[0][isort][-1],)*(k+1)]
    
    Ndim = 5
    dimensions = ['$\delta$', 'd', '$V_r$', '$\mu_\\alpha$', '$\mu_\delta$']
    
    Nstep, step = get_steps(log=log, Nstep=Nstep)
    pid, dp = get_varied_pars(vary)
    Npar = len(pid)
    best_step = optimal_stepsize(n, Nobs=Nobs, log=log, Nstep=Nstep, vary=vary)

    h = 2
    plt.close()
    fig, ax = plt.subplots(Ndim, Npar, figsize=(Npar*h*1.2,Ndim*h), sharex='col')

    for p in range(Npar):
        parameter = get_parlabel(pid)[p]
        units = ['km/s', 'kpc', '', '', '$M_\odot$']
        
        dydx = np.empty((Ndim, Nstep, Nobs))
        
        for j in range(Nstep):
            fin1 = np.load('../data/bspline_n{:d}_p{:d}_dx{:d}_log{:d}{:s}.npz'.format(n, p, j, log, alabel))
            fin2 = np.load('../data/bspline_n{:d}_p{:d}_dx{:d}_log{:d}{:s}.npz'.format(n, p, 2*Nstep - j - 1, log, alabel))
            fits1 = fin1['fits']
            fits2 = fin2['fits']
            for i in range(Ndim):
                dydx[i][j] = (fits1[i](ra) - fits2[i](ra))/(dp[p].value*(step[j] - step[2*Nstep-j-1]))
        
        for i in range(Ndim):
            plt.sca(ax[i][p])
            for k in range(Nobs):
                plt.plot(step[Nstep:][::-1] * dp[p], dydx[i,:,k], '-', ms=2, color='{}'.format(k/Nobs), lw=1.5)
            
            if i==Ndim-1:
                if len(units[p]):
                    plt.xlabel('$\Delta$ {} ({})'.format(parameter, units[p]))
                else:
                    plt.xlabel('$\Delta$ {}'.format(parameter))
            if p==0:
                plt.ylabel('d{}/dx'.format(dimensions[i]))

            plt.setp(plt.gca().get_yticklabels(), visible=ylabels)
            plt.gca().set_xscale('log')
        
            if fiducial:
                plt.axvline(step[best_step[p]]*dp[p].value, ls='-', lw=2, color='orange', zorder=0)
    
    plt.suptitle(name, fontsize='large')
    plt.tight_layout(h_pad=0.02, w_pad=0.02, rect=(0,0,1,0.95))
    plt.savefig('../plots/stepsize_{:d}_{:d}_{:d}_{:d}{:s}.png'.format(n, log, Nobs, fiducial, alabel))

    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['font.size'] = 18

def optimal_stepsize(n, Nobs=10, log=True, Nstep=20, vary='halo'):
    """"""
    Nstep, step = get_steps(log=log, Nstep=Nstep)
    pid, dp = get_varied_pars(vary)
    Npar = len(pid)
    
    stepid = np.arange(1,Nstep-1,1, dtype=np.int64)[::-1]
    best_step = np.ones(Npar, dtype=np.int64) * np.nan
    
    stream = stream_model(n, dt=1*u.Myr)
    Ndim = np.shape(stream.obs)[0] - 1
    ra = np.linspace(np.min(stream.obs[0])*1.05, np.max(stream.obs[0])*0.95, Nobs)
    
    if Nobs>1:
        #sig_obs = np.empty((Nobs, Ndim))
        #for i in range(Ndim):
            
        sig_obs_ = np.array([[0.01, 0.1, 0.1, 0.01, 0.01]])
        sig_obs = np.tile(sig_obs_, Nobs).reshape(Nobs, -1)
    else:
        sig_obs = np.array([[0.01, 0.1, 0.1, 0.01, 0.01]])
    
    for p in range(Npar):
        for s in stepid:
            if ~np.isfinite(best_step[p]):
                fitdn1 = np.load('../data/bspline_n{:d}_p{:d}_dx{:d}_log{:d}.npz'.format(n, p, s - 1, log))['fits']
                fitdn2 = np.load('../data/bspline_n{:d}_p{:d}_dx{:d}_log{:d}.npz'.format(n, p, 2*Nstep - s, log))['fits']
                fitup1 = np.load('../data/bspline_n{:d}_p{:d}_dx{:d}_log{:d}.npz'.format(n, p, s + 1, log))['fits']
                fitup2 = np.load('../data/bspline_n{:d}_p{:d}_dx{:d}_log{:d}.npz'.format(n, p, 2*Nstep - s - 2, log))['fits']
                
                criterion = np.zeros(Ndim, dtype=bool)
                
                for i in range(Ndim):
                    dydn = fitdn1[i](ra) - fitdn2[i](ra)
                    dyup = fitup1[i](ra) - fitup2[i](ra)
                    dy0 = np.array([x for x in sig_obs[:,i]])
                    
                    #dy0 /= (step[s] - step[2*Nstep - s - 1])
                    #dy0 = np.abs(dy0)
                    #dydn *= (step[s - 1] - step[2*Nstep - s]) / (step[s] - step[2*Nstep - s - 1])
                    #dyup *= (step[s + 1] - step[2*Nstep - s - 2]) / (step[s] - step[2*Nstep - s - 1])
                    
                    criterion[i] = np.all(np.less(np.abs(dydn - dyup), dy0))
                    #print(np.less(np.abs(dydn - dyup), dy0))
                    #print(s, dydn, dyup, dydn - dyup, dy0)
                
                if np.all(criterion):
                    best_step[p] = 2*Nstep - s - 1
                    #print(s, step[s], step[2*Nstep - s - 1])
    
    #print(best_step, step[np.int64(best_step)])
    return best_step.astype(int)
    
def plot_steps(n, p=0, Nstep=20, log=True, dt=0.2*u.Myr, vary='halo', verbose=False, align=True):
    """Plot stream for different values of a potential parameter"""
    
    if align:
        rotmatrix = np.load('../data/rotmatrix_{}.npy'.format(n))
    else:
        rotmatrix = None
    
    pparams0 = [430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s]
    pid, dp = get_varied_pars(vary)
    plabel = get_parlabel(pid[p])

    Nstep, step = get_steps(Nstep=Nstep, log=log)
    
    plt.close()
    fig, ax = plt.subplots(5,5,figsize=(20,10), sharex=True, gridspec_kw = {'height_ratios':[3, 1, 1, 1, 1]})
    
    # fiducial model
    stream0 = stream_model(n, pparams0=pparams0, dt=dt, rotmatrix=rotmatrix)
    
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
    plt.savefig('../plots/observable_steps_{:d}_p{:d}_Ns{:d}.png'.format(n, p, Nstep))

def step_convergence():
    """"""

# crbs using bspline
def bspline_crb(n, Ndim=6, istep=15, Nstep=20, log=True, vary='halo', Nobs=50, verbose=False, align=False):
    """"""
    if align:
        alabel = '_align'
    else:
        alabel = ''
        
    # typical uncertainties
    sig_obs = np.array([0.1, 2, 5, 0.1, 0.1])
    
    # mock observations
    stream = stream_model(n, dt=1*u.Myr)
    ra = np.linspace(np.min(stream.obs[0])*1.05, np.max(stream.obs[0])*0.95, Nobs)
    err = np.tile(sig_obs, Nobs).reshape(Nobs,-1)

    Nstep, step = get_steps(log=log, Nstep=Nstep)
    pid, dp = get_varied_pars(vary)
    Nvar = len(pid)
    
    Ndata = Nobs * (Ndim - 1)
    dydx = np.empty((Nvar, Ndata))
    cyd = np.empty(Ndata)
    
    # find derivatives, uncertainties
    for k in range(1,Ndim):
        fits = [None]*2
        for l, p in enumerate(pid):
            
            fin1 = np.load('../data/bspline_n{:d}_p{:d}_dx{:d}_log{:d}{:s}.npz'.format(n, l, istep, log, alabel))
            fin2 = np.load('../data/bspline_n{:d}_p{:d}_dx{:d}_log{:d}{:s}.npz'.format(n, l, 2*Nstep - istep - 1, log, alabel))
            fits1 = fin1['fits']
            fits2 = fin2['fits']
            
            dydx[l][(k-1)*Nobs:k*Nobs] = (fits1[k-1](ra) - fits2[k-1](ra))/(dp[l].value*(step[istep] - step[2*Nstep-istep-1]))
            cyd[(k-1)*Nobs:k*Nobs] = err[:,k-1]**2
    
    cy = np.diag(cyd)
    cyi = np.linalg.inv(cy)
    
    cxi = np.matmul(dydx, np.matmul(cyi, dydx.T))

    cx = np.linalg.inv(cxi)
    sx = np.sqrt(np.diag(cx))
    
    if verbose:
        print(np.diag(cxi))
        print(np.linalg.det(cxi))
        print(np.allclose(cxi, cxi.T), np.allclose(cx, cx.T))

    np.save('../data/crb/bspline_cxi{:s}_{:d}_{:d}_{:d}'.format(alabel, n, Ndim, istep), cxi)


############################
# great circle orientation #

def find_greatcircle(n, pparams=pparams_fid, dt=0.2*u.Myr):
    """"""
    
    stream = stream_model(n, pparams0=pparams, dt=dt)
    
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
    
    np.save('../data/rotmatrix_{:d}'.format(n), R)
    
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
    plt.savefig('../plots/gc_orientation_{:d}.png'.format(n))

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

################
# stream track #
def stream_track(n, dim=1, align=False):
    """"""
    pparams0 = [430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s]
    Ndeg = 5
    Nobs = 15
    if align:
        rotmatrix = np.load('../data/rotmatrix_{:d}.npy'.format(n))
    else:
        rotmatrix = None
    
    ylabel = ['R.A. (deg)', 'Dec (deg)', 'd (kpc)', '$V_r$ (km/s)', '$\mu_\\alpha$ (mas/yr)', '$\mu_\delta$ (mas/yr)']
    
    # fiducial model
    stream = stream_model(n, pparams0=pparams0, rotmatrix=rotmatrix)
    isort = np.argsort(stream.obs[0])
    ra = np.linspace(np.min(stream.obs[0])*1.05, np.max(stream.obs[0])*0.95, Nobs)
    k = 3
    t = np.r_[(stream.obs[0][isort][0],)*(k+1), ra, (stream.obs[0][isort][-1],)*(k+1)]
    
    plt.close()
    fig, ax = plt.subplots(2, 5, figsize=(16,4), gridspec_kw = {'height_ratios':[3, 1]}, sharex='col')

    for dim in range(1,6):
        fit_poly = np.poly1d(np.polyfit(stream.obs[0], stream.obs[dim], Ndeg))
        fit_spline = scipy.interpolate.make_lsq_spline(stream.obs[0][isort], stream.obs[dim][isort], t, k=k)
        
        plt.sca(ax[0][dim-1])
        plt.plot(stream.obs[0], stream.obs[dim], 'ko', label='stream model')
        plt.plot(stream.obs[0], fit_poly(stream.obs[0]), 'r.', label='polynomial fit')
        plt.plot(stream.obs[0], fit_spline(stream.obs[0]), 'b.', label='b-spline fit')
        plt.ylabel(ylabel[dim])
        if dim==1:
            plt.legend(frameon=False, fontsize='x-small', handlelength=0.2)
        
        plt.sca(ax[1][dim-1])
        plt.plot(stream.obs[0], stream.obs[dim] - fit_poly(stream.obs[0]), 'r.')
        plt.plot(stream.obs[0], stream.obs[dim] - fit_spline(stream.obs[0]), 'b.')
        
        plt.xlabel('R.A. (deg)')
        plt.ylabel('$\Delta$')
    
    plt.tight_layout()
    plt.savefig('../plots/stream_track_{}.png'.format(n))

# plot model
def stream_model(n, pparams0=[430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s], dt=1*u.Myr, rotmatrix=None, graph=False):
    """"""
    
    obsmode = 'equatorial'
    footprint = ''
    
    # Load streams
    if n==-1:
        observed = load_gd1(present=[0,1,2,3])
        age = 3*u.Gyr
        mi = 2e4*u.Msun
        mf = 2e-1*u.Msun
        x0, v0 = gd1_coordinates()
        xlims = [[190, 130], [0, 350]]
        ylims = [[15, 65], [5, 10], [-250, 150], [0, 250]]
        loc = 2
        name = 'GD-1'
    elif n==-3:
        observed = load_tri(present=[0,1,2,3])
        age = 5*u.Gyr
        mi = 2e4*u.Msun
        mf = 2e-1*u.Msun
        x0, v0 = tri_coordinates()
        xlims = [[25, 19], [0, 350]]
        ylims = [[10, 50], [20, 45], [-175, -50], [0, 250]]
        loc = 1
        name = 'Triangulum'
    elif n==-4:
        observed = load_atlas(present=[0,1,2,3])
        age = 2*u.Gyr
        mi = 2e4*u.Msun
        mf = 2e-1*u.Msun
        x0, v0 = atlas_coordinates()
        xlims = [[35, 10], [0, 350]]
        ylims = [[-40, -20], [15, 25], [50, 200], [0, 250]]
        loc = 3
        name = 'ATLAS'
    else:
        observed = load_pal5(present=[0,1,2,3])
        age = 2.7*u.Gyr
        mi = 1e5*u.Msun
        mf = 2e4*u.Msun
        x0, v0 = pal5_coordinates2()
        xlims = [[245, 225], [0, 350]]
        ylims = [[-4, 10], [21, 27], [-80, -20], [0, 250]]
        loc = 3
        name = 'Pal 5'
        
    ######################
    # Create mock stream

    #potential parameters
    potential = 'lmc'
    mlmc, xlmc = lmc_properties()
    # fixed: bulge and disk
    # Kupper et al. (2015)
    pf = [3.4e10, 0.7, 1e11, 6.5, 0.26]
    # ~MWPotential2014
    pf = [0.5e10, 0.7, 6.8e10, 3, 0.28]
    uf = [u.Msun, u.kpc, u.Msun, u.kpc, u.kpc]
    pfixed = [x*y for x,y in zip(pf, uf)]
    # free: halo + lmc mass ; fixed again: lmc position
    pparams = pfixed + pparams0[:7] + [x for x in xlmc]
    
    # progenitor parameters
    for i in range(3):
        x0[i] += pparams0[7+i].to(u.kpc).value
        v0[i] += pparams0[10+i].to(u.km/u.s).value
    pparams0[7:]
    
    # MWPotential2014
    #R0 ( kpc) 8 fixed
    #vc(R0) ( km s-1
    #) 220 fixed
    #fb 0.05 . . .
    #fd 0.60 . . .
    #fh 0.35 . . .
    #a ( kpc) 3.0 . . .
    #b ( pc) 280 . . .
    #Halo rs ( kpc) 16 . . .
    #Mvir (1012 M) 0.8 . . .
    #rvir ( kpc) 245
    # pf = [0.5e10, 0.7, 6.8e10, 3, 0.28]
    
    # Licquia & Newman (2015)
    #pf = [0.9e10, 5.2e10]
    
    distance = 8.3*u.kpc
    mr = pparams[5]**2 * pparams[6] / G * (np.log(1 + distance/pparams[6]) - distance/(distance + pparams[6]))
    vc_ = np.sqrt(G*mr/distance)
    vsun['vcirc'] = np.sqrt((198*u.km/u.s)**2 + vc_**2)
    
    params = {'generate': {'x0': x0*u.kpc, 'v0': v0*u.km/u.s, 'potential': potential, 'pparams': pparams, 'minit': mi, 'mfinal': mf, 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': dt, 'age': age, 'nstars': 400, 'integrator': 'lf'}, 'observe': {'mode': obsmode, 'nstars':-1, 'sequential':True, 'errors': [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,3,4,5], 'observer': mw_observer, 'footprint': footprint, 'rotmatrix': rotmatrix}}
    
    stream = Stream(**params['generate'])
    stream.generate()
    stream.observe(**params['observe'])
    
    #########################
    # Plot observed streams
    
    if graph:
        modcol = 'r'
        obscol = '0.5'
        ylabel = ['Dec (deg)', 'Distance (kpc)', 'Radial velocity (km/s)']
        Ndim = np.shape(observed.obs)[0]
    
        plt.close()
        fig, ax = plt.subplots(1, 3, figsize=(12,4))
        
        for i in range(3):
            plt.sca(ax[i])
            
            plt.xlim(xlims[0][0], xlims[0][1])
            plt.ylim(ylims[i][0], ylims[i][1])
            
            plt.xlabel('R.A. (deg)')
            plt.ylabel(ylabel[i])
            
            if i<Ndim-1:
                if i<2:
                    sample = np.arange(np.size(observed.obs[0]), dtype=int)
                else:
                    sample = observed.obs[i+1]>MASK
                plt.plot(observed.obs[0][sample], observed.obs[i+1][sample], 's', color=obscol, mec='none', ms=8)
            
            plt.plot(stream.obs[0], stream.obs[i+1], 'o', color=modcol, mec='none', ms=4)
        
        plt.tight_layout()
    
    return stream

def model_excursions(n, Nex=1, vary='potential'):
    """Create models around a fiducial halo potential"""
    
    pparams0 = [430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.kpc, 0*u.kpc, 0*u.kpc, 0*u.km/u.s, 0*u.km/u.s, 0*u.km/u.s]
    Npar = len(pparams0)
    
    pid, dp = get_varied_pars(vary)
    Nvar = len(pid)
    
    for i in range(Nvar):
        for k in range(-Nex, Nex+1):
            pparams = [x for x in pparams0]
            pparams[pid[i]] = pparams[pid[i]] + k*dp[i]
            
            stream = stream_model(n, pparams0=pparams)
            
            np.save('../data/models/stream_{:d}_{:d}_{:d}'.format(n, pid[i], k), stream.obs)

def get_varied_pars(vary):
    """Return indices and steps for a preset of varied parameters
    Parameters:
    vary - string setting the parameter combination to be varied, options: 'potential', 'progenitor', 'all'"""
    
    if vary=='potential':
        pid = [0,1,3,5,6]
        dp = [20*u.km/u.s, 2*u.kpc, 0.05*u.Unit(1), 0.05*u.Unit(1), 0.4e11*u.Msun]
    if vary=='halo':
        pid = [0,1,3,5]
        dp = [20*u.km/u.s, 2*u.kpc, 0.05*u.Unit(1), 0.05*u.Unit(1)]
        dp = [5*u.km/u.s, 1*u.kpc, 0.005*u.Unit(1), 0.01*u.Unit(1)]
        #dp = [20*u.km/u.s, 2*u.kpc, 0.1*u.Unit(1), 0.1*u.Unit(1)]
    elif vary=='progenitor':
        pid = [7,8,9,10,11,12]
        dp = [0.05*u.kpc for x in range(3)] + [2*u.km/u.s for x in range(3)]
    elif vary=='all':
        pid = []
        dp = []
        for v in ['potential', 'progenitor']:
            o1, o2 = get_varied_pars(v)
            pid += o1
            dp += o2
    else:
        pid = []
        dp = []
    
    return (pid, dp)

def get_parlabel(pid):
    """Return label for a list of parameter ids
    Parameter:
    pid - list of parameter ids"""
    
    master = ['$V_h$', '$R_h$', '$\phi$', '$q_1$', '$q_2$', '$q_z$', '$M_{lmc}$', '$X_p$', '$Y_p$', '$Z_p$', '$V_{x_p}$', '$V_{y_p}$', '$V_{z_p}$']
    
    if pid is list:
        labels = []
        
        for i in pid:
            labels += [master[i]]
    else:
        labels = master[pid]
    
    return labels

def plot_model(n, vary='potential'):
    """"""
    
    # Load streams
    if n==-1:
        observed = load_gd1(present=[0,1,2,3])
        age = 1.4*u.Gyr
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
        age = 5*u.Gyr
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
        age = 2*u.Gyr
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
        age = 2.7*u.Gyr
        mi = 1e5*u.Msun
        mf = 2e4*u.Msun
        x0, v0 = pal5_coordinates2()
        xlims = [[245, 225], [0, 350]]
        ylims = [[-4, 10], [21, 27], [-80, -20], [0, 250]]
        loc = 3
        name = 'Pal 5'
        footprint = 'sdss'
    
    modcol = 'r'
    obscol = '0.5'
    ylabel = ['Dec', 'd', '$V_r$']
    Ndim = np.shape(observed.obs)[0]
    
    pid, dp = get_varied_pars(vary)
    labels = get_parlabel(pid)
    Nvar = len(pid)

    plt.close()
    fig, ax = plt.subplots(Nvar, 3, figsize=(8,12), sharex=True, sharey='col')
    
    # plot data
    for i in range(3):
        for j in range(Nvar):
            plt.sca(ax[j][i])
            
            plt.xlim(xlims[0][0], xlims[0][1])
            plt.ylim(ylims[i][0], ylims[i][1])
            
            if j==Nvar-1:
                plt.xlabel('R.A.', fontsize='small')
            
            plt.ylabel(ylabel[i], fontsize='small')
            
            if i<Ndim-1:
                if i<2:
                    sample = np.arange(np.size(observed.obs[0]), dtype=int)
                else:
                    sample = observed.obs[i+1]>MASK
                plt.plot(observed.obs[0][sample], observed.obs[i+1][sample], 's', color=obscol, mec='none', ms=8)
    
    # plot models
    for i, p in enumerate(pid):
        # fiducial
        stream = np.load('../data/models/stream_{:d}_{:d}_{:d}.npy'.format(n, p, 0))
        for k in range(3):
            plt.sca(ax[i][k])
            plt.plot(stream[0], stream[k+1], 'o', color='k', mec='none', ms=2)
        
        for e in range(1, 6):
            for s in (-1, 1):
                stream = np.load('../data/models/stream_{:d}_{:d}_{:d}.npy'.format(n, p, s*e))
                modcol = mpl.cm.RdBu(0.5 + s*e/10)
                
                for k in range(3):
                    plt.sca(ax[i][k])
                    plt.plot(stream[0], stream[k+1], 'o', color=modcol, mec='none', ms=1)
        
        ax[i][0].annotate(labels[i], xy=(0, 0.5), xytext=(-ax[i][0].yaxis.labelpad - 5, 0), xycoords=ax[i][0].yaxis.label, textcoords='offset points', fontsize='small', ha='right', va='center', rotation='vertical')
    
    plt.tight_layout(h_pad=0, w_pad=0.2, rect=[0.02,0,1,1])
    plt.savefig('../plots/param_variation_{:2d}_{:s}.png'.format(n, vary))

def crb_all(n, Ndim=6, Nex=1, sign=1, vary='potential'):
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
    
    # typical uncertainties
    sig_obs = np.array([0.1, 2, 5, 0.1, 0.1])
    
    # mock observations
    Nobs = 50
    ra = np.linspace(np.min(observed.obs[0]), np.max(observed.obs[0]), Nobs)
    err = np.tile(sig_obs, Nobs).reshape(Nobs,-1)
    
    pid, dp = get_varied_pars(vary)
    Nvar = len(pid)
    
    Ndata = Nobs * (Ndim - 1)
    dydx = np.empty((Nvar, Ndata))
    cyd = np.empty(Ndata)
    
    # find derivatives, uncertainties
    for k in range(1,Ndim):
        fits = [None]*2
        for l, p in enumerate(pid):
            
            for i, j in enumerate(sorted([0,1*Nex*sign])):
                stream = np.load('../data/models/stream_{0:d}_{1:d}_{2:d}.npy'.format(n, pid[l], j))
                fits[i] = np.poly1d(np.polyfit(stream[0], stream[k],3))
            
            dydx[l][(k-1)*Nobs:k*Nobs] = (fits[1](ra) - fits[0](ra))/(dp[l].value*Nex)
            cyd[(k-1)*Nobs:k*Nobs] = err[:,k-1]**2
    
    cy = np.diag(cyd)
    cyi = np.linalg.inv(cy)
    
    cxi = np.matmul(dydx, np.matmul(cyi, dydx.T))

    cx = np.linalg.inv(cxi)
    sx = np.sqrt(np.diag(cx))
    
    plt.close()
    plt.figure()
    
    t = 1 - cx[:,0]/cx[0,:]
    plt.plot(t, 'ko')
    
    print(t)
    print(np.diag(cxi))
    print(np.linalg.det(cxi))
    print(np.allclose(cxi, cxi.T))
    print(np.allclose(cx, cx.T))

    np.save('../data/crb/full_cxi_{:d}_{:d}'.format(n, Ndim), cxi)

def plot_crb_all(paper=False):
    """Plot 2D Cramer-Rao lower bounds for the MW halo parameters based on individual streams and their combination"""
    
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['font.size'] = 15
    
    params0 = ['$V_h$ (km/s)', '$R_h$ (kpc)', '$q_1$', '$q_z$', '$M_{LMC}$']
    params = ['$\Delta$ '+x for x in params0]
    Nvar = len(params)
    
    streams = [-1,-2,-3,-4]
    Nstr = len(streams)
    
    ylim = [20, 0.5, 0.5]
    columns = ['GD-1', 'Pal 5', 'Triangulum', 'ATLAS']
    labels = ['RA, Dec, d', 'RA, Dec, d,\n$V_r$', 'RA, Dec, d,\n$V_r$, $\mu_\\alpha$, $\mu_\\delta$']
    
    cxi_all = np.zeros((3,5,5))
    
    plt.close()
    fig, ax = plt.subplots(Nvar-2, Nstr+2, figsize=(10,7), sharex='col', sharey='row')
    
    for l, Ndim in enumerate([3,4,6]):
        # Individual streams
        for i in range(Nstr):
            cxi = np.load('../data/crb/full_cxi_{:d}_{:d}.npy'.format(streams[i], Ndim))
            cxi = cxi[:Nvar,:Nvar]
            cxi_all[l] += cxi
            cx = np.linalg.inv(cxi)
            
            for j in range(1,Nvar-1):
                plt.sca(ax[j-1][i])
                cx_2d = np.array([[cx[0][0], cx[0][j]], [cx[j][0], cx[j][j]]])
                
                w, v = np.linalg.eig(cx_2d)
                theta = np.degrees(np.arccos(v[0][0]))
                width = np.sqrt(w[0])*2
                height = np.sqrt(w[1])*2
                
                e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=mpl.cm.bone(l/4), lw=2)
                ax[j-1][i].add_artist(e)
                
                plt.xlim(-150,150)
                plt.ylim(-ylim[j-1], ylim[j-1])
                
                if j==1:
                    plt.title(columns[i], fontsize='medium')
                
                if j==Nvar-2:
                    plt.xlabel(params[0])
                if i==0:
                    plt.ylabel(params[j])
        
        # All streams combined
        cx_all = np.linalg.inv(cxi_all[l])
        for j in range(1, Nvar-1):
            plt.sca(ax[j-1][Nstr])
            cx_all_2d = np.array([[cx_all[0][0], cx_all[0][j]], [cx_all[j][0], cx_all[j][j]]])
            
            w, v = np.linalg.eig(cx_all_2d)
            theta = np.degrees(np.arccos(v[0][0]))
            width = np.sqrt(w[0])*2
            height = np.sqrt(w[1])*2
            
            e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=mpl.cm.bone(l/4), lw=2)
            ax[j-1][Nstr].add_artist(e)
            
            plt.xlim(-150,150)
            plt.ylim(-ylim[j-1], ylim[j-1])
            
            if j==1:
                plt.title('Combined', fontsize='medium')
            if j==Nvar-2:
                plt.xlabel(params[0])
            
            plt.sca(ax[j-1][Nstr+1])
            plt.axis('off')
        
        plt.sca(ax[Nvar-3][Nstr+1])
        plt.plot(np.linspace(0,1,10), '-', color=mpl.cm.bone(l/4), lw=2, label=labels[l])
        plt.xlim(-2,-1)
        plt.legend(frameon=False, fontsize='small', handlelength=1, loc=3)
            
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/crlb_2d.pdf', bbox_inches='tight')
    if paper:
        plt.savefig('../paper/crlb_2d.pdf', bbox_inches='tight')

    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['font.size'] = 18

def plot_crb_triangle(n=-1, istep=18, vary='potential', out='save'):
    """Produce a triangle plot of 2D Cramer-Rao bounds for all model parameters using a given stream"""
    
    if vary=='all':
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['font.size'] = 6
        mpl.rcParams['xtick.major.size'] = 2
        mpl.rcParams['ytick.major.size'] = 2
    
    pid, dp = get_varied_pars(vary)
    Nvar = len(pid)
    
    columns = ['GD-1', 'Pal 5', 'Triangulum', 'ATLAS']
    name = columns[int(np.abs(n)-1)]
    
    labels = ['RA, Dec, d', 'RA, Dec, d,\n$V_r$', 'RA, Dec, d,\n$V_r$, $\mu_\\alpha$, $\mu_\\delta$']
    params0 = ['$V_h$ (km/s)', '$R_h$ (kpc)', '$q_1$', '$q_z$', '$M_{LMC}$', '$X_p$', '$Y_p$', '$Z_p$', '$V_{xp}$', '$V_{yp}$', '$V_{zp}$']
    params = ['$\Delta$ '+x for x in params0]
    ylim = [150, 20, 0.5, 0.5, 5e11]
    ylim = [20, 5, 0.2, 0.2]
    #ylim = [10, 1, 0.1, 0.1]
    
    plt.close()
    fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(8,8), sharex='col', sharey='row')
    
    # plot 2d bounds in a triangle fashion
    for l, Ndim in enumerate([3,4,6]):
        cxi = np.load('../data/crb/bspline_cxi_align_{:d}_{:d}_{:d}.npy'.format(n, Ndim, istep))
        #cxi = np.load('../data/crb/full_cxi_{:d}_{:d}.npy'.format(n, Ndim))
        cxi = cxi[:Nvar,:Nvar]
        cx = np.linalg.inv(cxi)
        
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar):
                plt.sca(ax[j-1][i])
                cx_2d = np.array([[cx[i][i], cx[i][j]], [cx[j][i], cx[j][j]]])
                
                w, v = np.linalg.eig(cx_2d)
                if np.all(np.isreal(v)):
                    theta = np.degrees(np.arccos(v[0][0]))
                    width = np.sqrt(w[0])*2
                    height = np.sqrt(w[1])*2
                    
                    e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=mpl.cm.bone(l/4), lw=2)
                    plt.gca().add_artist(e)
                
                if (vary=='potential') | (vary=='halo'):
                    plt.xlim(-ylim[i],ylim[i])
                    plt.ylim(-ylim[j], ylim[j])
                #else:
                    #if l==1:
                        #plt.xlim(-0.5*width, 0.5*width)
                        #plt.ylim(-0.5*height, 0.5*height)
                
                if j==Nvar-1:
                    plt.xlabel(params[i])
                
                if i==0:
                    plt.ylabel(params[j])
        
        plt.sca(ax[0][Nvar-2])
        plt.plot(np.linspace(-200,-100,10), '-', color=mpl.cm.bone(l/4), lw=2, label=labels[l])
        #plt.xlim(-2,-1)
        plt.legend(frameon=False, fontsize=14, handlelength=1)
    
    # turn off unused axes
    for i in range(0,Nvar-1):
        for j in range(i+1,Nvar-1):
            plt.sca(ax[i][j])
            plt.axis('off')
    
    plt.suptitle('{} stream'.format(name), fontsize='large')
    plt.tight_layout(h_pad=0.0, w_pad=0.0, rect=[0,0,1,0.97])
    
    if vary=='all':
        mpl.rcParams['axes.linewidth'] = 2
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['xtick.major.size'] = 10
        mpl.rcParams['ytick.major.size'] = 10
    
    if out=='save':
        #plt.savefig('../plots/crb_individual_{}_{}.png'.format(n, vary))
        plt.savefig('../plots/crb_bspline_individual_{}_{}_align_{:02d}.png'.format(n, vary, istep))
    else:
        return fig

def collate_individual_crb(vary='potential'):
    """"""
    
    # open pdf
    pp = PdfPages("../plots/crb_allstreams_{}.pdf".format(vary))
    
    for n in [-1,-2,-3,-4]:
        fig = plot_crb_triangle(n=n, vary=vary, out='return')
        pp.savefig(fig)
    
    pp.close()

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


