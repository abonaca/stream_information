from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

import streakline
import streakline2
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

import copy

# observers
# defaults taken as in astropy v2.0 icrs
mw_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_coord': coord.SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg, frame='icrs')}
vsun = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}
vsun0 = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

gc_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 0.1*u.kpc, 'roll': 0*u.deg, 'galcen_coord': coord.SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg, frame='icrs')}
vgc = {'vcirc': 0*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}
vgc0 = {'vcirc': 0*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

MASK = -9999
pparams_fid = [0.5e10*u.Msun, 0.7*u.kpc, 6.8e10*u.Msun, 3*u.kpc, 0.28*u.kpc, 430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.deg, 0*u.deg, 0*u.kpc, 0*u.km/u.s, 0*u.mas/u.yr, 0*u.mas/u.yr]


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


# make a streakline model of a stream

def stream_model(n, pparams0=[0.5e10*u.Msun, 0.7*u.kpc, 6.8e10*u.Msun, 3*u.kpc, 0.28*u.kpc, 430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.deg, 0*u.deg, 0*u.kpc, 0*u.km/u.s, 0*u.mas/u.yr, 0*u.mas/u.yr], dt=1*u.Myr, rotmatrix=None, graph=False, observer=mw_observer, vobs=vsun, footprint='', obsmode='equatorial'):
    """Create a streakline model of a stream
    baryonic component as in kupper+2015: 3.4e10*u.Msun, 0.7*u.kpc, 1e11*u.Msun, 6.5*u.kpc, 0.26*u.kpc"""
    
    # vary potential parameters
    potential = 'gal'
    pparams = pparams0[:11]
    
    # adjust circular velocity in this halo
    r = observer['galcen_distance']
    # nfw halo
    mr = pparams[5]**2 * pparams[6] / G * (np.log(1 + r/pparams[6]) - r/(r + pparams[6]))
    vch2 = G*mr/r
    # hernquist bulge
    vcb2 = G * pparams[0] * r * (r + pparams[1])**-2
    # miyamoto-nagai disk
    vcd2 = G * pparams[2] * r**2 * (r**2 + (pparams[3] + pparams[4])**2)**-1.5
    
    vobs['vcirc'] = np.sqrt(vch2 + vcb2 + vcd2)
    
    # vary progenitor parameters
    progenitor = progenitor_params(n)
    x0_obs, v0_obs = gal2eq(progenitor['x0'], progenitor['v0'], observer=observer, vobs=vobs)
    for i in range(3):
        x0_obs[i] += pparams0[12+i]
        v0_obs[i] += pparams0[15+i]
    
    # stream model parameters
    params = {'generate': {'x0': x0_obs, 'v0': v0_obs, 'progenitor': {'coords': 'equatorial', 'observer': observer, 'pm_polar': False}, 'potential': potential, 'pparams': pparams, 'minit': progenitor['mi'], 'mfinal': progenitor['mf'], 'rcl': 20*u.pc, 'dr': 0., 'dv': 0*u.km/u.s, 'dt': dt, 'age': progenitor['age'], 'nstars': 400, 'integrator': 'lf'}, 'observe': {'mode': obsmode, 'nstars':-1, 'sequential':True, 'errors': [2e-4*u.deg, 2e-4*u.deg, 0.5*u.kpc, 5*u.km/u.s, 0.5*u.mas/u.yr, 0.5*u.mas/u.yr], 'present': [0,1,2,3,4,5], 'observer': observer, 'vobs': vobs, 'footprint': footprint, 'rotmatrix': rotmatrix}}
    
    stream = Stream(**params['generate'])
    stream.generate()
    stream.observe(**params['observe'])
    
    ################################
    # Plot observed stream and model
    
    if graph:
        observed = load_stream(n)
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
            
            if i<Ndim-1:
                if i<2:
                    sample = np.arange(np.size(observed.obs[0]), dtype=int)
                else:
                    sample = observed.obs[i+1]>MASK
                plt.plot(observed.obs[0][sample], observed.obs[i+1][sample], 's', color=obscol, mec='none', ms=8, label='Observed stream')
            
            plt.plot(stream.obs[0], stream.obs[i+1], 'o', color=modcol, mec='none', ms=4, label='Fiducial model')
            
            if i==0:
                plt.legend(frameon=False, handlelength=0.5, fontsize='small')
        
        plt.tight_layout()
    
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

def find_greatcircle(n, pparams=pparams_fid, dt=0.2*u.Myr):
    """Save rotation matrix for a stream model"""
    
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


# observed streams

def load_stream(n):
    """Load stream observations"""
    
    if n==-1:
        observed = load_gd1(present=[0,1,2,3])
    elif n==-2:
        observed = load_pal5(present=[0,1,2,3])
    elif n==-3:
        observed = load_tri(present=[0,1,2,3])
    elif n==-4:
        observed = load_atlas(present=[0,1,2,3])
    
    return observed

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
        dp = [0.4e7*u.Msun, 0.005*u.kpc, 0.9e9*u.Msun, 0.002*u.kpc, 0.002*u.kpc]
        ## atlas & triangulum
        #dp = [0.4e5*u.Msun, 0.0005*u.kpc, 0.5e6*u.Msun, 0.0002*u.kpc, 0.002*u.kpc]
        ## pal5
        #dp = [0.4e3*u.Msun, 0.000005*u.kpc, 0.5e4*u.Msun, 0.000002*u.kpc, 0.00002*u.kpc]
    elif vary=='halo':
        pid = [5,6,8,10]
        dp = [20*u.km/u.s, 2*u.kpc, 0.05*u.Unit(1), 0.05*u.Unit(1)]
        dp = [30*u.km/u.s, 2.9*u.kpc, 0.05*u.Unit(1), 0.05*u.Unit(1)]
    elif vary=='progenitor':
        pid = [12,13,14,15,16,17]
        dp = [1*u.deg, 1*u.deg, 0.5*u.kpc, 20*u.km/u.s, 0.3*u.mas/u.yr, 0.3*u.mas/u.yr]
    else:
        pid = []
        dp = []
    
    return (pid, dp)

def get_parlabel(pid):
    """Return label for a list of parameter ids
    Parameter:
    pid - list of parameter ids"""
    
    master = ['$M_b$', '$a_b$', '$M_d$', '$a_d$', '$b_d$', '$V_h$', '$R_h$', '$\phi$', '$q_1$', '$q_2$', '$q_z$', '$M_{lmc}$', '$RA_p$', '$Dec_p$', '$d_p$', '$V_{r_p}$', '$\mu_{\\alpha_p}$', '$\mu_{\delta_p}$']
    master_units = ['$M_\odot$', 'kpc', '$M_\odot$', 'kpc', 'kpc', 'km/s', 'kpc', 'rad', '', '', '', '$M_\odot$', 'deg', 'deg', 'kpc', 'km/s', 'mas/yr', 'mas/yr']
    
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
        step = np.logspace(-2, 1, Nstep)
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

def step_convergence(n, Nstep=20, log=True, layer=1, dt=0.2*u.Myr, vary='halo', align=True, graph=False):
    """Check deviations in numerical derivatives for consecutive step sizes"""
    
    if align:
        rotmatrix = np.load('../data/rotmatrix_{}.npy'.format(n))
    else:
        rotmatrix = None
    
    pparams0 = pparams_fid
    pid, dp, vlabel = get_varied_pars(vary)
    Np = len(pid)
    units = ['km/s', 'kpc', '', '']
    units = ['kpc', 'kpc', 'kpc', 'km/s', 'km/s', 'km/s']
    punits = ['({})'.format(x) if len(x) else '' for x in units]

    Nstep, step = get_steps(Nstep=Nstep, log=log)

    dev_der = np.empty((Np, Nstep-2*layer))
    step_der = np.empty((Np, Nstep-2*layer))
    
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
    
    for p in range(Np):
        plabel = get_parlabel(pid[p])
        
        # excursions
        stream_fits = [[None] * 5 for x in range(2 * Nstep)]
        
        for i, s in enumerate(step[:]):
            pparams = [x for x in pparams0]
            pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
            stream = stream_model(n, pparams0=pparams, dt=dt, rotmatrix=rotmatrix)
            
            # fits
            iexsort = np.argsort(stream.obs[0])
            raex = np.linspace(np.percentile(stream.obs[0], 10), np.percentile(stream.obs[0], 90), Nobs)
            tex = np.r_[(stream.obs[0][iexsort][0],)*(k+1), raex, (stream.obs[0][iexsort][-1],)*(k+1)]
            fits_ex = [None]*5
            
            for j in range(5):
                fits_ex[j] = scipy.interpolate.make_lsq_spline(stream.obs[0][iexsort], stream.obs[j+1][iexsort], tex, k=k)
                stream_fits[i][j] = fits_ex[j]
        
        # symmetric derivatives
        ra_der = np.linspace(np.min(stream0.obs[0])*1.05, np.max(stream0.obs[0])*0.95, 100)
        dydx = np.empty((Nstep, 5, 100))
        
        for i in range(Nstep):
            color = mpl.cm.Greys_r(i/Nstep)
            for j in range(5):
                dy = stream_fits[i][j](ra_der) - stream_fits[-i-1][j](ra_der)
                dydx[i][j] = -dy / np.abs(2*step[i]*dp[p])
        
        # deviations from adjacent steps
        step_der[p] = -step[layer:Nstep-layer] * dp[p]
        
        for i in range(layer, Nstep-layer):
            dev_der[p][i-layer] = 0
            for j in range(5):
                for l in range(layer):
                    dev_der[p][i-layer] += np.sum((dydx[i][j] - dydx[i-l-1][j])**2)
                    dev_der[p][i-layer] += np.sum((dydx[i][j] - dydx[i+l+1][j])**2)
    
    np.savez('../data/step_convergence_{}_{}_Ns{}_log{}_l{}'.format(n, vlabel, Nstep, log, layer), step=step_der, dev=dev_der)
    
    if graph:
        plt.close()
        fig, ax = plt.subplots(1,Np,figsize=(4*Np,4))
        
        for p in range(Np):
            plt.sca(ax[p])
            plt.plot(step_der[p], dev_der[p], 'ko')
            
            plt.xlabel('$\Delta$ {} {}'.format(plabel, punits[p]))
            plt.ylabel('D')
            plt.gca().set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('../plots/step_convergence_{}_{}_Ns{}_log{}_l{}.png'.format(n, vlabel, Nstep, log, layer))

def choose_step(n, tolerance=2, Nstep=20, log=True, layer=1, vary='halo'):
    """"""
    
    pid, dp, vlabel = get_varied_pars(vary)
    Np = len(pid)
    plabels, units = get_parlabel(pid)
    punits = ['({})'.format(x) if len(x) else '' for x in units]
    
    t = np.load('../data/step_convergence_{}_{}_Ns{}_log{}_l{}.npz'.format(n, vlabel, Nstep, log, layer))
    dev = t['dev']
    step = t['step']
    
    best = np.empty(Np)
    
    # plot setup
    da = 4
    if Np>6:
        nrow = 2
        ncol = np.int64(np.ceil(Np/2))
    else:
        nrow = 1
        ncol = Np
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(da*ncol, da*nrow), squeeze=False)
    
    for p in range(Np):
        plt.sca(ax[int(p/ncol)][int(p%ncol)])
        plt.plot(step[p], dev[p], 'ko')
        
        # choose step
        dmin = np.min(dev[p])
        dtol = tolerance * dmin
        opt_step = np.min(step[p][dev[p]<dtol])
        opt_id = step[p]==opt_step
        best[p] = opt_step
        
        plt.axvline(opt_step, ls='-', color='r', lw=2)
        plt.plot(step[p][opt_id], dev[p][opt_id], 'ro')
        
        plt.axhline(dtol, ls='-', color='orange', lw=1)
        y0, y1 = plt.gca().get_ylim()
        plt.axhspan(y0, dtol, color='orange', alpha=0.3, zorder=0)
        
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.xlabel('$\Delta$ {} {}'.format(plabels[p], punits[p]))
        plt.ylabel('Derivative deviation')
        plt.title('{}'.format(plabels[p])+'$_{best}$ = '+'{:2.2g}'.format(opt_step), fontsize='small')
    
    np.save('../data/optimal_step_{}_{}'.format(n, vlabel), best)

    plt.tight_layout()
    plt.savefig('../plots/step_convergence_{}_{}_Ns{}_log{}_l{}.png'.format(n, vlabel, Nstep, log, layer))

def read_optimal_step(n, vary):
    """Return optimal steps for a range of parameter types"""
    
    if type(vary) is not list:
        vary = [vary]
    
    Nt = len(vary)
    dp = np.empty(0)
    
    for v in vary:
        dp_opt = np.load('../data/optimal_step_{}_{}.npy'.format(n, v))
        dp = np.concatenate([dp, dp_opt])
    
    return dp


# crbs using bspline

def bspline_crb(n, dt=0.2*u.Myr, vary='halo', Nobs=50, verbose=False, align=True):
    """"""
    if align:
        rotmatrix = np.load('../data/rotmatrix_{}.npy'.format(n))
        alabel = '_align'
    else:
        rotmatrix = None
        alabel = ''
        
    # typical uncertainties
    sig_obs = np.array([0.1, 2, 5, 0.1, 0.1])
    
    # mock observations
    pparams0 = pparams_fid
    stream0 = stream_model(n, pparams0=pparams0, dt=dt, rotmatrix=rotmatrix)
    
    ra = np.linspace(np.min(stream0.obs[0])*1.05, np.max(stream0.obs[0])*0.95, Nobs)
    err = np.tile(sig_obs, Nobs).reshape(Nobs,-1)

    pid, dp_fid, vlabel = get_varied_pars(vary)
    Np = len(pid)
    #dp_opt = np.load('../data/optimal_step_{}_{}.npy'.format(n, vlabel))
    dp_opt = read_optimal_step(n, vary)
    dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
    
    k = 3
    
    fits_ex = [[[None]*5 for x in range(2)] for y in range(Np)]

    for p in range(Np):
        for i, s in enumerate([-1, 1]):
            pparams = [x for x in pparams0]
            pparams[pid[p]] = pparams[pid[p]] + s*dp[p]
            stream = stream_model(n, pparams0=pparams, dt=dt, rotmatrix=rotmatrix)
            
            # fits
            iexsort = np.argsort(stream.obs[0])
            raex = np.linspace(np.percentile(stream.obs[0], 10), np.percentile(stream.obs[0], 90), Nobs)
            tex = np.r_[(stream.obs[0][iexsort][0],)*(k+1), raex, (stream.obs[0][iexsort][-1],)*(k+1)]
            
            for j in range(5):
                fits_ex[p][i][j] = scipy.interpolate.make_lsq_spline(stream.obs[0][iexsort], stream.obs[j+1][iexsort], tex, k=k)
    
    for Ndim in [3,4,6]:
        Ndata = Nobs * (Ndim - 1)
        cyd = np.empty(Ndata)
        dydx = np.empty((Np, Ndata))
        
        for j in range(1, Ndim):
            for p in range(Np):
                dy = fits_ex[p][0][j-1](ra) - fits_ex[p][1][j-1](ra)
                dydx[p][(j-1)*Nobs:j*Nobs] = -dy / np.abs(2*dp[p].value)
        
            cyd[(j-1)*Nobs:j*Nobs] = err[:,j-1]**2
        
        cy = np.diag(cyd)
        cyi = np.diag(1. / cyd)
        
        cxi = np.matmul(dydx, np.matmul(cyi, dydx.T))

        cx = np.linalg.inv(cxi)
        cx = np.matmul(np.linalg.inv(np.matmul(cx, cxi)), cx) # iteration of inverse improvement for large cond numbers
        sx = np.sqrt(np.diag(cx))
        
        if verbose:
            print(Ndim)
            print(np.diag(cxi))
            print(np.linalg.det(cxi))
            print(np.allclose(cxi, cxi.T), np.allclose(cx, cx.T), np.allclose(np.matmul(cx,cxi), np.eye(np.shape(cx)[0])))
            print('condition {:g}'.format(np.linalg.cond(cxi)))

        np.save('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}'.format(alabel, n, vlabel, Ndim), cxi)

def test_inversion(n, Ndim=6, vary=['halo', 'progenitor'], align=True):
    """"""
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
        
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
    
    cx = np.linalg.inv(cxi)
    cxi_ = np.linalg.inv(cx)
    print(np.linalg.cond(cxi))
    print(np.linalg.cond(cx))
    print(np.linalg.det(cxi))
    print(np.linalg.det(cx))
    print(np.linalg.norm(cxi)*np.linalg.norm(cx), np.linalg.norm(cxi), np.linalg.norm(cxi_))
    #print(np.matmul(cx,cxi))
    #print(cxi)
    plt.close()
    plt.figure()
    
    plt.hist(np.ravel(cxi), bins=np.logspace(-3,5,10))
    plt.gca().set_xscale('log')

def crb_triangle(n, vary, Ndim=6, align=True, plot='all'):
    """"""
    
    pid, dp, vlabel = get_varied_pars(vary)
    plabels, units = get_parlabel(pid)
    params = ['$\Delta$' + x + '({})'.format(y) for x,y in zip(plabels, units)]
    
    if align:
        alabel = '_align'
    else:
        alabel = ''
        
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
    cx = np.linalg.inv(cxi)
    print(cx[0][0])
    
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

def crb_triangle_alldim(n, vary, align=True, plot='all'):
    """"""
    
    pid, dp, vlabel = get_varied_pars(vary)
    plabels, units = get_parlabel(pid)
    params = ['$\Delta$' + x + '({})'.format(y) for x,y in zip(plabels, units)]
    
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    if plot=='halo':
        i0 = 0
        i1 = 4
    elif plot=='bary':
        i0 = 4
        i1 = 9
    elif plot=='progenitor':
        i0 = 9
        i1 = len(params)
    else:
        i0 = 0
        i1 = len(params)
    
    Nvar = i1 - i0
    params = params[i0:i1]
    label = ['RA, Dec, d', 'RA, Dec, d, $V_r$', 'RA, Dec, d, $V_r$, $\mu_\\alpha$, $\mu_\delta$']
    
    plt.close()
    dax = 2
    fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(dax*Nvar, dax*Nvar), sharex='col', sharey='row')
    
    for l, Ndim in enumerate([3, 4, 6]):
        cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
        cx = np.linalg.inv(cxi)
        cx = cx[i0:i1,i0:i1]
        
        for i in range(0,Nvar-1):
            for j in range(i+1,Nvar):
                plt.sca(ax[j-1][i])
                cx_2d = np.array([[cx[i][i], cx[i][j]], [cx[j][i], cx[j][j]]])
                
                w, v = np.linalg.eig(cx_2d)
                if np.all(np.isreal(v)):
                    theta = np.degrees(np.arccos(v[0][0]))
                    width = np.sqrt(w[0])*2
                    height = np.sqrt(w[1])*2
                    
                    e = mpl.patches.Ellipse((0,0), width=width, height=height, angle=theta, fc='none', ec=mpl.cm.bone(0.1+l/4), lw=2, label=label[l])
                    plt.gca().add_patch(e)
                
                if l==2:
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
    plt.savefig('../plots/crb_triangle_alldim_{:s}_{:d}_{:s}_{:s}.pdf'.format(alabel, n, vlabel, plot))


# accelerations

def halo_accelerations(x, pu=[pparams_fid[j] for j in [5,6,8,10]]):
    """Calculate derivatives of halo potential parameters wrt (Cartesian) components of the acceleration vector a"""
    
    p = np.array([j.value for j in pu])
    q = np.array([1, p[2], p[3]])
    
    # physical quantities
    x = x.value
    r = np.linalg.norm(x)
    a = p[0]**2 * p[1] * r**-3 * (1/(1+p[1]/r) - np.log(1+r/p[1])) * x * q**-2
    
    #  derivatives
    dmat = np.zeros((3, 4))
    
    # Vh
    dmat[:,0] = ( 2*a/p[0] )**-1
    
    # Rh
    dmat[:,1] = ( a/p[1] + p[0]**2 * p[1] * r**-3 * (1/(p[1]+p[1]**2/r) - 1/(r*(1+p[1]/r)**2)) * x * q**-2 )**-1
    
    # qy, qz
    for i in [1,2]:
        dmat[i,i+1] = ( -2*a[i]/q[i] )**-1
    
    return dmat

def bary_accelerations(x, pu=[pparams_fid[j] for j in range(5)]):
    """Calculate derivarives of bulge + disk potential parameters wrt (Cartesian) components of the acceleration vector a"""
    
    p = np.array([j.value for j in pu])
    
    # physical quantities
    #x = x.value
    r = np.linalg.norm(x)*u.kpc
    R = np.linalg.norm(x[:2])*u.kpc
    z = x[2]
    aux = np.sqrt(z**2 + pu[4]**2)
    
    ab = acc_bulge(x, p=pu[:2])
    ad = acc_disk(x, p=pu[2:])
    
    #  derivatives
    dmat = np.zeros((3, 5))
    
    # Mb
    dmat[:,0] = ab/pu[0]
    
    # ab
    dmat[:,1] = 2 * ab / (r + pu[1])
    
    # Md
    dmat[:,2] = ad / pu[2]
    
    # ad
    dmat[:,3] = 3 * ad * (pu[3] + aux) / (R**2 + (pu[3] + aux)**2)
    
    # bd
    dmat[:2,4] = 3 * ad[:2] * (pu[3] + aux) / (R**2 + (pu[3] + aux)**2) * pu[4] / aux
    dmat[2,4] = (3 * ad[2] * (pu[3] + aux) / (R**2 + (pu[3] + aux)**2) * pu[4] / aux - G * pu[2] * z * (R**2 + (pu[3] + aux)**2)**-1.5 * z**2 * (pu[4]**2 + z**2)**-1.5).value
    
    return dmat

def acc_bulge(x, p=[pparams_fid[j] for j in range(2)]):
    """"""
    r = np.linalg.norm(x)*u.kpc
    a = (-G*p[0]*x/(r * (r + p[1])**2)).to(u.pc*u.Myr**-2)
    
    return a

def acc_disk(x, p=[pparams_fid[j] for j in range(2,5)]):
    """"""
    R = np.linalg.norm(x[:2])*u.kpc
    z = x[2]
    a = (G*p[0]*x * (R**2 + (p[1] + np.sqrt(z**2 + p[2]**2))**2)**-1.5).to(u.pc*u.Myr**-2)
    a[2] *= 1 + p[2]/np.sqrt(z**2 + p[2]**2)
    
    return a

def acc_nfw(x, p=[pparams_fid[j] for j in [5,6,8,10]]):
    """"""
    r = np.linalg.norm(x)*u.kpc
    q = np.array([1*u.Unit(1), p[2], p[3]])
    a = (p[0]**2 * p[1] * r**-3 * (1/(1+p[1]/r) - np.log(1+r/p[1])) * x * q**-2).to(u.pc*u.Myr**-2)
    
    return a

def crb_ax(n, Ndim=6, vary=['halo', 'bary', 'progenitor'], align=True):
    """Calculate CRB inverse matrix for 3D acceleration at position x in a halo potential"""
    
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    # read in full inverse CRB for stream modeling
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
    cx = np.linalg.inv(cxi)
    
    # subset halo parameters
    Nhalo = 4
    cq = cx[:Nhalo,:Nhalo]
    cqi = np.linalg.inv(cq)
    
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
            ca = np.linalg.inv(cai)
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
    


