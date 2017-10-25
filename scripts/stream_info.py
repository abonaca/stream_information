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
pparams_fid = [0.5*u.Msun, 0.7*u.kpc, 6.8*u.Msun, 3*u.kpc, 0.28*u.kpc, 430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 0.*u.pc/u.Myr**2, 0.*u.pc/u.Myr**2, 0.*u.pc/u.Myr**2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0.*u.Gyr**-2, 0*u.deg, 0*u.deg, 0*u.kpc, 0*u.km/u.s, 0*u.mas/u.yr, 0*u.mas/u.yr]


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


# make a streakline model of a stream

def stream_model(n, pparams0=pparams_fid, dt=0.2*u.Myr, rotmatrix=None, graph=False, graphsave=False, observer=mw_observer, vobs=vsun, footprint='', obsmode='equatorial'):
    """Create a streakline model of a stream
    baryonic component as in kupper+2015: 3.4e10*u.Msun, 0.7*u.kpc, 1e11*u.Msun, 6.5*u.kpc, 0.26*u.kpc"""
    
    # vary potential parameters
    potential = 'quad'
    pparams = pparams0[:19]
    pparams[0] = pparams0[0]*1e10
    pparams[2] = pparams0[2]*1e10
    
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
        x0_obs[i] += pparams0[19+i]
        v0_obs[i] += pparams0[22+i]
    
    #potential = 'point'
    #pparams[0] = 5e9*u.Msun
    
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
        if graphsave:
            plt.savefig('../plots/mock_observables_s{}_p{}.png'.format(n, potential), dpi=150)
    
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
        dp = [1e-1*u.Msun, 0.005*u.kpc, 1e-1*u.Msun, 0.002*u.kpc, 0.002*u.kpc]
        ## atlas & triangulum
        #dp = [0.4e5*u.Msun, 0.0005*u.kpc, 0.5e6*u.Msun, 0.0002*u.kpc, 0.002*u.kpc]
        # pal5
        dp = [1e-2*u.Msun, 0.000005*u.kpc, 1e-2*u.Msun, 0.000002*u.kpc, 0.00002*u.kpc]
        dp = [1e-7*u.Msun, 0.5*u.kpc, 1e-7*u.Msun, 0.5*u.kpc, 0.5*u.kpc]
    elif vary=='halo':
        pid = [5,6,8,10]
        dp = [20*u.km/u.s, 2*u.kpc, 0.05*u.Unit(1), 0.05*u.Unit(1)]
        dp = [30*u.km/u.s, 2.5*u.kpc, 0.05*u.Unit(1), 0.05*u.Unit(1)]
    elif vary=='progenitor':
        pid = [19,20,21,22,23,24]
        dp = [1*u.deg, 1*u.deg, 0.5*u.kpc, 20*u.km/u.s, 0.3*u.mas/u.yr, 0.3*u.mas/u.yr]
    elif vary=='dipole':
        pid = [11,12,13]
        #dp = [1e-11*u.Unit(1), 1e-11*u.Unit(1), 1e-11*u.Unit(1)]
        dp = [0.3*u.pc/u.Myr**2, 0.3*u.pc/u.Myr**2, 0.3*u.pc/u.Myr**2]
    elif vary=='quad':
        pid = [14,15,16,17,18]
        dp = [0.5*u.Gyr**-2 for x in range(5)]
    else:
        pid = []
        dp = []
    
    return (pid, dp)

def get_parlabel(pid):
    """Return label for a list of parameter ids
    Parameter:
    pid - list of parameter ids"""
    
    master = ['$M_b$', '$a_b$', '$M_d$', '$a_d$', '$b_d$', '$V_h$', '$R_h$', '$\phi$', '$q_1$', '$q_2$', '$q_z$', '$a_{1,-1}$', '$a_{1,0}$', '$a_{1,1}$', '$a_{2,-2}$', '$a_{2,-1}$', '$a_{2,0}$', '$a_{2,1}$', '$a_{2,2}$', '$RA_p$', '$Dec_p$', '$d_p$', '$V_{r_p}$', '$\mu_{\\alpha_p}$', '$\mu_{\delta_p}$']
    master_units = ['$10^{10}$ $M_\odot$', 'kpc', '$10^{10}$ $M_\odot$', 'kpc', 'kpc', 'km/s', 'kpc', 'rad', '', '', '', 'pc/Myr$^2$', 'pc/Myr$^2$', 'pc/Myr$^2$', 'Gyr$^{-2}$', 'Gyr$^{-2}$', 'Gyr$^{-2}$', 'Gyr$^{-2}$', 'Gyr$^{-2}$', 'deg', 'deg', 'kpc', 'km/s', 'mas/yr', 'mas/yr']
    
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

def step_convergence(n, Nstep=20, log=True, layer=1, dt=0.2*u.Myr, vary='halo', align=True, graph=False, verbose=False):
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
        if verbose: print(p, plabel)
        
        # excursions
        stream_fits = [[None] * 5 for x in range(2 * Nstep)]
        
        for i, s in enumerate(step[:]):
            if verbose: print(i, s)
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

def bspline_crb(n, dt=0.2*u.Myr, vary='halo', Nobs=50, verbose=False, align=True, scale=True):
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
    dp_opt = read_optimal_step(n, vary)
    dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
    
    if scale:
        dp_unit = unity_scale(dp)
        dps = [x*y for x,y in zip(dp, dp_unit)]
    
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
    #for Ndim in [3,]:
        Ndata = Nobs * (Ndim - 1)
        cyd = np.empty(Ndata)
        dydx = np.empty((Np, Ndata))
        
        for j in range(1, Ndim):
            for p in range(Np):
                dy = fits_ex[p][0][j-1](ra) - fits_ex[p][1][j-1](ra)
                positive = np.abs(dy)>0
                print('{:d},{:d} {:s} min{:.1e} max{:1e} med{:.1e}'.format(j, p, get_parlabel(pid[p])[0], np.min(np.abs(dy[positive])), np.max(np.abs(dy)), np.median(np.abs(dy))))
                if scale:
                    dydx[p][(j-1)*Nobs:j*Nobs] = -dy / np.abs(2*dps[p].value)
                else:
                    dydx[p][(j-1)*Nobs:j*Nobs] = -dy / np.abs(2*dp[p].value)
                print('{:d},{:d} {:s} min{:.1e} max{:1e} med{:.1e}'.format(j, p, get_parlabel(pid[p])[0], np.min(np.abs(dydx[p][(j-1)*Nobs:j*Nobs][positive])), np.max(np.abs(dydx[p][(j-1)*Nobs:j*Nobs])), np.median(np.abs(dydx[p][(j-1)*Nobs:j*Nobs]))))
        
            cyd[(j-1)*Nobs:j*Nobs] = err[:,j-1]**2
        
        cy = np.diag(cyd)
        cyi = np.diag(1. / cyd)
        
        caux = np.matmul(cyi, dydx.T)
        cxi = np.matmul(dydx, caux)

        cx = np.linalg.inv(cxi)
        cx = np.matmul(np.linalg.inv(np.matmul(cx, cxi)), cx) # iteration of inverse improvement for large cond numbers
        sx = np.sqrt(np.diag(cx))
        
        for i, m in enumerate([cy, cyi, dydx, caux, cxi]):
            positive = m>0
            print('{:d} {:g} {:g}'.format(i, np.min(m[positive]), np.max(m)))
        
        if verbose:
            print(Ndim)
            print(np.diag(cxi))
            print(np.linalg.det(cxi))
            print(np.allclose(cxi, cxi.T), np.allclose(cx, cx.T), np.allclose(np.matmul(cx,cxi), np.eye(np.shape(cx)[0])))
            print('condition {:g}'.format(np.linalg.cond(cxi)))

        np.save('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}'.format(alabel, n, vlabel, Ndim), cxi)

def unity_scale(dp):
    """"""
    dim_scale = 10**np.array([2, 3, 3, 2, 4, 3, 7, 7, 5, 7, 7, 4, 4, 4, 4, 3, 3, 3, 4, 3, 4, 4, 4])
    #dim_scale = 10**np.array([2, 3, 3, 1, 3, 2, 5, 5, 3, 5, 5, 2, 2, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3])
    #dim_scale = 10**np.array([2, 3, 3, 1, 3, 2, 5, 5, 3, 5, 5, 2, 2, 4, 4, 3, 3, 3])
    dp_unit = [(dim_scale[x]*dp[x].value)**-1 for x in range(len(dp))]
    
    return dp_unit

def test_inversion(n, Ndim=6, vary=['halo', 'progenitor'], align=True):
    """"""
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
        
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
    N = np.shape(cxi)[0]
    
    cx_ = np.linalg.inv(cxi)
    
    cx = stable_inverse(cxi, verbose=True)
    cx_ii = stable_inverse(cx, verbose=True)
    
    print('condition {:g}'.format(np.linalg.cond(cxi)))
    print('stable inverse', np.allclose(np.matmul(cx,cxi), np.eye(N)))
    print('linalg inverse', np.allclose(np.matmul(cx_,cxi), np.eye(N)))
    print('inverse inverse', np.allclose(cx_ii, cxi))

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
        
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
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

def crb_triangle_alldim(n, vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], align=True, plot='all', fast=False, scale=True):
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
    label = ['RA, Dec, d', 'RA, Dec, d, $V_r$', 'RA, Dec, d, $V_r$, $\mu_\\alpha$, $\mu_\delta$']

    plt.close()
    dax = 2
    fig, ax = plt.subplots(Nvar-1, Nvar-1, figsize=(dax*Nvar, dax*Nvar), sharex='col', sharey='row')
    
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
    plt.savefig('../plots/crb_triangle_alldim_{:s}_{:d}_{:s}_{:s}.pdf'.format(alabel, n, vlabel, plot))

###
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
    
    dmat[0] = np.array([x[1], 0, -s*x[0], x[2], x[0]])
    dmat[1] = np.array([x[0], x[2], -s*x[1], 0, -x[1]])
    dmat[2] = np.array([0, x[1], 2*s*x[2], x[0], 0])
    
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
    dict_acc = {'bary': [acc_bulge, acc_disk], 'halo': [acc_nfw], 'dipole': [acc_dipole], 'quad': [acc_quad], 'point': [acc_kepler]}
    accelerations = []
    
    for c in components:
        accelerations += dict_acc[c]
    
    for acc in accelerations:
        a_ = acc(x)
        acart += a_
    
    return acart

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
    dict_der = {'bary': [pder_bulge, pder_disk], 'halo': [pder_nfw], 'dipole': [pder_dipole], 'quad': [pder_quad], 'point': [pder_kepler]}
    derivatives = []
    
    for c in components:
        derivatives += dict_der[c]
    
    for ader in derivatives:
        da_ = ader(x)
        dacart = np.hstack((dacart, da_))
    
    return dacart

def crb_acart(n, Ndim=6, vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], component='all', align=True, d=20, Nb=50, fast=False, scale=True, relative=True, progenitor=False):
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
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
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
    np.savez('../data/crb_acart{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}'.format(alabel, n, vlabel, component, Ndim, d, Nb, relative), acc=af, x=xin, der=derf)
    
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
    plt.savefig('../plots/crb_acc_cart{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}.png'.format(alabel, n, vlabel, component, Ndim, d, Nb, relative))

def crb_acart_cov(n, Ndim=6, vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], component='all', j=0, align=True, d=20, Nb=30, fast=False, scale=True, relative=True, progenitor=False, batch=False):
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
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
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
    np.savez('../data/crb_acart_cov{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(alabel, n, vlabel, component, Ndim, d, Nb, relative, progenitor), x=xin, der=derf, c=ca)
    
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
        plt.savefig('../plots/crb_acc_cart_cov{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.png'.format(alabel, n, vlabel, component, np.abs(j), Ndim, d, Nb, relative, progenitor))


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

def a_crbcov_vecfield(n, Ndim=6, vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], component='all', j=0, align=True, d=20, Nb=10, fast=False, scale=True, relative=False, progenitor=False, batch=False):
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
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
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
        plt.savefig('../plots/afield_crbcov{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}_{:d}.png'.format(alabel, n, vlabel, component, np.abs(j), Ndim, d, Nb, relative))


def summary(n, mode='scalar', vary=['progenitor', 'bary', 'halo', 'dipole', 'quad'], component='all'):
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
    
    pp = PdfPages('../plots/acceleration_{}_{}_{}_{}.pdf'.format(n, vlabel, component, mode))
    
    for i in range(Niter):
        print(i, Niter)
        fig = fn[mode](-1, progenitor=True, batch=True, vary=vary, component=component, j=-i, d=20, Nb=bins[mode])
        pp.savefig(fig)
    
    pp.close()

# cylindrical coordinates

def acc_cyl(x, p=pparams_fid, components=['bary', 'halo', 'dipole']):
    """"""
    
    acart = np.zeros(3) * u.pc*u.Myr**-2
    acyl = np.zeros(2) * u.pc*u.Myr**-2
    
    dict_acc = {'bary': [acc_bulge, acc_disk], 'halo': [acc_nfw], 'dipole': [acc_dipole]}
    accelerations = []
    for c in components:
        accelerations += dict_acc[c]
    
    for acc in accelerations:
        acart += acc(x)
    
    acyl[0] = np.sqrt(acart[0]**2 + acart[1]**2)
    acyl[1] = acart[2]
    
    return acyl

def ader_cyl(x, components=['bary', 'halo', 'dipole']):
    """"""
    
    acyl = np.zeros(2) * u.pc*u.Myr**-2
    dacyl = np.empty((2,0))
    
    dict_acc = {'bary': [acc_bulge, acc_disk], 'halo': [acc_nfw], 'dipole': [acc_dipole]}
    dict_der = {'bary': [der_bulge, der_disk], 'halo': [der_nfw], 'dipole': [der_dipole]}
    
    accelerations = []
    derivatives = []
    
    for c in components:
        accelerations += dict_acc[c]
        derivatives += dict_der[c]
    
    for acc, ader in zip(accelerations, derivatives):
        a_ = acc(x)
        da_ = ader(x)
        
        acyl[0] = np.sqrt(a_[0]**2 + a_[1]**2)
        acyl[1] = a_[2]
    
        da_cyl = np.empty((2,np.shape(da_)[1]))
        da_cyl[1] = da_[2]
        if acyl[0]!=0:
            da_cyl[0] = a_[0]/acyl[0]*da_[0] + a_[1]/acyl[0]*da_[1]
        else:
            #da_cyl[0] = np.zeros(np.shape(da_)[1])
            da_cyl[0] = np.sqrt(da_[0]**2 + da_[1]**2)
            #print(x, da_[2])
            #print(x, da_cyl[0])
        
        dacyl = np.hstack((dacyl, da_cyl))
    
    return dacyl
    
def crb_acyl(n, Ndim=6, vary=['progenitor', 'bary', 'halo', 'dipole'], component='all', align=True, d=20, Nb=50, fast=False, scale=True, relative=True):
    """"""
    pid, dp_fid, vlabel = get_varied_pars(vary)
    dp_opt = read_optimal_step(n, vary)
    dp = [x*y.unit for x,y in zip(dp_opt, dp_fid)]
    if align:
        alabel = '_align'
    else:
        alabel = ''
    if relative:
        vmin = 1e-2
        vmax = 1
        rlabel = ' / a'
    else:
        vmin = 1e-3
        vmax = 1e5
        rlabel =  ' (pc Myr$^{-2}$)'
    
    # read in full inverse CRB for stream modeling
    cxi = np.load('../data/crb/bspline_cxi{:s}_{:d}_{:s}_{:d}.npy'.format(alabel, n, vlabel, Ndim))
    if fast:
        cx = np.linalg.inv(cxi)
    else:
        cx = stable_inverse(cxi)
    
    # choose the appropriate components:
    Nprog, Nbary, Nhalo, Ndipole = [6, 5, 4, 3]
    nstart = {'bary': Nprog, 'halo': Nprog + Nbary, 'dipole': Nprog + Nbary + Nhalo, 'all': Nprog}
    nend = {'bary': Nprog + Nbary, 'halo': Nprog + Nbary + Nhalo, 'dipole': Nprog + Nbary + Nhalo + Ndipole, 'all': np.shape(cx)[0]}
    
    if component in ['bary', 'halo', 'dipole']:
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
        scale_vec = np.array([x.value for x in dp[nstart[component]:nend[component]]])
        scale_mat = np.outer(scale_vec, scale_vec)
        cqi *= scale_mat

    x0, v0 = gd1_coordinates()
    Rp = np.linalg.norm(x0[:2])
    zp = x0[2]
    
    R = np.linspace(-2*d, 2*d, 2*Nb)
    k = x0[1]/x0[0]
    x = R/np.sqrt(1+k**2)
    y = k * x
    
    z = np.linspace(-d, d, Nb)
    
    xv, zv = np.meshgrid(x, z)
    yv, zv = np.meshgrid(y, z)
    xin = np.array([np.ravel(xv), np.ravel(yv), np.ravel(zv)]).T

    Npix = np.size(xv)
    af = np.empty((Npix, 2))
    derf = np.empty((Npix, 2, Npot))
    
    for i in range(Npix):
        xi = xin[i]*u.kpc
        a = acc_cyl(xi, components=components)
        
        dqda = ader_cyl(xi, components=components)
        derf[i] = dqda
        
        cai = np.matmul(dqda, np.matmul(cqi, dqda.T))
        if fast:
            ca = np.linalg.inv(cai)
        else:
            ca = stable_inverse(cai)
        a_crb = np.sqrt(np.diag(ca)) * u.pc * u.Myr**-2
        if relative:
            af[i] = np.abs(a_crb/a)
        else:
            af[i] = a_crb
    
    # save
    np.savez('../data/crb_acyl{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}'.format(alabel, n, vlabel, component, Ndim, d, Nb, relative), acc=af, x=xin, der=derf)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(9,9))
    
    label = ['$\Delta$ $a_R$', '$\Delta$ $a_Z$']
    
    for i in range(2):
        plt.sca(ax[i])
        #im = plt.imshow(1 - af[:,i].reshape(Nb, 2*Nb)/np.flipud(af[:,i].reshape(Nb, 2*Nb)), origin='lower', extent=[-2*d, 2*d, -d, d], cmap=mpl.cm.RdBu, vmin=-1, vmax=1)
        im = plt.imshow(af[:,i].reshape(Nb, 2*Nb), origin='lower', extent=[-2*d, 2*d, -d, d], cmap=mpl.cm.gray, norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax)
        plt.plot(Rp, zp, 'r*', ms=10)
        #im = plt.imshow(xg[i].reshape(Nb, 2*Nb), origin='lower', extent=[0.1, 2*d, 0.1, d], cmap=mpl.cm.gray)
        
        plt.xlabel('R (kpc)')
        plt.ylabel('Z (kpc)')
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        plt.ylabel(label[i] + rlabel)
        
    plt.tight_layout()
    plt.savefig('../plots/crb_acc_cyl{:s}_{:d}_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}.png'.format(alabel, n, vlabel, component, Ndim, d, Nb, relative))

def acc_orbit(n, Ndim=6, vary=['halo', 'bary', 'progenitor'], align=True, d=20, Nb=50):
    """"""
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    # constraints
    data = np.load('../data/crb_acyl{:s}_{:d}_{:s}_{:d}_{:d}_{:d}.npz'.format(alabel, n, vlabel, Ndim, d, Nb))
    #print(np.median(data['acc'], axis=0))
    
    # orbit
    orbit = stream_orbit(n)
    R = np.linalg.norm(orbit['x'][:2,:].to(u.kpc), axis=0)
    z = orbit['x'][2].to(u.kpc)
    c = np.arange(np.size(z))
    
    # progenitor
    x0, v0 = gd1_coordinates()
    Rp = np.linalg.norm(x0[:2])
    zp = x0[2]
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(9,9))
    
    label = ['$\Delta$ $a_R$ / $a_R$', '$\Delta$ $a_Z$ / $a_Z$']
    
    for i in range(2):
        plt.sca(ax[i])
        im = plt.imshow(data['acc'][:,i].reshape(Nb, 2*Nb), origin='lower', extent=[0.1, 2*d, 0.1, d], cmap=mpl.cm.gray, norm=mpl.colors.LogNorm(), vmin=1e-2, vmax=1)
        #plt.plot(Rp, zp, 'r*', ms=10)
        
        plt.scatter(R[::-1], np.abs(z[::-1]), c=c[::-1], cmap=mpl.cm.YlGn_r, s=15)
        
        plt.xlabel('R (kpc)')
        plt.ylabel('|Z| (kpc)')
        plt.xlim(0.1, 2*d)
        plt.ylim(0.1, d)
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        plt.ylabel(label[i])
        
    plt.tight_layout()
    plt.savefig('../plots/crb_acc_prog{:s}_{:d}_{:s}_{:d}.png'.format(alabel, n, vlabel, Ndim))

def acc_symmetries(n, Ndim=6, vary=['progenitor', 'bary', 'halo', 'dipole'], align=True, d=20, Nb=50, relative=True):
    """Visualize whether CRBs on the acceleration field are N-S and E-W symmetric"""
    
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    data = np.load('../data/crb_acyl{:s}_{:d}_{:s}_all_{:d}_{:d}_{:d}_{:d}.npz'.format(alabel, n, vlabel, Ndim, d, Nb, relative))
    
    x0, v0 = gd1_coordinates()
    Rp = np.linalg.norm(x0[:2])
    zp = x0[2]
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(9,6), sharex='col', sharey=True)
    
    label = ['$\Delta$ $a_R$ / $a_R$', '$\Delta$ $a_Z$ / $a_Z$']
    
    for i in range(2):
        plt.sca(ax[i][0])
        im = plt.imshow((1 - data['acc'][:,i].reshape(Nb, 2*Nb)/np.flipud(data['acc'][:,i].reshape(Nb, 2*Nb))), origin='lower', extent=[-2*d, 2*d, -d, d], cmap=mpl.cm.RdBu, vmin=-1, vmax=1)
        plt.plot(Rp, zp, 'r*', ms=10)
        
        plt.xlim(-2*d, 2*d)
        plt.ylim(-d, d)
        
        if i==0:
            plt.title('N - S residuals', fontsize='medium')
        plt.sca(ax[i][1])
        im = plt.imshow((1 - data['acc'][:,i].reshape(Nb, 2*Nb)/np.fliplr(data['acc'][:,i].reshape(Nb, 2*Nb))), origin='lower', extent=[-2*d, 2*d, -d, d], cmap=mpl.cm.RdBu, vmin=-1, vmax=1)
        plt.plot(Rp, zp, 'r*', ms=10)
        
        plt.xlim(-2*d, 2*d)
        plt.ylim(-d, d)
        
        if i==0:
            plt.title('E - W residuals', fontsize='medium')
    
    plt.sca(ax[0][0])
    plt.ylabel('Z (kpc)')
    
    plt.sca(ax[1][0])
    plt.xlabel('R (kpc)')
    plt.ylabel('Z (kpc)')
    
    plt.sca(ax[1][1])
    plt.xlabel('R (kpc)')
    
    plt.tight_layout()
    plt.savefig('../plots/crb_acc_symmetry{:s}_{:d}_{:s}_{:d}_{:d}_{:d}_{:d}.png'.format(alabel, n, vlabel, Ndim, d, Nb, relative))

def acc_radialdep(n, Ndim=6, alldim=False, vary=['halo', 'bary', 'progenitor'], align=True, d=20, Nb=50):
    """Show radial dependence of the acceleration constraints"""
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    # plot constraints from a single observational setup
    if alldim==False:
        data = np.load('../data/crb_acyl{:s}_{:d}_{:s}_{:d}_{:d}_{:d}.npz'.format(alabel, n, vlabel, Ndim, d, Nb))
        
        plt.close()
        fig, ax = plt.subplots(2,1,figsize=(7,7), sharex=True, sharey=True)
        
        plt.sca(ax[0])
        plt.plot(np.linalg.norm(data['x'][:,:2], axis=1), np.tanh(data['acc'][:,0]), 'ko', ms=2)
        plt.ylim(0, 0.5)
        plt.ylabel('$\Delta$ $a_R$ / $|a_R|$')
        
        plt.sca(ax[1])
        plt.plot(np.linalg.norm(data['x'][:,:2], axis=1), np.tanh(data['acc'][:,1]), 'ko', ms=2)
        plt.xlabel('R (kpc)')
        plt.ylabel('$\Delta$ $a_z$ / $|a_z|$')
        
        plt.tight_layout()
        plt.savefig('../plots/crb_acc_radial{:s}_{:d}_{:s}_{:d}.png'.format(alabel, n, vlabel, Ndim))
    else:
        plt.close()
        fig, ax = plt.subplots(2,1,figsize=(7,7), sharex=True, sharey=True)
        
        labels = ['RA, Dec, d', 'RA, Dec, d, $V_r$', 'RA, Dec, d, $V_r$, $\mu_\\alpha$, $\mu_\delta$']
        colors = [mpl.cm.magma(x/3) for x in range(3)]
        
        for i, Ndim in enumerate([3,4,6]):
            data = np.load('../data/crb_acyl{:s}_{:d}_{:s}_{:d}_m_{:d}_{:d}.npz'.format(alabel, n, vlabel, Ndim, d, Nb))
            
            plt.sca(ax[0])
            plt.plot(np.linalg.norm(data['x'][:,:2], axis=1), np.tanh(data['acc'][:,0]), 'o', ms=2, color=colors[i], label=labels[i])
            
            plt.sca(ax[1])
            plt.plot(np.linalg.norm(data['x'][:,:2], axis=1), np.tanh(data['acc'][:,1]), 'o', ms=2, color=colors[i])
        
        plt.sca(ax[0])
        plt.ylim(0, 0.5)
        plt.ylabel('$\Delta$ $a_R$ / $|a_R|$')
        plt.legend(loc=1, frameon=False, fontsize='small', handlelength=0.5, markerscale=2.5)
        
        plt.sca(ax[1])
        plt.xlabel('R (kpc)')
        plt.ylabel('$\Delta$ $a_z$ / $|a_z|$')
        
        plt.tight_layout()
        plt.savefig('../plots/crb_acc_radial{:s}_{:d}_{:s}_alldim.png'.format(alabel, n, vlabel))

def acc_derivatives(vary=['halo', 'bary', 'progenitor'], n=-1, Ndim=6, align=True, d=20, Nb=50):
    """"""
    pid, dp, vlabel = get_varied_pars(vary)
    if align:
        alabel = '_align'
    else:
        alabel = ''
    
    # constraints
    data = np.load('../data/crb_acyl{:s}_{:d}_{:s}_{:d}_{:d}_{:d}.npz'.format(alabel, n, vlabel, Ndim, d, Nb))
    #print(np.median(data['acc'], axis=0))
    
    # orbit
    orbit = stream_orbit(n)
    R = np.linalg.norm(orbit['x'][:2,:].to(u.kpc), axis=0)
    z = orbit['x'][2].to(u.kpc)
    c = np.arange(np.size(z))
    
    # progenitor
    x0, v0 = gd1_coordinates()
    Rp = np.linalg.norm(x0[:2])
    zp = x0[2]
    
    plt.close()
    fig, ax = plt.subplots(9,2,figsize=(7,14))
    
    label = ['$a_R$', '$a_Z$']
    ylabel = ['$V_h$', '$R_h$', '$q_1$', '$q_z$', '$M_b$', '$a_b$', '$M_d$', '$a_d$', '$b_d$']
    
    for j in range(2):
        for i in range(9):
            plt.sca(ax[i][j])

            z1, z2 = zscale.zscale(data['der'][:,j,i].reshape(Nb, 2*Nb))
            im = plt.imshow(data['der'][:,j,i].reshape(Nb, 2*Nb), origin='lower', extent=[0.1, 2*d, 0.1, d], cmap=mpl.cm.Blues, vmin=z1, vmax=z2)
            
            plt.xlim(0.1, 2*d)
            plt.ylim(0.1, d)
            plt.text(0.9, 0.9, 'd{:s} / d{:s}'.format(label[j], ylabel[i]), fontsize='small', ha='right', va='top', transform=plt.gca().transAxes)
            
            if i==8:
                plt.xlabel('R (kpc)', fontsize='small')
                plt.setp(plt.gca().get_xticklabels(), fontsize='small')
            else:
                plt.setp(plt.gca().get_xticklabels(), visible=False)
            
            if j==0:
                plt.ylabel('|Z| (kpc)', fontsize='small')
                plt.setp(plt.gca().get_yticklabels(), fontsize='small')
            else:
                plt.setp(plt.gca().get_yticklabels(), visible=False)
            
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="3%", pad=0.1)
            plt.colorbar(im, cax=cax)
            plt.setp(plt.gca().get_yticklabels(), fontsize='x-small')
            
        
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/acc_derivatives_{:d}_{:s}_{:d}_{:d}.png'.format(n, vlabel, d, Nb))

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

def fancy_name(n):
    """Return nicely formatted stream name"""
    names = {-1: 'GD-1', -2: 'Palomar 5', -3: 'Triangulum', -4: 'ATLAS'}
    
    return names[n]

def prog_orbit3d(n, symmetry=False):
    """"""
    
    orbit = stream_orbit(n)

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
    plt.title('{}'.format(fancy_name(n)))
    
    plt.tight_layout()
    plt.savefig('../plots/orbit_3d_{}_{:d}.png'.format(n, symmetry))

def stream_orbit(n, pparams0=[0.5e10*u.Msun, 0.7*u.kpc, 6.8e10*u.Msun, 3*u.kpc, 0.28*u.kpc, 430*u.km/u.s, 30*u.kpc, 1.57*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1), 2.5e11*u.Msun, 0*u.deg, 0*u.deg, 0*u.kpc, 0*u.km/u.s, 0*u.mas/u.yr, 0*u.mas/u.yr], dt=0.2*u.Myr, rotmatrix=None, graph=False, observer=mw_observer, vobs=vsun, footprint='', obsmode='equatorial'):
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
    #stream.generate()
    #stream.observe(**params['observe'])
    stream.prog_orbit()
    
    return stream.orbit


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
                    theta = np.degrees(np.arccos(v[0][0]))
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
