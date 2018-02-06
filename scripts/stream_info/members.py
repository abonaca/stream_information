from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import astropy
from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import fits
from astropy.wcs import WCS
import gala.coordinates as gc

import sfd
import myutils
import zscale

import scipy.stats
import scipy.interpolate
import scipy.ndimage.filters as filters
import scipy.optimize

from os.path import expanduser
home = expanduser("~")

north = ['ACS', 'ATLAS', 'Ach', 'Coc', 'GD1', 'Hyl', 'Kwa', 'Let', 'Mol', 'Mur', 'NGC5466', 'Oph', 'Orp', 'PS1A', 'PS1B', 'PS1C', 'PS1D', 'PS1E', 'Pal5', 'San', 'Sca', 'Sty', 'TriPis']
mw_observer = {'z_sun': 27.*u.pc, 'galcen_distance': 8.3*u.kpc, 'roll': 0*u.deg, 'galcen_coord': coord.SkyCoord(ra=266.4051*u.deg, dec=-28.936175*u.deg, frame='icrs')}
vsun = {'vcirc': 237.8*u.km/u.s, 'vlsr': [11.1, 12.2, 7.3]*u.km/u.s}

sample = ['atlas', 'acheron', 'cocytos', 'gd1', 'hermus', 'kwando', 'lethe', 'molonglo', 'murrumbidgee', 'ngc5466', 'ophiuchus', 'orinoco', 'ps1a', 'ps1b', 'ps1c', 'ps1d', 'ps1e', 'pal5', 'sangarius', 'scamander', 'styx', 'tri']
mateudict = {'atlas': 'ATLAS', 'acheron': 'Ach', 'cocytos': 'Coc', 'gd1': 'GD1', 'hermus': 'Her', 'kwando': 'Kwa', 'lethe': 'Let', 'molonglo': 'Mol', 'murrumbidgee': 'Mur', 'ngc5466': 'NGC5466', 'ophiuchus': 'Oph', 'orinoco': 'Ori', 'ps1a': 'PS1A', 'ps1b': 'PS1B', 'ps1c': 'PS1C', 'ps1d': 'PS1D', 'ps1e': 'PS1E', 'pal5': 'Pal5', 'sangarius': 'San', 'scamander': 'Sca', 'styx': 'Sty', 'tri': 'TriPis'}

def split_extensions():
    """"""
    
    hdu = fits.open('/home/ana/data/mf_26dist_equ_FeHm1.5_12gyr_gr.fits.gz')
    hdu.info()
    Next = len(hdu)
    
    for i in range(Next):
        im = hdu[i].data
        hdr = hdu[i].header
        gen_hdr = hdu[0].header
        gen_hdr['EXTNAME'] = hdr['EXTNAME']
        
        hdunew_ = fits.PrimaryHDU(im)
        hdunew_.header = gen_hdr
        hdunew = fits.HDUList([hdunew_])
        hdunew.info()
        hdunew.writeto('/home/ana/data/ps1_maps/map_{:02d}.fits.gz'.format(i), overwrite=True)

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

def test(stream='ATLAS', i=19):
    """"""
    fname = '/home/ana/data/ps1_maps/map_{:02d}.fits.gz'.format(i)
    hdu = fits.open(fname)
    
    im = hdu[0].data
    hdr = hdu[0].header
    Nx, Ny = np.shape(im)

    w = WCS(fname)
    vec = np.array([[0,0], [Ny-1, Nx-1]]) #, [0, Nx-1], [Ny-1, 0]])
    corners = w.wcs_pix2world(vec, 0)
    decmin = np.min(corners[:,1])
    decmax = np.max(corners[:,1])
    ramin = np.min(corners[:,0]) - 360
    ramax = np.max(corners[:,0])
    
    s = Table.read('/home/ana/projects/python/galstreams/footprints/galstreams.footprint.ALL.dat', format='ascii.commented_header')
    #s.pprint()
    ind = s['IDst'] == stream
    #s = s[ind]
    ra, dec = atlas_track()
    
    plt.close()
    plt.figure(figsize=(10,4))
    
    plt.imshow(im, origin='lower', extent=[ramax, ramin, decmin, decmax], cmap='binary', vmin=0, vmax=0.0008)
    plt.plot(s['RA_deg'][ind], s['DEC_deg'][ind], 'r-', lw=2, alpha=0.1)
    #plt.plot(ra,dec, 'r-', lw=3, alpha=0.1)
    
    plt.xlim(ramax, ramin)
    plt.ylim(decmin, decmax)
    plt.xlabel('R.A. (deg)')
    plt.ylabel('Dec (deg)')
    
    plt.tight_layout()
    plt.savefig('../plots/streams/{}_{}.png'.format(stream, i), dpi=200)


def translate_name(name):
    """Return stream name from Mateu+2018"""
    name_dict = {'atlas': 'ATLAS', 'acheron': 'Ach', 'cocytos': 'Coc', 'gd1': 'GD1', 'kwando': 'Kwa', 'lethe': 'Let', 'molonglo': 'Mol', 'murrumbidgee': 'Mur', 'ngc5466': 'NGC5466', 'ophiuchus': 'Oph', 'orinoco': 'Ori', 'ps1a': 'PS1A', 'ps1b': 'PS1B', 'ps1c': 'PS1C', 'ps1d': 'PS1D', 'ps1e': 'PS1E', 'pal5': 'Pal5', 'sangarius': 'San', 'scamander': 'Sca', 'styx': 'Sty', 'tri': 'TriPis'}
    
    return name_dict[name]

def map_distances():
    """Save distances to PS1 match-filtered maps"""
    
    N = 26
    dist = np.empty(N)
    
    for i in range(N):
        hdul = fits.open('{}/data/ps1_maps/map_{:02d}.fits.gz'.format(home, i))
        extname = hdul[0].header['EXTNAME']
        dist[i] = float(extname.split(' ')[-2])
        hdul.close()
    
    np.save('../data/ps1_maps_distances', dist)

# stream track

# for streams in PS only
coords = []
def onclick_storecoords(event, fig, name, npoint):
    """Store coordinates of clicked points"""
    global ix, iy
    ix, iy = event.xdata, event.ydata

    global coords
    coords.append((ix, iy))
    print(len(coords))

    if len(coords)==npoint:
        print('done')
        carr = np.array(coords)
        c = Table(np.array(coords), names=('ra', 'dec'))
        
        c.pprint()
        c.write('../data/streams/{}_coords.fits'.format(name), overwrite=True)
        coords = []

def plot_ps1(name='ps1a', get_coords=False, npoint=10):
    """"""
    
    shape = Table.read('../data/streams/stream_shape.txt', format='ascii.commented_header')
    dist = shape['d'][shape['name']==name]
    
    map_dist = np.load('../data/ps1_maps_distances.npy')
    idist = np.argmin(np.abs(map_dist - dist))
    #print(dist, idist)
    
    hdul = fits.open('{}/data/ps1_maps/map_{:02d}.fits.gz'.format(home, idist))
    data = hdul[0].data
    head = hdul[0].header
    map_distance = head['EXTNAME']
    #print(map_distance, dist, idist)
    
    t = Table.read('{}/projects/python/galstreams/footprints/galstreams.footprint.ALL.dat'.format(home), format='ascii.commented_header')
    ind = t['IDst']==translate_name(name)
    t = t[ind]
    
    dx = 3
    ra1 = np.min(t['RA_deg']) - dx
    ra2 = np.max(t['RA_deg']) + dx
    dec1 = np.min(t['DEC_deg']) - dx
    dec2 = np.max(t['DEC_deg']) + dx
    
    d = 5
    ramin = np.int64(np.floor(ra1/d))*d
    ramax = np.int64(np.ceil(ra2/d))*d
    decmin = np.int64(np.floor(dec1/d))*d
    decmax = np.int64(np.ceil(dec2/d))*d
    
    decmin = max(decmin, -30)
    dec1 = max(dec1, -30)
    
    # PS-1 tiles to download
    xg = np.arange(ramin, ramax, d)
    yg = np.arange(decmin, decmax, d)
    xx, yy = np.meshgrid(xg, yg)
    tf = Table([xx.ravel(), yy.ravel()], names=('ra', 'dec'))
    tf = tf[tf['dec']>=-30]
    #tf.pprint()
    tf.write('../data/streams/tiles_{}'.format(name), format='ascii.no_header', overwrite=True)
    
    # slice array
    ra_fid = head['CRVAL1']
    dec_fid = head['CRVAL2']
    dx = head['CDELT1']
    dy = head['CDELT2']
    x_fid = head['CRPIX1']
    y_fid = head['CRPIX2']
    
    i1 = np.int64(x_fid + dx**-1*(ramin - ra_fid))
    i2 = np.int64(x_fid + dx**-1*(ramax - ra_fid))
    j1 = np.int64(y_fid + dy**-1*(decmin - dec_fid))
    j2 = np.int64(y_fid + dy**-1*(decmax - dec_fid))
    
    if (ra1<0) | (ra2>360):
        Nx, Ny = np.shape(data)
        i1 = Nx
        i2 = 0
    
    print(ra1, ra2, dec1, dec2)
    print(j1, j2, i2, i1)
    data = data[j1:j2,i2:i1]
    
    # smooth
    nsmooth = 5
    for i in range(nsmooth):
        data = filters.gaussian_filter(data, 1)
    data -= np.min(data)
    data += 0.01
    data /= np.max(data)
    
    da = 6
    oa = 2
    delta_ra = np.abs(ramax - ramin)
    delta_dec = np.abs(decmax - decmin)
    hwr = delta_dec / delta_ra
    h = da + oa
    w = 2 * da / hwr + oa
    
    print(hwr, h, w)
    vmin, vmax = zscale.zscale(data)
    
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(w,h))
    
    plt.sca(ax[0])
    plt.imshow(data, origin='lower', extent=[ramax, ramin, decmin, decmax], cmap='binary', norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax)
    plt.xlabel('RA')
    plt.ylabel('Dec')
    
    plt.sca(ax[1])
    plt.imshow(data, origin='lower', extent=[ramax, ramin, decmin, decmax], cmap='binary', norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax)
    isort = np.argsort(t['RA_deg'])
    plt.plot(t['RA_deg'][isort], t['DEC_deg'][isort], 'r-', alpha=0.2)
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.ylim(decmin, decmax)
    
    if get_coords:
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick_storecoords(event, fig, name, npoint))
    
    plt.tight_layout()

# for streams with kinematics
def reformat_input(name='ophiuchus'):
    """Create an input file for getting the streamline
    Also creates a list of PS1 tiles to download"""
    
    t = Table.read('../data/streams/{}_input.txt'.format(name), format='ascii.commented_header')
    #t.pprint()
    
    d = dm2d(t['dm'])
    de = 0.5 * (dm2d(t['dm'] + t['dme']) - dm2d(t['dm'] - t['dme']))
    c = coord.SkyCoord(ra=t['ra']*u.deg, dec=t['dec']*u.deg, distance=d.to(u.kpc), **mw_observer)
    cgal = c.transform_to(coord.Galactic)
    #print(cgal)
    v = gc.vhel_to_gal(c, rv=t['vr']*u.km/u.s, pm=[t['mul']*u.mas/u.yr, t['mub']*u.mas/u.yr], **vsun)
    veq = gc.vgal_to_hel(c, v, **vsun)
    #print(veq[-1].to(u.km/u.s))
    
    tout = Table([t['ra']*u.deg, t['dec']*u.deg, d.to(u.kpc), de.to(u.kpc), veq[-1].to(u.km/u.s), t['vre'], veq[0].to(u.mas/u.yr), t['mule'], veq[1].to(u.mas/u.yr), t['mube']], names=('ra', 'dec', 'd', 'd_err', 'vr', 'vr_err', 'pmra', 'pmra_err', 'pmdec', 'pmdec_err'))
    tout.pprint()
    tout.write('../data/streams/{}_coords.fits'.format(name), overwrite=True)
    
    # stream limits
    dx = 3
    ra1 = np.min(t['ra']) - dx
    ra2 = np.max(t['ra']) + dx
    dec1 = np.min(t['dec']) - dx
    dec2 = np.max(t['dec']) + dx
    
    d = 5
    ramin = np.int64(np.floor(ra1/d))*d
    ramax = np.int64(np.ceil(ra2/d))*d
    decmin = np.int64(np.floor(dec1/d))*d
    decmax = np.int64(np.ceil(dec2/d))*d
    
    decmin = max(decmin, -30)
    dec1 = max(dec1, -30)
    
    # PS-1 tiles to download
    xg = np.arange(ramin, ramax, d)
    yg = np.arange(decmin, decmax, d)
    xx, yy = np.meshgrid(xg, yg)
    tf = Table([xx.ravel(), yy.ravel()], names=('ra', 'dec'))
    tf = tf[tf['dec']>=-30]
    tf.write('../data/streams/tiles_{}'.format(name), format='ascii.no_header', overwrite=True)

def dm2d(dm):
    """Convert a distance modulus to distance"""
    d = 10**(0.2*(dm+5)) * u.pc
    
    return d

# for literature tracks
def g17b_endpoints():
    """"""
    
    #kwando: 210,-85; 230,-75
    #molonglo 70,-62; 70,-90
    #murrumbidgee: 93,-15; 210, -85
    #orinoco: 230,-83; 18,-45
    l = np.array([210, 230, 70, 70, 93, 210, 230, 18])*u.deg
    b = np.array([-85, -75, -62, -90, -15, -85, -83, -45])*u.deg
    
    gc = coord.SkyCoord(l=l, b=b, frame=coord.Galactic)
    print(gc.icrs)
    
def poly_tracks(Ncoord=20):
    """Create list of members based on literature tracks"""
    
    # first element lowest order -- exactly opposite of numpy convention
    poly = {'hermus': [241.571, 1.37841, -0.148870, +0.00589502, -1.03927e-4, 7.28133e-7],
            'kwando': [-7.817, -2.354, 0.1202, -0.00215],
            'molonglo': [345.017, -0.5843, 0.01822],
            'murrumbidgee': [367.893, -0.4647, -0.008626, 0.000118, 1.2347e-6, -1.13758e-7],
            'orinoco': [-25.5146, 0.1672, -0.003827, -0.0002835, -5.3133e-6],
            'ps1d': [141.017, 0.208, -0.02491, 0.000609, -1.20989e-6],
            'sangarius': [148.9492, -0.03811, 0.001505],
            'scamander': [155.642, -0.1, -0.00191, -0.0003346, 1.47775e-5]}
    direction = {'hermus': 'ra', 'kwando': 'dec', 'molonglo': 'ra', 'murrumbidgee': 'ra', 'orinoco': 'dec', 'ps1d': 'ra', 'sangarius': 'ra', 'scamander': 'ra'}
    minmax = {'hermus': [0,40], 'kwando': [18, 30], 'molonglo': [-27, -8], 'murrumbidgee': [-27, 39], 'orinoco': [0,20], 'scamander': [-5,35], 'ps1d': [-5, 25], 'sangarius': [-5, 35]}
    streams_ = direction.keys()
    Nst = len(streams_)
    
    t = Table.read('/home/ana/projects/python/galstreams/footprints/galstreams.footprint.ALL.dat', format='ascii.commented_header')
    
    for i in range(Nst):
        name = streams_[i]
        mname = mateudict[name]
        print(streams_[i], np.sum(t['IDst']==mname))
        
        istream = t['IDst']==mname
        p = np.poly1d(np.array(poly[name])[::-1])
        print(name, p)
        
        if direction[name]=='ra':
            dec = np.linspace(minmax[name][0], minmax[name][1], Ncoord)
            ra = p(dec)
        else:
            ra = np.linspace(minmax[name][0], minmax[name][1], Ncoord)
            dec = p(ra)
        
        tout = Table(np.array([ra, dec]).T, names=('ra', 'dec'))
        tout.write('../data/streams/{}_coords.fits'.format(name), overwrite=True)
        
        stream_tiles(name=name)


def stream_tiles(name='atlas', dx=1):
    """Output a list of PS1 tiles to download"""
    
    t = Table.read('../data/streams/{}_coords.fits'.format(name))
    # stream limits
    ra1 = np.min(t['ra']) - dx
    ra2 = np.max(t['ra']) + dx
    dec1 = np.min(t['dec']) - dx
    dec2 = np.max(t['dec']) + dx
    
    d = 5
    ramin = np.int64(np.floor(ra1/d))*d
    ramax = np.int64(np.ceil(ra2/d))*d
    decmin = np.int64(np.floor(dec1/d))*d
    decmax = np.int64(np.ceil(dec2/d))*d
    
    decmin = max(decmin, -30)
    dec1 = max(dec1, -30)
    
    # PS-1 tiles to download
    xg = np.arange(ramin, ramax, d)
    yg = np.arange(decmin, decmax, d)
    xx, yy = np.meshgrid(xg, yg)
    
    ind = xx>=360
    xx[ind] -= 360
    
    tf = Table([xx.ravel(), yy.ravel()], names=('ra', 'dec'))
    tf = tf[tf['dec']>=-30]
    tf.write('../data/streams/tiles_{}'.format(name), format='ascii.no_header', overwrite=True)


def poly_streamline(name='atlas', deg=3):
    """Fit polynomial to the stream track"""
    
    t = Table.read('../data/streams/{}_coords.fits'.format(name))
    t.pprint()
    
    # decide whether ra or dec is a dependent variable
    dra = np.max(t['ra']) - np.min(t['ra'])
    ddec = np.max(t['dec']) - np.min(t['dec'])
    
    if ddec>dra:
        x_ = t['dec']
        y_ = t['ra']
        var = 'dec'
    else:
        x_ = t['ra']
        y_ = t['dec']
        var = 'ra'
    
    print(var)
    
    # fit a polynomial
    p = np.polyfit(x_, y_, deg)
    polybest = np.poly1d(p)
    print(p)
    np.savez('../data/streams/{}_poly'.format(name), p=p, var=var)
    
    x = np.linspace(np.min(x_), np.max(x_), 100)
    y = np.polyval(polybest, x)
    if var=='ra':
        R = find_greatcircle(x, y)
    else:
        R = find_greatcircle(y, x)
    
    np.save('../data/streams/{}_rotmat'.format(name), R)
    
    xi = np.linspace(35, 100, 100)
    eta = np.ones(100)
    ra1, dec1 = myutils.rotate_angles(xi, eta, np.linalg.inv(R))
    if np.any(t['ra']>360):
        ind = ra1<180
        ra1[ind] +=360
    
    plt.close()
    plt.figure()
    
    plt.plot(t['ra'], t['dec'], 'k.', ms=4)
    if var=='ra':
        plt.plot(x, y, 'r-')
    else:
        plt.plot(y, x, 'r-')
    plt.plot(ra1, dec1, 'b-')
    
    xi = np.linspace(35, 100, 100)
    eta = np.ones(100) * -1
    ra1, dec1 = myutils.rotate_angles(xi, eta, np.linalg.inv(R))
    if np.any(t['ra']>360):
        ind = ra1<180
        ra1[ind] +=360
    
    plt.plot(ra1, dec1, 'b-')
    
    plt.xlabel('R.A. (deg)')
    plt.ylabel('Dec (deg)')
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('../plots/streams/{}_track.png'.format(name))

def sph2cart(ra, dec):
    """Convert two angles on a unit sphere to a 3d vector"""
    
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    
    return (x, y, z)

def find_greatcircle(ra_deg, dec_deg):
    """Save rotation matrix for a stream model"""
    
    #stream = stream_model(name, pparams0=pparams, dt=dt)
    
    ## find the pole
    #ra = np.radians(stream.obs[0])
    #dec = np.radians(stream.obs[1])
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    
    rx = np.cos(ra) * np.cos(dec)
    ry = np.sin(ra) * np.cos(dec)
    rz = np.sin(dec)
    r = np.column_stack((rx, ry, rz))
    #r = sph2cart(ra, dec)

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
    
    xi, eta = myutils.rotate_angles(ra_deg, dec_deg, R)
    
    # put xi = 50 at the beginning of the stream
    xi[xi>180] -= 360
    xi += 360
    xi0 = np.min(xi) - 50
    R2 = myutils.rotmatrix(-xi0, 2)
    R = np.dot(R2, np.matmul(R1, R0))
    xi, eta = myutils.rotate_angles(ra_deg, dec_deg, R)
    
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


import sfd

def make_catalog(name='atlas', cut=True):
    """"""
    
    d = 5
    ra, dec = np.loadtxt('../data/streams/tiles_{}'.format(name), unpack=True)
    ra1 = np.min(ra)
    ra2 = np.max(ra) + d
    dec1 = np.min(dec)
    dec2 = np.max(dec) + d
    print(ra1, ra2, dec1, dec2)
    
    t = read_rect([ra1, ra2, dec1, dec2])

    # select stars
    stars = np.abs(t['median_ap'][:,2] - t['median'][:,2]) < 0.2
    if cut:
        t = t[stars]
        label = ''
    else:
        label = '_nocut'
    
    # correct for reddening
    c = coord.SkyCoord(ra=t['ra_ok']*u.deg, dec=t['dec_ok']*u.deg)
    reddening = sfd.reddening(c, survey='PS1', filters='gri')
    
    tout = Table(np.array([t['ra_ok'], t['dec_ok'], t['median'][:,0] - reddening[:,0], t['median'][:,1] - reddening[:,1], t['median'][:,2] - reddening[:,2]]).T, names=('ra', 'dec', 'g', 'r', 'i'))
    
    tout.write('../data/streams/{:s}_catalog{:s}.fits'.format(name, label), overwrite=True)

def read_rect(rect, d=5, verbose=False):
    """Read PS1 tiles"""
    
    ra1, ra2, dec1, dec2 = rect
    
    ramin = np.int64(np.floor(ra1/d))*d
    ramax = np.int64(np.ceil(ra2/d))*d
    decmin = np.int64(np.floor(dec1/d))*d
    decmax = np.int64(np.ceil(dec2/d))*d
    
    for i_, ra in enumerate(range(ramin, ramax, d)):
        for j_, dec in enumerate(range(decmin, decmax, d)):
            if verbose: print('reading {} {}'.format(ra, dec))
            
            tin = Table.read(home+'/data/ps1/ps1_{:d}.{:d}.fits'.format(ra, dec))
            ind = (tin['ra']<=ra2) & (tin['ra']>=ra1) & (tin['dec']<=dec2) & (tin['dec']>=dec1)
            tin = tin[ind]
            
            if (i_==0) & (j_==0):
                tout = tin.copy()
            else:
                tout = astropy.table.vstack([tout, tin])
    
    return(tout)

def stream_coords(name='atlas'):
    """"""
    # streakline
    ts = Table.read('../data/streams/{}_coords.fits'.format(name))
    dp_ = np.load('../data/streams/{}_poly.npz'.format(name))
    p_ = dp_['p']
    var = dp_['var']
    poly = np.poly1d(p_)
    
    if var=='ra':
        x = np.linspace(np.min(ts['ra']), np.max(ts['ra']), 1000)
    else:
        x = np.linspace(np.min(ts['dec']), np.max(ts['dec']), 1000)
    y = np.polyval(poly, x)
    if var=='ra':
        q = sph2cart(np.radians(x), np.radians(y))
    else:
        q = sph2cart(np.radians(y), np.radians(x))
    
    # consider only stars within 1 deg of the best-fitting great circle
    t = Table.read('../data/streams/{}_catalog.fits'.format(name))
    R = np.load('../data/streams/{}_rotmat.npy'.format(name))
    xi, eta = myutils.rotate_angles(t['ra'], t['dec'], R)
    ind = np.abs(eta)<1
    t = t[ind]

    if var=='ra':
        xi, eta = myutils.rotate_angles(x, y, R)
    else:
        xi, eta = myutils.rotate_angles(y, x, R)

    p = sph2cart(np.radians(t['ra']), np.radians(t['dec']))
    
    distances = np.tensordot(p, q, axes=([0,],[0,]))
    distances = np.arccos(distances)
    idmin = np.argmin(distances, axis=1)
    
    l = xi[idmin]
    b = np.min(distances, axis=1)
    b[t['dec']<y[idmin]] *= -1
    b = np.degrees(b)
    
    tout = Table(np.array([t['ra'], t['dec'], l, b, t['g'], t['r'], t['i']]).T, names=('ra', 'dec', 'l', 'b', 'g', 'r', 'i'))
    tout.write('../data/streams/{}_allcoords.fits'.format(name), overwrite=True)
    
    plt.close()
    plt.figure()
    
    plt.plot(tout['ra'], tout['dec'], 'k.')
    plt.plot(ts['ra'], ts['dec'], 'ro')

def filter_weights(t, xe, ye, filter, b1='g', b2='r', d0=0, d=0):
    """"""
    
    # shift filter in distance
    if d0!=0:
        ye = ye - 5*np.log10(d0*100)
        ye = ye + 5*np.log10(d*100)
    
    xi = np.searchsorted(xe, t[b1] - t[b2]) - 1
    yi = np.searchsorted(ye, t[b2]) - 1
    inside = (xi<np.size(xe)-1) & (xi>=0) & (yi<np.size(ye)-1) & (yi>=0)
    
    w = np.zeros(len(t))
    w[inside] = filter[xi[inside],yi[inside]]
    
    return w

def stream_probabilities(name='atlas', sigma=0.25, d=22, Ntop=100, sigma_d=2, seed=98):
    """"""
    t = Table.read('../data/streams/{}_allcoords.fits'.format(name))
    s = Table.read('../data/streams/stream_shape.txt', format='ascii.commented_header')
    s = s[s['name']==name]
    sigma = s['sigma_w']
    d = s['d']
    sigma_d = s['sigma_d']
    
    # spatial
    gauss = scipy.stats.norm(0, sigma)
    prob_spatial = gauss.pdf(t['b'])
    prob_spatial /= np.max(prob_spatial)
    
    f = np.load('../data/m13_filter.npz')
    prob_cmd = filter_weights(t, f['color_bin'], f['mag_bin'], f['filter'], d0=f['dist'], d=d, b1='g', b2='i')
    prob_cmd /= np.max(prob_cmd)
    
    prob_tot = prob_cmd * prob_spatial
    
    # select top 100
    ind = np.argpartition(prob_tot, -Ntop)[-Ntop:]
    t = t[ind]
    prob_tot = prob_tot[ind]
    
    nanvec = np.ones_like(t['ra']) * np.nan
    err_astrometry = np.ones_like(t['ra']) * (0.5*u.arcsec).to(u.deg).value
    np.random.seed(seed)
    arr_d = np.random.randn(Ntop)*sigma_d + d
    err_d = np.ones(Ntop) * sigma_d
    
    tout = Table(np.array([t['ra'], err_astrometry, t['dec'], err_astrometry, arr_d, err_d, nanvec, nanvec, nanvec, nanvec, nanvec, nanvec, prob_tot]).T, names=('ra', 'ra_err', 'dec', 'dec_err', 'd', 'd_err', 'vr', 'vr_err', 'pmra', 'pmra_err', 'pmdec', 'pmdec_err', 'p'), dtype=('f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f'))
    
    tout.pprint()
    tout.write('../data/lib/{}_members.fits'.format(name), overwrite=True)
    
    plt.close()
    plt.figure()
    
    plt.scatter(t['ra'], t['dec'], s=prob_tot*4)
    ##plt.plot(t['g']-t['r'], t['r'], 'k.', ms=1, alpha=0.05)
    #plt.scatter(t['g']-t['r'], t['r'], s=prob_tot*4)
    #plt.xlim(-0.5,1)
    #plt.ylim(24,14)

def stream_kinematics(name='atlas'):
    """Add kinematic data to the members' list"""
    
    t = Table.read('../data/lib/{}_members.fits'.format(name))
    tkin = Table.read('../data/streams/{}_coords.fits'.format(name))
    tkin.pprint()
    
    if ('vr' in tkin.colnames) & (np.all(np.isnan(t['vr']))):
        Nkin = len(tkin)
        
        ind_min = t['p'].argsort()[:Nkin]
        
        for k in ['d', 'd_err', 'vr', 'vr_err', 'pmra', 'pmra_err', 'pmdec', 'pmdec_err']:
            t[k][ind_min] = tkin[k]
        
        p = np.ones(Nkin)
        t['p'][ind_min] = p
        
        t.pprint()
        t.write('../data/lib/{}_members.fits'.format(name), overwrite=True)
        
        
