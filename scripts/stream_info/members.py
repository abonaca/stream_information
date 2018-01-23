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

import sfd
import myutils

import scipy.stats
import scipy.interpolate
import scipy.ndimage.filters as filters
import scipy.optimize

from os.path import expanduser
home = expanduser("~")

north = ['ACS', 'ATLAS', 'Ach', 'Coc', 'GD1', 'Hyl', 'Kwa', 'Let', 'Mol', 'Mur', 'NGC5466', 'Oph', 'Orp', 'PS1A', 'PS1B', 'PS1C', 'PS1D', 'PS1E', 'Pal5', 'San', 'Sca', 'Sty', 'TriPis']


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
    print(map_distance)
    
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
    
    # PS-1 tiles to download
    xg = np.arange(ramin, ramax, d)
    yg = np.arange(decmin, decmax, d)
    xx, yy = np.meshgrid(xg, yg)
    tf = Table([xx.ravel(), yy.ravel()], names=('ra', 'dec'))
    tf.pprint()
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
    
    data = data[j1:j2,i2:i1]
    
    # smooth
    nsmooth = 5
    for i in range(nsmooth):
        data = filters.gaussian_filter(data, 1)
    data -= np.min(data)
    data += 0.01
    data /= np.max(data)
    
    h = 8
    w = h * (ramax-ramin) / (decmax - decmin)
    
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(2*w,h))
    
    plt.sca(ax[0])
    plt.imshow(data, origin='lower', extent=[ramax, ramin, decmin, decmax], cmap='binary', norm=mpl.colors.LogNorm())
    plt.xlabel('RA')
    plt.ylabel('Dec')
    
    plt.sca(ax[1])
    plt.imshow(data, origin='lower', extent=[ramax, ramin, decmin, decmax], cmap='binary', norm=mpl.colors.LogNorm())
    isort = np.argsort(t['RA_deg'])
    plt.plot(t['RA_deg'][isort], t['DEC_deg'][isort], 'r-', alpha=0.2)
    plt.xlabel('RA')
    plt.ylabel('Dec')
    
    if get_coords:
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick_storecoords(event, fig, name, npoint))
    
    plt.tight_layout()


def poly_streamline(name='atlas'):
    """"""
    
    t = Table.read('../data/streams/{}_coords.fits'.format(name))
    t.pprint()
    
    # fit a polynomial
    deg = 3
    p = np.polyfit(t['ra'], t['dec'], deg)
    polybest = np.poly1d(p)
    print(p)
    np.savetxt('../data/streams/{}_poly.txt'.format(name), p)
    
    x = np.linspace(np.min(t['ra']), np.max(t['ra']), 100)
    y = np.polyval(polybest, x)
    R = find_greatcircle(x, y)
    np.save('../data/streams/{}_rotmat'.format(name), R)
    
    xi = np.linspace(50, 75, 100)
    eta = np.ones(100)
    ra1, dec1 = myutils.rotate_angles(xi, eta, np.linalg.inv(R))
    
    plt.close()
    plt.figure()
    
    plt.plot(t['ra'], t['dec'], 'ko')
    plt.plot(x, y, 'r-')
    plt.plot(ra1, dec1, 'b-')
    
    xi = np.linspace(50, 75, 100)
    eta = np.ones(100) * -1
    ra1, dec1 = myutils.rotate_angles(xi, eta, np.linalg.inv(R))
    plt.plot(ra1, dec1, 'b-')
    
    plt.xlabel('R.A. (deg)')
    plt.ylabel('Dec (deg)')
    
    plt.tight_layout()

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


def stream_coords(name='atlas'):
    """"""
    # streakline
    ts = Table.read('../data/streams/{}_coords.fits'.format(name))
    p_ = np.loadtxt('../data/streams/{}_poly.txt'.format(name))
    poly = np.poly1d(p_)
    
    x = np.linspace(np.min(ts['ra']), np.max(ts['ra']), 1000)
    y = np.polyval(poly, x)
    q = sph2cart(np.radians(x), np.radians(y))
    
    # consider only stars within 1 deg of the best-fitting great circle
    t = Table.read('../data/streams/{}_catalog.fits'.format(name))
    R = np.load('../data/streams/{}_rotmat.npy'.format(name))
    xi, eta = myutils.rotate_angles(t['ra'], t['dec'], R)
    ind = np.abs(eta)<1
    t = t[ind]

    xi, eta = myutils.rotate_angles(x, y, R)

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
    
    plt.plot(ts['ra'], ts['dec'], 'ko')

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
    
    # spatial
    gauss = scipy.stats.norm(0, sigma)
    prob_spatial = gauss.pdf(t['b'])
    prob_spatial /= np.max(prob_spatial)
    
    f = np.load('../data/m13_filter.npz')
    prob_cmd = filter_weights(t, f['color_bin'], f['mag_bin'], f['filter'], d0=f['dist'], d=d, b1='g', b2='i')
    prob_cmd /= np.max(prob_cmd)
    
    prob_tot = prob_cmd * prob_spatial
    
    # select top 100
    ind = np.argpartition(prob_cmd, -Ntop)[-Ntop:]
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
