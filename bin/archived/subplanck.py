from __future__ import print_function
from orphics import maps,io,cosmology,catalogs
from pixell import enmap,reproject
from enlib import pointsrcs
import numpy as np
import os,sys
import ptfit

imap = enmap.read_map("/home/msyriac/data/depot/xlens/HFI_SkyMap_143_2048_R2.02_full_cutout_h0.fits")
shape,wcs = imap.shape,imap.wcs
fwhm = 7.0
noise = 33.
rs = np.deg2rad(np.linspace(0,3.,10000))
rbeam = maps.gauss_beam_real(rs,fwhm)

ells = np.arange(0,8000,1)
theory = cosmology.default_theory()
ps = theory.lCl('TT',ells)[None,None,...]

fras = []
fdecs = []
fflux = []

ras,decs,fflux = np.loadtxt("d56_fluxes.txt",unpack=True)
print(decs)
srcs = np.zeros((len(ras),3))
srcs[:,0] = np.deg2rad(decs)
srcs[:,1] = np.deg2rad(ras)
srcs[:,2] = fflux
print(fflux[0])
template = pointsrcs.sim_srcs(shape[-2:], wcs, srcs, (rs,rbeam))
# npix = 80
# stamp = reproject.cutout(imap,  ra=np.deg2rad(ras[0]), dec=np.deg2rad(decs[0]),  npix=npix)
# io.plot_img(stamp)
# stamp = reproject.cutout(template,  ra=np.deg2rad(ras[0]), dec=np.deg2rad(decs[0]),  npix=npix)
# io.plot_img(stamp)

# io.plot_img(template,"template.png",lim=300)
io.hplot(template,"template")
io.hplot(imap,"imap")
io.hplot(imap-template,"subbed")
