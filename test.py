from pixell import enmap
import numpy as np
from orphics import maps,io,cosmology
from enlib import pointsrcs
from ptfit.ptfit import ptsrc_fit
    
np.random.seed(100)

# Stamp geometry
deg = 30./60.
px = 0.5
shape,wcs = maps.rect_geometry(width_deg=deg,px_res_arcmin=px,proj='car')
modlmap = enmap.modlmap(shape,wcs)

# CMB power
theory = cosmology.default_theory()
ells = np.arange(0,6000,1)
cltt = theory.lCl('TT',ells)
ps = cltt.reshape((1,1,ells.size))

# point source sim
beam = 1.5
true_flux = 200
sigma = np.deg2rad(beam/2./np.sqrt(2.*np.log(2.))/60.)
getsrc = lambda f: pointsrcs.sim_srcs(shape, wcs, np.array(((0.,0.,f),)), np.deg2rad(beam/60.))
src = getsrc(true_flux)

# CMB sim
beam_cmb = beam
kbeam = maps.gauss_beam(modlmap,beam_cmb)
m = maps.filter_map(enmap.rand_map(shape,wcs,ps),kbeam)
morig = m.copy()
io.plot_img(m)

m += src

# noise sim
noise_muK_arcmin = 10.0
nrealization = maps.white_noise(shape,wcs,noise_muK_arcmin)
io.plot_img(nrealization)
pmap = maps.psizemap(shape,wcs)*((180.*60./np.pi)**2.)
div = enmap.ones(shape,wcs)*pmap/noise_muK_arcmin**2.

m += nrealization
io.plot_img(m)

# fit
pflux,cov,fit = ptsrc_fit(m,0.,0.,np.deg2rad(beam/60.),div=div,ps=ps,beam=beam_cmb)
print(pflux,cov)
io.plot_img(m-fit)
