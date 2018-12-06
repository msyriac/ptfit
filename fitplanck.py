from __future__ import print_function
from orphics import maps,io,cosmology,catalogs
from pixell import enmap,reproject
import numpy as np
import os,sys
import ptfit

cols = catalogs.load_fits("/home/msyriac/data/planck/COM_PCCS_143_R2.01.fits",['RA','DEC','GAUFLUX','GAUFLUX_ERR'])
ras = cols['RA']
decs = cols['DEC']
fluxes = cols['GAUFLUX']
sn = fluxes / cols['GAUFLUX_ERR']
# print(sn.min(),sn.max(),sn.mean())
# io.hist(sn,bins=np.linspace(0,20,20))
# sys.exit()
fluxcut = 0
# io.hist(fluxes,bins=np.linspace(0,1000,40))
# print(fluxes.min(),fluxes.max(),len(fluxes[fluxes>fluxcut]))
# sys.exit()

imap = enmap.read_map("/home/msyriac/data/depot/xlens/HFI_SkyMap_143_2048_R2.02_full_cutout_h0.fits")
arc = 20.
npix = int(arc/0.5)

fwhm = 7.0
noise = 33.
rs = np.deg2rad(np.linspace(0,90.,10000))
rbeam = maps.gauss_beam_real(rs,fwhm)

ells = np.arange(0,8000,1)
theory = cosmology.default_theory()
ps = theory.lCl('TT',ells)[None,None,...]

fras = []
fdecs = []
fflux = []
oflux = []

for ra,dec,flux in zip(ras,decs,fluxes):

    
    stamp = reproject.cutout(imap, width=None, ra=np.deg2rad(ra), dec=np.deg2rad(dec), pad=1,  npix=npix)
    if stamp is None: continue
    if flux>fluxcut:
        # io.plot_img(stamp,lim=300)
        pass
    else:
        continue
    # print(stamp.shape)

    modlmap = stamp.modlmap()
    n2d = modlmap*0. + (noise*np.pi/180./60.)**2.
    #n2d[modlmap>6000] = 0
    pflux,cov,fit = ptfit.ptsrc_fit(stamp,np.deg2rad(dec),np.deg2rad(ra),(rs,rbeam),div=None,ps=ps,beam=fwhm,iau=False,
              n2d=n2d,totp2d=None)
    print(pflux)
    # io.plot_img(fit)
    # io.plot_img(stamp-fit,lim=300)
    fras.append(ra)
    fdecs.append(dec)
    fflux.append(pflux)
    oflux.append(flux)

io.save_cols("d56_fluxes.txt",(fras,fdecs,fflux,oflux))
    

