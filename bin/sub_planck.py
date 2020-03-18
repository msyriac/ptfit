from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,mpi,stats
from pixell import enmap,reproject,powspec
from enlib import pointsrcs
import numpy as np
import os,sys
import yaml
import ptfit
import traceback


"""
Removing point sources from Planck map.

We use the ACT merged point source catalog. We will remove
all of these point sources from the Planck maps.

We split the ACT catalog into 2 sets, A and B.

Set A has sources that are bright in Planck.
These will be fit for using the Planck coadd at each frequency.
This accounts for any variable sources in this set.

Set B consists of faint sources. For these, we will use
the best fit amplitude from ACT data, scaled by ratio
of beam solid angles for Planck and ACT. This will incorrectly
subtract a fraction of the faint sources that have varied
significantly between the time Planck and ACT data were taken.


How do we decide how to split the ACT catalog into Set A 
and Set B? We take amplitude_ACT (muK), scale by solid angle beam ratio, 
 and divide by
noise_per_beam_planck (muK) to get estimated S/N in planck.
If this is less than 1, we put it in Set B, else in Set A.

noise_per_beam_planck = wnoise (muK-arcmin) / beam_fwhm (arcmin)


cat_file columns
[name] [ra] [dec] [snr] [I] [Ierr] [?]

"""

sncut = 3.
freq = "p"+sys.argv[1]
yaml_file = "input/paths.yml"
with open(yaml_file) as f:
    paths = yaml.safe_load(f)
yaml_file = "input/planck.yml"
with open(yaml_file) as f:
    cplanck = yaml.safe_load(f)
pfwhm = cplanck[freq]['fwhm']



tasks,ras,decs,a_sns,a_amps,a_err_amps,famps,covs = np.loadtxt("results_p%s_fe1b666af90c30a72f4259546a586f86aa37d98a.txt" %  freq[1:] ,unpack=True)
amps = famps.copy()
amps[a_sns<=sncut] = a_amps[a_sns<=sncut].copy()
srcs = np.stack((np.deg2rad(decs),np.deg2rad(ras),amps)).T 


# Load map
for i in range(2):
    imap = enmap.read_map(paths['planck_files'] + "planck_hybrid_%s_2way_%d_map.fits" % (freq[1:],i),sel=np.s_[0,...])
    shape,wcs = imap.shape,imap.wcs
    model = pointsrcs.sim_srcs(shape[-2:], wcs, srcs, np.deg2rad(pfwhm/60.)) # wrong -- should be sigma?
    smap = imap - model
    enmap.write_map(paths['planck_files'] + "planck_hybrid_%s_2way_%d_map_I_srcfree.fits" % (freq[1:],i),smap)
    enmap.write_map(paths['planck_files'] + "planck_hybrid_%s_2way_%d_map_I_model.fits" % (freq[1:],i),model)
    
