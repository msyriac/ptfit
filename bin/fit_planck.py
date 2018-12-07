from __future__ import print_function
from orphics import maps,io,cosmology,mpi,stats
from pixell import enmap,reproject,powspec
import numpy as np
import os,sys
import yaml
import ptfit


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

def solid_angle(fwhm):
    sigma = np.deg2rad(fwhm/60.)/(8*np.log(2))**0.5 
    return 2*np.pi*sigma**2
    

def correct_amplitude(amp,act_fwhm,planck_fwhm):
    planck_solid_angle = solid_angle(planck_fwhm)
    act_solid_angle = solid_angle(act_fwhm)
    return amp * act_solid_angle / planck_solid_angle


sncut = 30.
arc = 30.
npix = int(arc/0.5)
freq = "p"+sys.argv[1]
yaml_file = "input/paths.yml"
with open(yaml_file) as f:
    paths = yaml.safe_load(f)
yaml_file = "input/planck.yml"
with open(yaml_file) as f:
    cplanck = yaml.safe_load(f)
afreq = cplanck[freq]['afreq']
tnoise = cplanck[freq]['tnoise']
pfwhm = cplanck[freq]['fwhm']
parea = cplanck[freq]['area']
afwhm = cplanck["a"+afreq]['fwhm']
cat_file = paths['cat_files'].replace('???',afreq)
ras,decs,act_amps = np.loadtxt(cat_file,usecols=[1,2,4],unpack=True)
amps = correct_amplitude(act_amps,afwhm,pfwhm)
noise = tnoise/np.sqrt(parea)
noise_alt = tnoise/pfwhm
print(noise,noise_alt)
sns = amps/noise
Ntot = len(amps)
# Set A
a_ras = ras[sns>sncut]
a_decs = decs[sns>sncut]
a_amps = amps[sns>sncut]
# Set B
b_ras = ras[sns<=sncut]
b_decs = decs[sns<=sncut]
b_amps = amps[sns<=sncut]
# Count
Na = len(a_ras)
Nb = len(b_ras)
assert Ntot==(Na+Nb)
# MPI distribute
comm,rank,my_tasks = mpi.distribute(Na,verbose=True)
if rank==0: io.hist(sns,bins=np.linspace(0.,20.,40),save_file=io.dout_dir+"%s_sns.png" % freq)
if rank==0:
    print("Total number of sources: ", Ntot)
    print("Set A: ", Na, " | ", Na*100./Ntot, " % ")
    print("Set B: ", Nb, " | ", Nb*100./Ntot, " % ")
s = stats.Stats(comm)
# Load map
imap = 0.
imap0 = enmap.read_map(paths['planck_files'] + "planck_hybrid_%s_2way_0_map.fits" % freq[1:],sel=np.s_[0,...])
divmap0 = enmap.read_map(paths['planck_files'] + "planck_hybrid_%s_2way_0_ivar.fits" % freq[1:],sel=np.s_[0,...])
imap += imap0*divmap0
del imap0
imap1 = enmap.read_map(paths['planck_files'] + "planck_hybrid_%s_2way_1_map.fits" % freq[1:],sel=np.s_[0,...])
divmap1 = enmap.read_map(paths['planck_files'] + "planck_hybrid_%s_2way_1_ivar.fits" % freq[1:],sel=np.s_[0,...])
imap += imap1*divmap1
del imap1
div = np.nan_to_num(1./(divmap0+divmap1))
del divmap0,divmap1
imap *= div
ps = powspec.read_spectrum("input/cosmo2017_10K_acc3_scalCls.dat") # CHECK
# beam
rs = np.deg2rad(np.linspace(0,90.,10000)) # FIXME
rbeam = maps.gauss_beam_real(rs,pfwhm)
# Loop
rejected = []
for task in my_tasks:
    ra = a_ras[task]
    dec = a_decs[task]
    stamp = reproject.cutout(imap, ra=np.deg2rad(ra), dec=np.deg2rad(dec), pad=1,  npix=npix)
    modlmap = stamp.modlmap()
    n2d = modlmap*0. + (tnoise*np.pi/180./60.)**2.
    if stamp is None: 
        s.add_to_stats("rejected",(task,))
        continue
    famp,cov,pfit = ptfit.ptsrc_fit(stamp,np.deg2rad(dec),np.deg2rad(ra),(rs,rbeam),div=None,ps=ps,beam=pfwhm,n2d=n2d)
    assert cov.size==1
    s.add_to_stats("results",(task,a_amps[task],famp.reshape(-1)[0],cov.reshape(-1)[0]))
    print(famp.reshape(-1)[0],a_amps[task])
    # io.plot_img(stamp)
    # io.plot_img(stamp-pfit)
    
    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))

s.get_stats()

if rank==0:
    results = s.vectors['results']
    np.savetxt("results_%s.txt" % freq,results)
    try: 
        rejected = s.vectors['rejected']
        np.savetxt("rejected_%s.txt" % freq,rejected)
    except:
        pass
