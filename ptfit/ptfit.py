"""
Point source fitting module.
Dependencies: pixell, enlib.pointsrcs, numpy, scipy
"""

from __future__ import print_function
from pixell import enmap
import numpy as np
from scipy.stats import chi2
from scipy.interpolate import interp1d
from enlib import pointsrcs



def ptsrc_fit(imap,dec,ra,rbeam,div=None,ps=None,beam=None,iau=False,
              n2d=None,totp2d=None):
    """Fit a point source to a square stamp.

    Args:
        imap: an ndmap of shape (ncomp,N,N) or (N,N) containing the 
        point source. ncomp can be 1 (assumed intensity) or 3 (assumed
        I,Q,U).
        div: an ndmap of shape (ncomp,N,N) or (N,N) containing the 
        inverse variance in each pixel.
        ra: Right ascension of point source in radians.
        dec: Declination of point source in radians.
        ps: CMB T,E,B power spectrum. Allowed shapes are (nells,) and
        (ncomp,ncomp,nells), starting at ell=0, in compatible units
        as the map.
        beam: The beam in the map specified either as a float (interpreted
        as FWHM in arcminutes) or as an (nells,) array containing the 
        beam transfer function starting at ell=0 or (N,N)  array containing
        the 2D beam transfer function. This Fourier beam is 
        used in the CMB power.
        rbeam: The beam in the map specified as either a float or
        [{r,val},npoint] array  containing the radial beam function. 
        This beam is used in the point source template.
        n2d: Specify the I,Q,U noise (ncomp,ncomp,N,N) 2D power spectrum
        and ignore div.
        totp2d: Specify the total (ncomp,ncomp,N,N) 2D power spectrum
        of I,Q,U and ignore div, n2d, ps and beam.

    Returns:
        pflux: (ncomp,) specifying best fit peak flux
        var: (ncomp,ncomp,) specifying peak flux covarince. 
        Warning: this number depends
        purely either on totp2d or on ps and div, and does not account for 
        other sources of noise present in the map.
        fit: ndmap containing the best fit point source template.


    """
    shape,wcs = imap.shape, imap.wcs
    Ny,N = shape[-2:]
    assert Ny==N
    ncomp = 1 if len(shape)==2 else shape[0]
    assert ncomp==1 or ncomp==3
    template = pointsrcs.sim_srcs(shape[-2:], wcs, np.array(((dec,ra,1),)), rbeam)
    if totp2d is None:
        modlmap = enmap.modlmap(shape,wcs)
        if ps.ndim==1: ps = ps.reshape((1,1,ps.size))
        nells = ps.shape[-1]
        ps = ps[:ncomp,:ncomp,:]
        ells = np.arange(nells)
        cmb2d = np.zeros((ncomp,ncomp,N,N))
        for i in range(ncomp):
            for j in range(i,ncomp):
                cmb2d[i,j] = cmb2d[j,i] = interp(ells,ps[i,j])(modlmap)
        beam = np.asarray(beam)
        if beam.ndim==0: beam2d = gauss_beam(modlmap,beam)
        elif beam.ndim==1:
            nells = beam.size
            ells = np.arange(nells)
            beam2d = maps.interp(ells,beam)(modlmap)
        elif beam.ndim==2: beam2d = beam
        ccov = stamp_pixcov_from_theory(N,
                                        enmap.enmap(cmb2d,wcs),
                                        n2d_IQU=0. if n2d is None else n2d,
                                        beam2d=beam2d,iau=iau,
                                        return_pow=False)
        if n2d is None:
            if div.ndim == 2: div = div[None,...]
            dncomp = div.shape[0]
            ncov = np.zeros((ncomp,ncomp,N*N,N*N))
            if dncomp==1:
                # assuming div is inverse variance in intensity
                # so Q and U should have x 1/2
                ncovI = np.diag(1./div.reshape(-1))
                ncov[0,0] = ncovI
                if ncomp>1: ncov[1,1] = ncov[2,2] = ncovI*2.
            elif dncomp==3:
                assert ncomp==dncomp
                for i in range(dncomp): ncov[i,i] = 1./div[i].reshape(-1)
            else:
                raise ValueError
            ccov += ncov
    else:
        ccov = fcov_to_rcorr(shape,wcs,totp2d,N)
    # --- Make sure that the pcov is in the right order vector(I,Q,U) ---
    # It is currently in (ncomp,ncomp,n,n) order
    # We transpose it to (ncomp,n,ncomp,n) order
    # so that when it is reshaped into a 2D array,
    # a row/column will correspond to an (I,Q,U) vector
    ccov = np.transpose(ccov,(0,2,1,3))
    ccov = ccov.reshape((ncomp*N**2,ncomp*N**2))
    funcs = []
    for i in range(ncomp):
        tzeros = np.zeros((ncomp*N**2,))
        tzeros[i*ncomp*N**2:(i+1)*ncomp*N**2] = template.reshape(-1)
        funcs.append(lambda x: tzeros.copy())
    pflux,cov,_,_ = fit_linear_model(ells,imap.reshape(-1),ccov,funcs=funcs,dofs=None)
    pflux = pflux.reshape((ncomp,))
    cov = cov.reshape((ncomp,ncomp))
    fit = np.zeros((ncomp,N,N))
    for i in range(ncomp): fit[i] = template.copy()*pflux[i]
    return pflux,cov,fit


"""
=============
HELPERS
=============
"""

# duplicated from orphics.maps
def gauss_beam(ell,fwhm):
    tht_fwhm = np.deg2rad(fwhm / 60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))

# duplicated from orphics.maps
def interp(x,y,bounds_error=False,fill_value=0.,**kwargs):
    return interp1d(x,y,bounds_error=bounds_error,fill_value=fill_value,**kwargs)

# duplicated from orphics.maps and orphics.pixcov
def rotate_pol_power(shape,wcs,cov,iau=False,inverse=False):
    """Rotate a 2D power spectrum from TQU to TEB (inverse=False) or
    back (inverse=True). cov is a (3,3,Ny,Nx) 2D power spectrum.
    WARNING: This function is duplicated from orphics.maps to make 
    this module independent. Ideally, it should be implemented in
    enlib.enmap.
    """
    rot = np.zeros((3,3,cov.shape[-2],cov.shape[-1]))
    rot[0,0,:,:] = 1
    prot = enmap.queb_rotmat(enmap.lmap(shape,wcs), inverse=inverse, iau=iau)
    rot[1:,1:,:,:] = prot
    Rt = np.transpose(rot, (1,0,2,3))
    tmp = np.einsum("ab...,bc...->ac...",rot,cov)
    rp2d = np.einsum("ab...,bc...->ac...",tmp,Rt)    
    return rp2d

# duplicated from orphics.pixcov
def stamp_pixcov_from_theory(N,cmb2d_TEB,n2d_IQU=0.,beam2d=1.,iau=False,return_pow=False):
    """Return the pixel covariance for a stamp N pixels across given the 2D IQU CMB power spectrum,
    2D beam template and 2D IQU noise power spectrum.
    """
    n2d = n2d_IQU
    cmb2d = cmb2d_TEB
    assert cmb2d.ndim==4
    ncomp = cmb2d.shape[0]
    assert cmb2d.shape[1]==ncomp
    assert ncomp==3 or ncomp==1
    
    wcs = cmb2d.wcs
    shape = cmb2d.shape[-2:]

    if ncomp==3: cmb2d = rotate_pol_power(shape,wcs,cmb2d,iau=iau,inverse=True)
    p2d = cmb2d*beam2d**2.+n2d
    if not(return_pow): return fcov_to_rcorr(shape,wcs,p2d,N)
    return fcov_to_rcorr(shape,wcs,p2d,N), cmb2d

# duplicated from orphics.pixcov
def fcov_to_rcorr(shape,wcs,p2d,N):
    """Convert a 2D PS into a pix-pix covariance
    """
    ncomp = p2d.shape[0]
    p2d *= np.prod(shape[-2:])/enmap.area(shape,wcs)
    ocorr = enmap.zeros((ncomp,ncomp,N*N,N*N),wcs)
    for i in range(ncomp):
        for j in range(i,ncomp):
            dcorr = ps2d_to_mat(p2d[i,j].copy(), N).reshape((N*N,N*N))
            ocorr[i,j] = dcorr.copy()
            if i!=j: ocorr[j,i] = dcorr.copy()
    return ocorr

# duplicated from orphics.stats
def fit_linear_model(x,y,ycov,funcs,dofs=None,deproject=True):
    """
    Given measurements with known uncertainties, this function fits those to a linear model:
    y = a0*funcs[0](x) + a1*funcs[1](x) + ...
    and returns the best fit coefficients a0,a1,... and their uncertainties as a covariance matrix
    """
    s = solve if deproject else np.linalg.solve
    C = ycov
    y = y[:,None] 
    A = np.zeros((y.size,len(funcs)))
    for i,func in enumerate(funcs):
        A[:,i] = func(x)
    cov = np.linalg.inv(np.dot(A.T,solve(C,A)))
    b = np.dot(A.T,solve(C,y))
    X = np.dot(cov,b)
    YAX = y - np.dot(A,X)
    chisquare = np.dot(YAX.T,solve(C,YAX))
    dofs = len(x)-len(funcs)-1 if dofs is None else dofs
    pte = 1 - chi2.cdf(chisquare, dofs)    
    return X,cov,chisquare/dofs,pte

# duplicated from orphics.stats
class Solver(object):
    """
    Calculate Cinv . x
    """
    def __init__(self,C,u=None):
        """
        C is an (NxN) covariance matrix
        u is an (Nxk) template matrix for rank-k deprojection
        """
        N = C.shape[0]
        if u is None: u = np.ones((N,1))
        Cinvu = np.linalg.solve(C,u)
        self.precalc = np.dot(Cinvu,np.linalg.solve(np.dot(u.T,Cinvu),u.T))
        self.C = C
    def solve(self,x):
        Cinvx = np.linalg.solve(self.C,x)
        correction = np.dot(self.precalc,Cinvx)
        return Cinvx - correction
    
# duplicated from orphics.stats
def solve(C,x,u=None):
    """
    Typically, you do not want to invert your covariance matrix C, but really just want
    Cinv . x , for some vector x.
    You can get that with np.linalg.solve(C,x).
    This function goes one step further and deprojects a common mode for the entire
    covariance matrix, which is often what is needed.
    """
    N = C.shape[0]
    s = Solver(C,u=u)
    return s.solve(x)

### ENMAP HELPER FUNCTIONS AND CLASSES
# These have been copied from enlib.jointmap since that module
# requires compilation (but these functions don't). They are
# also duplicated in orphics.pixcov

def map_ifft(x): return enmap.ifft(x).real
def corrfun_thumb(corr, n):
    tmp = np.roll(np.roll(corr, n, -1)[...,:2*n], n, -2)[...,:2*n,:]
    return np.roll(np.roll(tmp, -n, -1), -n, -2)


def corr_to_mat(corr, n):
    res = enmap.zeros([n,n,n,n],dtype=corr.dtype)
    for i in range(n):
        tmp = np.roll(corr, i, 0)[:n,:]
        for j in range(n):
            res[i,j] = np.roll(tmp, j, 1)[:,:n]
    return res
def ps2d_to_mat(ps2d, n):
    corrfun = map_ifft(ps2d+0j)/(ps2d.shape[-2]*ps2d.shape[-1])**0.5
    thumb   = corrfun_thumb(corrfun, n)
    mat     = corr_to_mat(thumb, n)
    return mat


