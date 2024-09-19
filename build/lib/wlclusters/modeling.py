import numpy as np
import astropy.units as u
from .deproject import MyDeprojVol
import pymc as pm



def rho_nfw_cr(radii, pmod, parnames=['cdelt', 'rdelt'], delta=200., rho_crit=None):
    """
    PyMC (Theano) function for the Navarro-Frenk-White (NFW) density profile (Navarro et al. 1996).

    Args:
    - radii: Array of radial distances in Mpc.
    - pmod: Parameter model, containing values for concentration and radius/mass in the order defined by parnames.
    - parnames: List defining the parameter names; possible values are ['cdelt', 'rdelt'], ['cdelt', 'mdelt'],
                ['log10cdelt', 'log10mdelt'], ['cdelt', 'log10mdelt']. Default is ['cdelt', 'rdelt'].
    - delta: Chosen overdensity, default is 200.
    - rho_crit: Critical density of the universe at the cluster's redshift (required for mdelt to rdelt conversion).

    Returns:
    - NFW density profile divided by the critical density of the Universe (multiply by rho_crit to get physical density).
    """

    # Calculate r as the midpoints of radii
    r = (radii[1:] + radii[:-1]) / 2 * 1000.  # Convert radii to kpc

    # Extract parameters based on parnames
    if parnames == ['cdelt', 'rdelt']:
        cdelt, rdelt = pmod
    elif parnames == ['cdelt', 'mdelt']:
        cdelt, mdelt = pmod
        # Convert mdelt to rdelt using the relation between mass and radius
        rdelt = (3 * mdelt / (4 * np.pi * delta * rho_crit)) ** (1 / 3)
    elif parnames == ['log10cdelt', 'log10mdelt']:
        log10cdelt, log10mdelt = pmod
        cdelt = 10**log10cdelt
        mdelt = 10**log10mdelt
        rdelt = (3 * mdelt / (4 * np.pi * delta * rho_crit))**(1/3)
    elif parnames == ['cdelt', 'log10mdelt']:
        cdelt, log10mdelt = pmod
        mdelt = 10**log10mdelt
        rdelt =(3 * mdelt / (4 * np.pi * delta * rho_crit))**(1/3)
    else:
        raise ValueError("Invalid parnames specified.")

    # Calculate delta_crit using PyMC math functions
    delta_crit = (delta / 3) * (cdelt**3) * (pm.math.log(1. + cdelt) - cdelt / (1 + cdelt)) ** (-1)

    # Return NFW density profile
    return delta_crit / ((cdelt * r / rdelt) * ((1. + (cdelt * r / rdelt)) ** 2))


def rho_to_sigma(radii_bins, rho):

    """
   Recovers the projected surface mass density by projecting a density profile

    Args:
    - radii_bins
    - rho: density profile

    Returns:
    - surface mass density in M_sun * Mpc**-2
    """

    deproj = MyDeprojVol(radii_bins[:-1], radii_bins[1:])
    proj_vol = deproj.deproj_vol().T
    area_proj = np.pi * (-(radii_bins[:-1] * 1e6) ** 2 + (radii_bins[1:] * 1e6) ** 2)
    sigma = pm.math.dot(proj_vol, rho) / area_proj
    return sigma * 1e12


def dsigma_trap(sigma, radii):
    """
   computes Delta Sigma using numerical trapezoidal intergration

    Args:
    - sigma: projected surface mass density
    - radii

    Returns:
    - Delta Sigma
    """

    rmean = (radii[1:] + radii[:-1]) / 2
    rmean2 = (rmean[1:] + rmean[:-1]) / 2
    m = np.tril(np.ones((len(rmean2) + 1, len(rmean2) + 1)))
    dr = rmean[1:] - rmean[:-1]

    ndr = len(dr)

    arg0 = sigma[0] * (rmean2[0] ** 2) / 2
    arg1 = dr * (sigma[1:] * rmean[1:] + sigma[:-1] * rmean[:-1]) / 2

    list_stack = [arg0]

    for i in range(ndr):
        list_stack.append(arg1[i])

    arg = pm.math.stack(list_stack)
    a = pm.math.dot(m, arg)
    sigmabar = (2 / (rmean ** 2)) * a
    dsigma = sigmabar - sigma
    return dsigma


def get_shear(sigma, dsigma, mean_sigm_crit_inv, fl):
    """
   computes the expected tangential shear profile using Seitz and Schneider 1997 (or Umestu 2020 equation 93)

    Args:
    - sigma: projected surface mass density
    - dsigma: Delta Sigma
    - mean_sigm_crit_inv: value of the inverse mean critical density <sigcrit**-1> in Mpc**2.Msun**-1
    - fl: value of <sigcrit**-2> / (<sigcrit**-1>**2)

    Returns:
    - mean tangential shear
    """

    shear = (dsigma * mean_sigm_crit_inv) / (1 - fl * sigma * mean_sigm_crit_inv)

    return shear


def get_radplus(radii, rmin=1e-3, rmax=1e2, nptplus=19):
    """
   for the numericat integration to be successful, it is useful to create a set of fictive points at low radii aswell
   as between each data point

    Args:
    - radii: input radii of the data points
    - dsigma: Delta Sigma
    - mean_sigm_crit_inv: value of the inverse mean critical density <sigcrit**-1> in Mpc**2.Msun**-1
    - fl: value of <sigcrit**-2> / (<sigcrit**-1>**2)

    Returns:
    - radplus: new set of extra/inter-polated radii values, including the values of "radii"
    - rmeanplus: central values of "radplus"
    - evalrad: indices of the "radii" points within "radplus"
    """

    if nptplus % 2 == 0:
        nptplus = nptplus + 1
    rmean = (radii[1:] + radii[:-1]) / 2.
    radplus = np.logspace(np.log10(rmin), np.log10(radii[0]), nptplus)
    for i in range(len(radii) - 1):
        vplus = np.linspace(radii[i], radii[i + 1], nptplus + 1)
        radplus = np.append(radplus, vplus[1:])
    radplus = np.append(radplus, np.logspace(np.log10(radplus[-1]), np.log10(rmax), 20)[1:])
    rmeanplus = (radplus[1:] + radplus[:-1]) / 2.
    nsym = int(np.floor(nptplus / 2))
    evalrad = (np.arange(nptplus + nsym - 1, nptplus + nsym + len(rmean) * nptplus, nptplus))[:len(rmean)]
    return radplus, rmeanplus, evalrad


def WLmodel(WLdata, pmod):
    """
   Modeling of the mean tangential shear for a given density profile at a given redshift

    Args:
    - WLdata: class containing all the useful informations about the cluster (radii binning, redshift, msci, fl..)
    - pmod: list containing the density profile parameters values, in the case NFW, cdelta and rdelta

    Returns:
    - gplus: predicted mean tangential shear profile
    - rm: radii bins after inter/extra-polation by the function get_radplus
    - ev: indices of the input data radii points within rm, i.e. the radii binning given in input is rm[ev]
    """
    
    radplus, rm, ev = get_radplus(WLdata.radii_wl)

    rho_out = rho_nfw_cr(radplus, pmod) * WLdata.rho_crit

    sig = rho_to_sigma(radplus, rho_out)

    dsigma = dsigma_trap(sig, radplus)

    gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)

    return gplus, rm, ev



"""
The following functions, of which the names end with "*_np" are the numpy equivalent of the theano functions above.
They are useful for plotting, or when it is needed to evaluate the theano functions.
"""
def rho_nfw_cr_np(radii, c200, r200, delta=200.):
    r = (radii[1:] + radii[:-1]) / 2 * 1000.
    delta_crit = (delta / 3) * (c200 ** 3) * (np.log(1. + c200) - c200 / (1 + c200)) ** (-1)
    return delta_crit / ((c200 * r / r200) * ((1. + (c200 * r / r200)) ** 2))


def rho_to_sigma_np(radii_bins, rho):
    deproj = MyDeprojVol(radii_bins[:-1], radii_bins[1:])
    proj_vol = deproj.deproj_vol().T
    area_proj = np.pi * (-(radii_bins[:-1] * 1e6) ** 2 + (radii_bins[1:] * 1e6) ** 2)
    sigma = np.dot(proj_vol, rho) / area_proj
    return sigma * 1e12


def dsigma_trap_np(sigma, radii):
    rmean = (radii[1:] + radii[:-1]) / 2
    rmean2 = (rmean[1:] + rmean[:-1]) / 2
    m = np.tril(np.ones((len(rmean2) + 1, len(rmean2) + 1)))
    dr = rmean[1:] - rmean[:-1]

    ndr = len(dr)

    arg0 = sigma[0] * (rmean2[0] ** 2) / 2
    arg1 = dr * (sigma[1:] * rmean[1:] + sigma[:-1] * rmean[:-1]) / 2

    arg = np.append(arg0, arg1)

    a = np.dot(m, arg)
    sigmabar = (2 / (rmean ** 2)) * a
    dsigma = sigmabar - sigma
    return dsigma


def WLmodel_np(WLdata, pmod):
    radplus, rm, ev = get_radplus(WLdata.radii_wl)
    rho_out = rho_nfw_cr_np(radplus, pmod) * WLdata.rho_crit
    sig = rho_to_sigma_np(radplus, rho_out)
    dsigma = dsigma_trap_np(sig, radplus)
    gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)
    return gplus, rm, ev

class WLData:
    '''
    This class is initialized for each cluster in the catalog
    :type cosmo: astropy.cosmology
    '''
    def __init__(self, redshift, rin=None, rout=None, gplus=None, err_gplus=None,
                 sigmacrit_inv=None, fl=None, cosmo=None, unit='proper'):

        if rin is None or rout is None or gplus is None or err_gplus is None:

            print('Missing input, please provide rin, rout, gplus, and err_gplus')

            return

        if sigmacrit_inv is None:

            print('The mean value of sigma_crit is required')

            return

        if fl is None:

            print('The second order correction factor is not given, we will do the calculation at first order')

        self.gplus = gplus

        self.err_gplus = err_gplus

        if cosmo is None:

            from astropy.cosmology import Planck15 as cosmo

        if unit == 'proper':
            amin2kpc = cosmo.kpc_proper_per_arcmin(redshift).value
        if unit == 'comoving':
            amin2kpc = cosmo.kpc_comoving_per_arcmin(redshift).value

        self.rin_wl = rin * amin2kpc / 1e3 # Mpc

        self.rout_wl = rout * amin2kpc / 1e3

        self.radii_wl = np.append(self.rin_wl[0], self.rout_wl)

        self.rin_wl_am = rin

        self.rout_wl_am = rout

        self.rref_wl = (self.rin_wl + self.rout_wl) / 2.

        self.rho_crit = (cosmo.critical_density(redshift).to(u.M_sun * u.Mpc**-3)).value

        self.msigmacrit = sigmacrit_inv

        self.fl = fl
