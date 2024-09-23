import numpy as np
import astropy.units as u


def find_closest_redshift(z, z_arr):
    """
    Find the closest redshift to a target value in a given array of redshifts.

    Args:
        z (float): The target redshift value.
        z_arr (numpy.ndarray): An array of redshift values to search from.

    Returns:
        float: The redshift value in `z_arr` that is closest to the target redshift `z`.
    """
    return z_arr[np.abs(z_arr - z).argmin()]


def rdelt_to_mdelt(r, z, cosmo, delta=200):
    """
    Convert radius `r_delta` to mass `m_delta` for a given redshift and cosmology.

    Args:
        r (float): The radius `r_delta` in kpc.
        z (float): The redshift of the cluster.
        cosmo (astropy.cosmology.Cosmology): Cosmology object used for calculations (e.g., Planck15).
        delta (float, optional): Overdensity factor (default is 200, corresponding to `r200`).

    Returns:
        float: The mass `m_delta` corresponding to the given radius `r_delta` at redshift `z` and overdensity `delta`.
    """
    rhoc = cosmo.critical_density(z).to(u.M_sun * u.kpc**-3).value
    return (4/3) * np.pi * delta * rhoc * r**3

def mdelt_to_rdelt(m, z, cosmo, delta=200):
    """
    Convert mass `m_delta` to radius `r_delta` for a given redshift and cosmology.

    Args:
        m (float): The mass `m_delta` in solar masses.
        z (float): The redshift of the cluster.
        cosmo (astropy.cosmology.Cosmology): Cosmology object used for calculations (e.g., Planck15).
        delta (float, optional): Overdensity factor (default is 200, corresponding to `m200`).

    Returns:
        float: The radius `r_delta` corresponding to the given mass `m_delta` at redshift `z` and overdensity `delta`.
    """
    rhoc = cosmo.critical_density(z).to(u.M_sun * u.kpc**-3).value
    return (m / ((4/3) * np.pi * delta * rhoc))**(1/3)

