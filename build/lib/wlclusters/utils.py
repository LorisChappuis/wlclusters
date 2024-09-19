import numpy as np
import astropy.units as u


def find_closest_redshift(z, z_arr):
    return z_arr[np.abs(z_arr - z).argmin()]


def rdelt_to_mdelt(r, z, cosmo, delta=200):
    rhoc = cosmo.critical_density(z).to(u.M_sun * u.kpc**-3).value
    return (4/3) * np.pi * delta * rhoc * r**3

def mdelt_to_rdelt(m, z, cosmo, delta=200):
    rhoc = cosmo.critical_density(z).to(u.M_sun * u.kpc**-3).value
    return (m / ((4/3) * np.pi * delta * rhoc))**(1/3)

