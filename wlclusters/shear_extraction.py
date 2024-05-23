import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
import astropy.constants as c
from tqdm import tqdm

def compute_tangential_shear_profile(sources, center, z_cl, bin_edges, dz, cosmo):
    """
    Compute the tangential shear profile around a cluster center.

    Args:
    - sources (DataFrame): Source catalogue DataFrame containing columns for 'RA', 'Dec', 'e_1', and 'e_2'.
    - center (list): List containing the RA and Dec coordinates of the cluster center in deg.
    - z_cl (float): Redshift of the cluster.
    - bin_edges (array-like): Array containing the bin edges for radial profile calculation in Mpc.
    - dz (float, optional): Redshift offset for source selection. Defaults to 0.1.

    Returns:
    - bin_edges_deg (ndarray): Array of bin edges in deg.
    - bin_mean (ndarray): Array of mean bin values in deg.
    - signal (ndarray): Array of shear signal values.
    - errors (ndarray): Array of errors associated with each bin.
    """
    x, y = sources['RA'] - center[0], sources['Dec'] - center[1]
    theta = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    g1 = sources['e_1']
    g2 = -sources['e_2']
    gamma_plus = -g1 * np.cos(2 * phi) - g2 * np.sin(2 * phi)

    filtered_sources = sources[(sources['z_p'] >= z_cl + dz) & (theta <= max(bin_edges))]
    kpcp = cosmo.kpc_proper_per_arcmin(z_cl).value

    nbins = len(bin_edges) - 1
    bin_edges_deg = (bin_edges * 1000) / (kpcp * 60)
    bin_mean = (bin_edges_deg[:-1] + bin_edges_deg[1:]) / 2
    signal = np.zeros(nbins)
    bin_count = np.zeros(nbins)
    variance = np.zeros(nbins)
    errors = np.zeros(nbins)

    for i in range(nbins):
        mask = np.logical_and(theta >= bin_edges_deg[i], theta < bin_edges_deg[i + 1])
        bin_count[i] = np.sum(mask)
        if bin_count[i] > 0:
            signal[i] = np.sum(gamma_plus[mask]) / bin_count[i]
        else:
            signal[i] = 0

        if bin_count[i] > 1:
            pixels_in_bin = np.where(mask)
            pixel_values = gamma_plus[pixels_in_bin]
            variance[i] = np.var(pixel_values, ddof=1)
            errors[i] = np.sqrt(variance[i] / bin_count[i])
        else:
            variance[i] = 0
            errors[i] = 0

    return bin_edges_deg, bin_mean, signal, errors


def return_sigmacrit(sources, center, z_cl, bin_edges, dz, cosmo):
    """
    Compute the inverse mean critical density over the whole radial range around the cluster.

    Args:
    - sources (DataFrame): Source catalogue DataFrame containing columns for 'RA', 'Dec', 'e_1', and 'e_2'.
    - center (list): List containing the RA and Dec coordinates of the cluster center in deg.
    - z_cl (float): Redshift of the cluster.
    - bin_edges (array-like): Array containing the bin edges for radial profile calculation in Mpc.
    - dz (float, optional): Redshift offset for source selection. Defaults to 0.1.

    Returns:
    - mean_sigm_crit_inv (float): value of the inverse mean critical density <sigcrit**-1> in Mpc**2.Msun**-1.
    - fl (float): value of <sigcrit**-2> / (<sigcrit**-1>**2), useful for 2nd order approximation of the shear.
    """

    kpcp = cosmo.kpc_proper_per_arcmin(z_cl).value

    bin_edges_deg = (bin_edges * 1000) / (kpcp * 60)

    binmin, binmax = min(bin_edges_deg), max(bin_edges_deg)

    sources_zcut = sources[sources['z_p'] >= z_cl + dz]

    theta = np.sqrt((sources_zcut['RA'] - center[0]) ** 2 + (sources_zcut['Dec'] - center[1]) ** 2)

    mask = np.where(np.logical_and(theta <= binmax, theta >= binmin))

    zs = sources_zcut['z_p'][mask]

    c_mpc = c.c.to(u.Mpc / u.s)

    g_mpc = c.G.to(u.Mpc ** 3 / (u.kg * u.s ** 2))

    prefactor_mpc = c_mpc ** 2 / (4 * np.pi * g_mpc)

    dl = cosmo.angular_diameter_distance(z_cl)

    ds = cosmo.angular_diameter_distance(zs)

    dls = cosmo.angular_diameter_distance_z1z2(z_cl, zs)

    sigma_crit_mpc = prefactor_mpc * ds / (dl * dls)

    mean_sigm_crit_inv = float(np.mean(1 / (sigma_crit_mpc).to(u.M_sun / u.Mpc ** 2)).value)

    fl = float(np.mean((sigma_crit_mpc).to(u.M_sun / u.Mpc ** 2) ** -2) / (
        np.mean((sigma_crit_mpc).to(u.M_sun / u.Mpc ** 2) ** -1)) ** 2)

    return mean_sigm_crit_inv, fl


def shear_extraction(cluster_cat, sources, bin_edges, dz, cosmo):
    profiles = []
    for cluster in tqdm(cluster_cat):
        clust_center = [cluster['RA'], cluster['Dec']]
        clust_z = cluster['z_p']
        bin_edges_deg, bin_mean, signal, errors = compute_tangential_shear_profile(
            sources, clust_center, clust_z, bin_edges, dz=0.1, cosmo=cosmo)
        msci, fl = return_sigmacrit(sources, clust_center, clust_z, bin_edges, dz=dz, cosmo=cosmo)

        profile = Table()
        profile['ID'] = [cluster['ID']] * len(bin_mean)
        profile['rin'] = bin_edges_deg[:-1]*60
        profile['rout'] = bin_edges_deg[1:]*60
        profile['gplus'] = signal
        profile['errors'] = errors
        profile['msci'] = [msci] * len(bin_mean)
        profile['fl'] = [fl] * len(bin_mean)
        profiles.append(profile)

    shear_profiles = profiles[0]  # Start with the first profile
    for profile in profiles[1:]:
        shear_profiles = vstack([shear_profiles, profile])

    return shear_profiles