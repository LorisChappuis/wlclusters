import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
import astropy.constants as c
from tqdm import tqdm
from astropy.coordinates import SkyCoord

from .lss import compute_shape_noise_error, get_lss_cov


def compute_tangential_shear_profile(sources, center, z_cl, bin_edges, dz, cosmo, unit='proper', sigma_g=0.26):
    """
    Compute the tangential shear profile around a cluster center, accounting for responsivity (R) in each bin.

    Args:
    - sources (DataFrame): Source catalogue DataFrame containing columns for 'RA', 'Dec', 'e_1', 'e_2', and
                            optionally the weights, multiplicative bias, additive biases, and RMS ellipticity (e_rms).
    - center (list): List containing the RA and Dec coordinates of the cluster center in degrees.
    - z_cl (float): Redshift of the cluster.
    - bin_edges (array-like): Array containing the bin edges for radial profile calculation in Mpc.
    - dz (float, optional): Redshift offset for source selection. Defaults to 0.1.
    - cosmo (Cosmology): Cosmology object for distance calculations.
    - unit (str): Unit for distance calculation ('proper' or 'comoving'). Defaults to 'proper'.
    - sigma_g (float, optional): The intrinsic shape noise per shear component. Defaults to 0.26.

    Returns:
    - bin_edges_deg (ndarray): Array of bin edges in degrees.
    - bin_mean (ndarray): Array of mean bin values in degrees.
    - signal (ndarray): Array of shear signal values.
    - errors (ndarray): Array of errors associated with each bin (including shape noise).
    """

    if 'z_p' not in sources.columns:
        raise ValueError("The 'z_p' column is missing in the sources DataFrame.")

    # Source selection based on redshift
    sources = sources[sources['z_p'] >= z_cl + dz]
    x, y = sources['RA'] - center[0], sources['Dec'] - center[1]
    theta = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    g1 = sources['e_1']
    g2 = -sources['e_2']

    # Check for optional columns
    use_response = 'e_rms' in sources.columns
    use_weights = 'weight' in sources.columns
    use_multiplicative_bias = 'm_bias' in sources.columns
    use_additive_bias = 'c_1_bias' in sources.columns and 'c_2_bias' in sources.columns

    if use_weights:
        w = sources['weight']
    else:
        w = np.ones(len(sources))  # Assume weights are 1 if not provided

    if use_multiplicative_bias:
        m = sources['m_bias']
    else:
        m = np.zeros(len(sources))  # Assume no multiplicative bias if not provided

    if use_additive_bias:
        c1 = sources['c_1_bias']
        c2 = sources['c_2_bias']
        g1 = g1 - c1
        g2 = g2 - c2

    gamma_plus = -g1 * np.cos(2 * phi) - g2 * np.sin(2 * phi)

    if unit == 'proper':
        kpcp = cosmo.kpc_proper_per_arcmin(z_cl).value
    elif unit == 'comoving':
        kpcp = cosmo.kpc_comoving_per_arcmin(z_cl).value
    else:
        raise ValueError("Unit must be 'proper' or 'comoving'.")

    nbins = len(bin_edges) - 1
    bin_edges_deg = (bin_edges * 1000) / (kpcp * 60)
    bin_mean = (bin_edges_deg[:-1] + bin_edges_deg[1:]) / 2
    signal = np.zeros(nbins)
    errors = np.zeros(nbins)

    # Loop through bins and compute shear, errors, and responsivity
    for i in range(nbins):
        mask = np.logical_and(theta >= bin_edges_deg[i], theta < bin_edges_deg[i + 1])

        if np.sum(mask) > 0:
            # Calculate responsivity R(Ri) for this bin
            if use_response:
                e_rms = sources['e_rms'][mask]
                R_i = 1 - np.sum(w[mask] * e_rms ** 2) / np.sum(w[mask])
            else:
                R_i = 0.5  # Assume 2*R=1 if not provided

            # Compute weighted shear including multiplicative bias and responsivity
            weighted_shear = w[mask] * gamma_plus[mask] / (2 * R_i * (1 + m[mask]))
            signal[i] = np.sum(weighted_shear) / np.sum(w[mask])

            # Compute the shape noise for this bin
            errors[i] = compute_shape_noise_error(sources, mask, sigma_g=sigma_g, use_weights=use_weights)
        else:
            signal[i] = 0
            errors[i] = 0

    return bin_edges_deg, bin_mean, signal, errors




def return_sigmacrit(sources, center, z_cl, bin_edges, dz, cosmo, unit='proper'):
    """
    Compute the inverse mean critical density over the whole radial range around the cluster,
    taking into account weights.

    Args:
    - sources (DataFrame): Source catalogue DataFrame containing columns for 'RA', 'Dec', 'e_1', 'e_2', and 'weight'.
    - center (list): List containing the RA and Dec coordinates of the cluster center in deg.
    - z_cl (float): Redshift of the cluster.
    - bin_edges (array-like): Array containing the bin edges for radial profile calculation in Mpc.
    - dz (float, optional): Redshift offset for source selection. Defaults to 0.1.

    Returns:
    - mean_sigm_crit_inv (float): value of the inverse mean critical density <sigcrit**-1> in Mpc**2.Msun**-1.
    - fl (float): value of <sigcrit**-2> / (<sigcrit**-1>**2), useful for 2nd order approximation of the shear.
    """

    if unit == 'proper':
        kpcp = cosmo.kpc_proper_per_arcmin(z_cl).value
    elif unit == 'comoving':
        kpcp = cosmo.kpc_comoving_per_arcmin(z_cl).value
    else:
        raise ValueError("Unit must be 'proper' or 'comoving'.")

    bin_edges_deg = (bin_edges * 1000) / (kpcp * 60)

    binmin, binmax = min(bin_edges_deg), max(bin_edges_deg)

    sources_zcut = sources[sources['z_p'] >= z_cl + dz]

    theta = np.sqrt((sources_zcut['RA'] - center[0]) ** 2 + (sources_zcut['Dec'] - center[1]) ** 2)
    mask = np.logical_and(theta <= binmax, theta >= binmin)

    zs = sources_zcut['z_p'][mask]
    w = sources_zcut['weight'][mask] if 'weight' in sources_zcut.columns else np.ones(len(zs))

    c_mpc = c.c.to(u.Mpc / u.s)
    g_mpc = c.G.to(u.Mpc ** 3 / (u.kg * u.s ** 2))
    prefactor_mpc = c_mpc ** 2 / (4 * np.pi * g_mpc)

    dl = cosmo.angular_diameter_distance(z_cl)
    ds = cosmo.angular_diameter_distance(zs)
    dls = cosmo.angular_diameter_distance_z1z2(z_cl, zs)

    sigma_crit_mpc = (prefactor_mpc * ds / (dl * dls)).to(u.M_sun / u.Mpc ** 2)

    # Weighted mean of the inverse critical density
    mean_sigm_crit_inv = np.sum(w * sigma_crit_mpc**-1) / np.sum(w)

    # Calculate the second order term f_l
    fl_num =  np.sum(w * sigma_crit_mpc**-2)/np.sum(w)
    fl_dom =  mean_sigm_crit_inv**2
    fl = fl_num/fl_dom

    return mean_sigm_crit_inv.value, fl.value


def shear_extraction(cluster_cat, sources, bin_edges, dz, cosmo, unit='proper', sources_denoised=None, lss=False, security_distance=1000):
    """
    Iterates the shear extraction over all clusters of the given catalog.
    Optionally computes the Large Scale Structure (LSS) covariance matrices.

    Args:
    - cluster_cat (DataFrame): Galaxy cluster catalog DataFrame containing columns for:
        'RA', 'Dec'(position of the cluster)
        'ID', (unique ID of the cluster)
        'z_p' (redshift)
    - sources (DataFrame): Source catalogue DataFrame containing columns for 'RA', 'Dec', 'e_1', and 'e_2'
    - bin_edges (array-like): Array containing the bin edges for radial profile calculation in Mpc.
    - dz (float, optional): Redshift offset for source selection. Defaults to 0.1.
    - cosmo: cosmology initialized with astropy.cosmology
    - unit (str, optional): Unit for distance calculation ('proper' or 'comoving'). Defaults to 'proper'.
    - lss (bool, optional): Whether to compute the LSS covariance matrices. Defaults to False.
    - security_distance (float, optional): Minimum distance from any cluster in kpc for LSS computation.

    Returns:
    - shear_profiles: an astropy table containing the columns:
        'ID': (unique ID of the cluster),
        'rin', 'rout': the edges of each concentric bin in which the extraction was done, in arcmins,
        'gplus': the mean tangential shear,
        'errors': incertitude on the mean tangential shear,
        'msci': value of the inverse mean critical density <sigcrit**-1> in Mpc**2.Msun**-1.
        'fl': value of <sigcrit**-2> / (<sigcrit**-1>**2), useful for 2nd order approximation of the shear.
    - covariance_table: (optional) Astropy table containing the covariance matrices for each cluster, if lss=True.
    """
    profiles = []
    covariance_matrices = []

    # Wrap the loop over clusters with tqdm
    for cluster in tqdm(cluster_cat, desc="Processing Clusters"):
        clust_center = [cluster['RA'], cluster['Dec']]
        clust_z = cluster['z_p']

        # Compute the tangential shear profile
        bin_edges_deg, bin_mean, signal, errors = compute_tangential_shear_profile(
            sources, clust_center, clust_z, bin_edges, dz=0.1, cosmo=cosmo, unit=unit)

        # Compute the mean inverse critical density and fl
        msci, fl = return_sigmacrit(sources, clust_center, clust_z, bin_edges, dz=dz, cosmo=cosmo, unit=unit)

        # Store the shear profile in a table
        profile = Table()
        profile['ID'] = [cluster['ID']] * len(bin_mean)
        profile['rin'] = bin_edges_deg[:-1] * 60
        profile['rout'] = bin_edges_deg[1:] * 60
        profile['gplus'] = signal
        profile['errors'] = errors
        profile['msci'] = [msci] * len(bin_mean)
        profile['fl'] = [fl] * len(bin_mean)
        profiles.append(profile)

        # If LSS covariance matrix calculation is requested
        if lss:
            # Use denoised sources for LSS covariance matrix computation
            random_shear_profiles = extract_random_shear_profiles(
                clust_z, cluster_cat, sources_denoised, bin_edges, dz, cosmo, unit=unit, security_distance=security_distance
            )
            n_bins = len(bin_edges) - 1
            covariance_matrix = get_lss_cov(random_shear_profiles, n_bins)
            covariance_matrices.append([cluster['ID'], covariance_matrix])

    shear_profiles = vstack(profiles)

    if lss:
        # Create an Astropy table for the covariance matrices
        covariance_table = Table(rows=covariance_matrices, names=['ID', 'covariance_matrix'])
        return shear_profiles, covariance_table

    return shear_profiles


def extract_random_shear_profiles(z_cl, cluster_cat, sources, bin_edges, dz, cosmo, n_random=100, security_distance=500, unit='proper'):
    """
    Extract random shear profiles to estimate LSS covariance.

    Args:
    - cluster_cat (DataFrame): Galaxy cluster catalog containing columns for 'RA', 'Dec', and 'z_p'.
    - sources (DataFrame): Source catalogue with columns for 'RA', 'Dec', 'e_1', 'e_2', etc.
    - bin_edges (array-like): Array containing the bin edges for radial profile calculation in Mpc.
    - dz (float): Redshift offset for source selection.
    - cosmo: Cosmology object for distance calculations.
    - n_random (int): Number of random points to extract shear profiles from. Defaults to 100.
    - security_distance (float): Minimum distance in kpc from known clusters to avoid. Defaults to 500 kpc.
    - unit (str): Unit for distance calculation ('proper' or 'comoving'). Defaults to 'proper'.

    Returns:
    - random_shear_profiles: Astropy Table with random shear profiles.
    """

    random_profiles = []

    # Convert security distance to degrees
    if unit == 'proper':
        kpcp = cosmo.kpc_proper_per_arcmin(z_cl).value
    elif unit == 'comoving':
        kpcp = cosmo.kpc_comoving_per_arcmin(z_cl).value
    security_distance_deg = security_distance / (kpcp * 60)
    edge_distance = np.max(bin_edges) / (kpcp * 60)

    # Create a SkyCoord object for cluster centers
    cluster_coords = SkyCoord(ra=cluster_cat['RA']*u.deg, dec=cluster_cat['Dec']*u.deg)

    # Generate random points
    for _ in range(n_random):
        valid_point_found = False
        while not valid_point_found:
            # Generate a random point within the survey area
            rand_ra = np.random.uniform(np.min(sources['RA']+security_distance_deg), np.max(sources['RA'])-security_distance_deg)
            rand_dec = np.random.uniform(np.min(sources['Dec']+security_distance_deg), np.max(sources['Dec'])-security_distance_deg)
            random_point = SkyCoord(ra=rand_ra*u.deg, dec=rand_dec*u.deg)

            # Check if the random point is far enough from all known clusters
            sep = random_point.separation(cluster_coords).deg
            if np.all(sep > security_distance_deg):
                valid_point_found = True

        # Compute shear profile at the valid random point
        bin_edges_deg, bin_mean, signal, errors = compute_tangential_shear_profile(
            sources, [rand_ra, rand_dec], z_cl, bin_edges, dz=dz, cosmo=cosmo, unit=unit
        )
        msci, fl = return_sigmacrit(sources, [rand_ra, rand_dec], z_cl, bin_edges, dz=dz, cosmo=cosmo, unit=unit)

        # Save the profile
        profile = Table()
        profile['RA'] = [rand_ra] * len(bin_mean)
        profile['Dec'] = [rand_dec] * len(bin_mean)
        profile['rin'] = bin_edges_deg[:-1]*60
        profile['rout'] = bin_edges_deg[1:]*60
        profile['gplus'] = signal
        profile['errors'] = errors
        profile['msci'] = [msci] * len(bin_mean)
        profile['fl'] = [fl] * len(bin_mean)
        random_profiles.append(profile)

    # Combine all profiles into a single table
    random_shear_profiles = random_profiles[0]
    for profile in random_profiles[1:]:
        random_shear_profiles = vstack([random_shear_profiles, profile])

    return random_shear_profiles


def get_lss_cov_for_z(z_arr, cluster_cat, sources_denoised, bin_edges, dz, cosmo, n_random=100, unit='proper', security_distance=0.):
    # Use denoised sources for LSS covariance matrix computation
    covariance_matrices = []

    for z in tqdm(z_arr):
        random_shear_profiles = extract_random_shear_profiles(z, cluster_cat, sources_denoised, bin_edges, dz, cosmo, n_random=n_random,
                                      security_distance=security_distance, unit=unit)
        n_bins = len(bin_edges) - 1
        covariance_matrix = get_lss_cov(random_shear_profiles, n_bins)
        covariance_matrices.append([z, covariance_matrix])
    covariance_table = Table(rows=covariance_matrices, names=['z', 'covariance_matrix'])
    return  covariance_table