import numpy as np

def compute_shape_noise_error(sources, mask, sigma_g=0.26, use_weights=False):
    """
    Compute the shape noise error for a given bin.

    Args:
    - sources (DataFrame): Source catalogue DataFrame containing galaxy data.
    - mask (ndarray): Boolean array indicating which sources are in the current bin.
    - sigma_g (float, optional): The intrinsic shape noise per shear component. Defaults to 0.26.
    - use_weights (bool, optional): Whether to use weights in the calculation. Defaults to False.

    Returns:
    - shape_noise_error (float): The shape noise error for the bin.
    """
    if np.sum(mask) == 0:
        return 0.0  # No galaxies in the bin, no error

    # Select sources in the bin
    galaxies_in_bin = sources[mask]

    if use_weights:
        w = galaxies_in_bin['weight']
    else:
        w = np.ones(len(galaxies_in_bin))  # If no weights, set weights to 1

    # Calculate shape noise variance per bin
    numerator = np.sum(w ** 2 * sigma_g ** 2)
    denominator = np.sum(w) ** 2

    shape_noise_variance = numerator / denominator
    shape_noise_error = np.sqrt(shape_noise_variance)

    return shape_noise_error

def get_lss_cov(random_shear_profiles, n_bins):
    """
    Compute the covariance matrix from random shear profiles.

    Args:
    - random_shear_profiles (Table): Astropy Table containing the random shear profiles with columns:
        'RA', 'Dec', 'rin', 'rout', 'gplus', 'errors', 'msci', 'fl'.
    - n_bins (int): Number of radial bins used in the shear profile.

    Returns:
    - covariance_matrix (ndarray): Covariance matrix of shape (n_bins, n_bins).
    """

    # Extract the tangential shear profiles (gplus) from the random shear profiles
    gplus_profiles = []
    for i in range(len(random_shear_profiles) // n_bins):
        gplus = random_shear_profiles['gplus'][i * n_bins: (i + 1) * n_bins]
        gplus_profiles.append(gplus)

    gplus_profiles = np.array(gplus_profiles)  # Shape: (n_random, n_bins)

    # Compute the mean shear profile
    mean_gplus = np.mean(gplus_profiles, axis=0)  # Shape: (n_bins,)

    # Initialize the covariance matrix
    covariance_matrix = np.zeros((n_bins, n_bins))

    # Compute the covariance matrix
    n_random = gplus_profiles.shape[0]
    for i in range(n_bins):
        for j in range(n_bins):
            covariance_matrix[i, j] = (
                    np.sum((gplus_profiles[:, i] - mean_gplus[i]) * (gplus_profiles[:, j] - mean_gplus[j])) / (
                        n_random - 1)
            )

    return covariance_matrix