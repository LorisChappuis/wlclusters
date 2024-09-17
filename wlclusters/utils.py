import numpy as np
def find_closest_redshift(z, z_arr):
    return z_arr[np.abs(z_arr - z).argmin()]

