import numpy as np
from tqdm import tqdm
from .modeling import WLData, WLmodel_np
import random

def wldata_from_ID(lens_id,
                   cluster_cat,
                   shear_profiles,
                   results,
                   all_chains=None,
                   return_shear=False,
                   return_shear_model='envelope',
                   cosmo=None):

    """
    Function to initalize a WLdata class for individual clusters.
    If done after fitting, can optionally compute shear profiles in two scenarios :
        - return_shear_model='envelope' --> will compute the shear profile for each set of parameters in 'all_chains',
        - return_shear_model='median parameters' --> will compute the shear profile only for the median parameters.


    Parameters:
    - lens_id: ID of the lens.
    - cluster_cat: Catalog containing cluster information.
    - shear_profiles: Catalog containing shear profile information.
    - results: Results from lensing analysis.
    - all_chains: Posterior chains for parameter estimates (needed for 'envelope' model).
    - return_shear: Whether to return shear profile data (default: False).
    - return_shear_model: Type of shear model to return ('median parameters' or 'enveloppe').
    - cosmo: Cosmological parameters.

    Returns:
    - wldata: Weak lensing data object.
    - Optionally returns shear profiles based on the selected model.
    """

    # Extract relevant data based the ID of the chosen cluster
    results_id = results[np.isin(results['ID'], lens_id)]
    shear_id = shear_profiles[np.isin(shear_profiles['ID'], lens_id)]
    z_cl = cluster_cat[np.isin(cluster_cat['ID'], lens_id)]['z_p']
    rin = shear_id['rin']
    rout = shear_id['rout']
    gplus = shear_id['gplus']
    err_g = shear_id['errors']
    msci = shear_id['msci'][0]
    fl = shear_id['fl'][0]

    # Create a WLData object with the extracted data
    wldata = WLData(redshift=z_cl, rin=rin, rout=rout, gplus=gplus,
                    err_gplus=err_g, sigmacrit_inv=msci,
                    fl=fl, cosmo=cosmo)

    # Check if shear profiles should be returned
    if return_shear:

        if return_shear_model == 'median parameters':

            pmod_med = [results_id['c200_med'], results_id['r200_med']]
            # Compute shear profile
            gplus_med, rm, ev = WLmodel_np(wldata, pmod_med)
            # Mask to cut the radial extrapolation
            mask = (rm >= min(wldata.rin_wl)) & (rm <= max(wldata.rout_wl))
            return wldata, gplus_med[mask], rm[mask]


        elif return_shear_model == 'envelope':

            if all_chains is None:
                print('Error: posterior chains are needed to compute the envelope.')
                return


            chains = all_chains[np.isin(all_chains['ID'], lens_id)]

            # Sample 500 rows randomly from the chains for computational efficiency
            sampled_indices = random.sample(range(len(chains[0][1])), 500)
            pmod = np.array((chains[0][1][sampled_indices], chains[0][2][sampled_indices])).T
            gplus_results = []

            # Loop through the sampled chain rows to compute shear profile
            for i in tqdm(range(len(sampled_indices))):
                gplus, rm, ev = WLmodel_np(wldata, pmod[i])
                # Mask to cut the radial extrapolation
                mask = (rm >= min(wldata.rin_wl)) & (rm <= max(wldata.rout_wl))
                gplus_results.append(gplus[mask])

            return wldata, gplus_results, rm[mask]

    return wldata