import pymc as pm
import numpy as np
from astropy.table import Table
from tqdm import tqdm
from .modeling import WLData, WLmodel, rdelt_to_mdelt


def run(cluster_cat, shear_profiles, cosmo):
    all_c200_chains = []
    all_r200_chains = []

    for cluster in tqdm(cluster_cat):
        clust_id = cluster['ID']
        clust_z = cluster['z_p']

        mask = shear_profiles['ID'] == clust_id
        cluster_profiles = shear_profiles[mask]

        rin = cluster_profiles['rin']
        rout = cluster_profiles['rout']
        gplus = cluster_profiles['gplus']
        errors = cluster_profiles['errors']

        mean_sigcrit_inv, fl = cluster_profiles['msci'][0], cluster_profiles['fl'][0]

        wldata = WLData(redshift=clust_z, rin=rin, rout=rout, gplus=gplus,
                        err_gplus=errors, sigmacrit_inv=mean_sigcrit_inv, fl=fl, cosmo=cosmo)

        basic_model = pm.Model()

        with basic_model:
            cdelt = pm.Uniform(name='cdelt', lower=0., upper=10.)
            rdelt = pm.Uniform(name='rdelt', lower=200., upper=4000.)
            pmod = [cdelt, rdelt]
            gmodel, rm, ev = WLmodel(wldata, pmod)
            g_obs = pm.Normal('WL', mu=gmodel[ev], observed=wldata.gplus, sigma=wldata.err_gplus)

        with basic_model:
            trace = pm.sample()

        all_c200_chains.append(np.array(trace.posterior['cdelt']).flatten())
        all_r200_chains.append(np.array(trace.posterior['rdelt']).flatten())

    all_chains = Table()
    all_chains['ID'] = [cluster['ID'] for cluster in cluster_cat]
    all_chains['c200'] = all_c200_chains
    all_chains['r200'] = all_r200_chains


    # Calculate median, 16th, and 84th percentiles for c200 and r200
    c200_med = np.median(all_chains['c200'], axis=1)
    c200_perc_16 = np.percentile(all_chains['c200'], 16, axis=1)
    c200_perc_84 = np.percentile(all_chains['c200'], 84, axis=1)

    r200_med = np.median(all_chains['r200'], axis=1)
    r200_perc_16 = np.percentile(all_chains['r200'], 16, axis=1)
    r200_perc_84 = np.percentile(all_chains['r200'], 84, axis=1)

    # Calculate m200
    z_p = cluster_cat['z_p']

    m200_med = rdelt_to_mdelt(r200_med, z_p, cosmo)
    m200_perc_16 = rdelt_to_mdelt(r200_perc_16, z_p, cosmo)
    m200_perc_84 = rdelt_to_mdelt(r200_perc_84, z_p, cosmo)

    results_table = Table()

    # Add columns to the table
    results_table['ID'] = cluster_cat['ID']
    results_table['m200_med'] = m200_med
    results_table['m200_perc_16'] = m200_perc_16
    results_table['m200_perc_84'] = m200_perc_84
    results_table['r200_med'] = r200_med
    results_table['r200_perc_16'] = r200_perc_16
    results_table['r200_perc_84'] = r200_perc_84
    results_table['c200_med'] = c200_med
    results_table['c200_perc_16'] = c200_perc_16
    results_table['c200_perc_84'] = c200_perc_84

    return all_chains, results_table
