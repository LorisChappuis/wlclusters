import pymc as pm
import numpy as np
from astropy.table import Table
from tqdm import tqdm
from .modeling import WLData, WLmodel, rdelt_to_mdelt
from .utils import *

def run(cluster_cat, shear_profiles, cosmo, covtype='None', input_covmat=None, unit='proper', ndraws=2000, ntune=1000):
    all_c200_chains = []
    all_r200_chains = []

    if covtype == 'lss_cov':
        if 'ID' in input_covmat.colnames:
            print('using lss cov for each exact redshift')
        elif 'z' in input_covmat.colnames:
            print('using lss cov for an array of redshift')
        else:
            print('ERROR: The lss_cov table must contain either an "ID" or "z" column.')
    elif covtype == 'tot_cov':
        if 'ID' in input_covmat.colnames:
            print('using tot cov for each exact redshift')
        elif 'z' in input_covmat.colnames:
            print('using tot cov for an array of redshift')
        else:
            print('ERROR: The tot_cov table must contain either an "ID" or "z" column.')
    else:
        print('no lss covariance')



    for cluster in tqdm(cluster_cat):
        clust_id = cluster['ID']
        clust_z = cluster['z_p']

        mask = shear_profiles['ID'] == clust_id
        cluster_profiles = shear_profiles[mask]
        if covtype == 'lss_cov':
            lss_cov = input_covmat
            if 'ID' in lss_cov.colnames:
                # If the covariance matrix is based on cluster IDs
                cluster_lss_cov = lss_cov['covariance_matrix'][np.isin(lss_cov['ID'], clust_id)][0]
            elif 'z' in lss_cov.colnames:
                # If the covariance matrix is based on redshift
                closest_z = find_closest_redshift(clust_z, lss_cov['z'])
                cluster_lss_cov = lss_cov[np.isin(lss_cov['z'], closest_z)]['covariance_matrix'][0]
            else:
                raise ValueError('The lss_cov table must contain either an "ID" or "z" column.')
        elif covtype == 'tot_cov':
            tot_cov = input_covmat
            if 'ID' in tot_cov.colnames:
                # If the covariance matrix is based on cluster IDs
                cluster_tot_cov = tot_cov['covariance_matrix'][np.isin(tot_cov['ID'], clust_id)][0]
            elif 'z' in tot_cov.colnames:
                # If the covariance matrix is based on redshift
                closest_z = find_closest_redshift(clust_z, tot_cov['z'])
                cluster_tot_cov = tot_cov[np.isin(tot_cov['z'], closest_z)]['covariance_matrix'][0]
            else:
                raise ValueError('The tot_cov table must contain either an "ID" or "z" column.')
        else:
            cluster_lss_cov = np.zeros((len(cluster_profiles), len(cluster_profiles)))

        rin = cluster_profiles['rin']
        rout = cluster_profiles['rout']
        gplus = cluster_profiles['gplus']
        errors = cluster_profiles['errors']

        mean_sigcrit_inv, fl = cluster_profiles['msci'][0], cluster_profiles['fl'][0]

        wldata = WLData(redshift=clust_z, rin=rin, rout=rout, gplus=gplus,
                        err_gplus=errors, sigmacrit_inv=mean_sigcrit_inv, fl=fl, cosmo=cosmo, unit=unit)
        if covtype == 'tot_cov':
            cov_mat = cluster_tot_cov
        else:
            cov_mat = cluster_lss_cov + np.diag(np.square(wldata.err_gplus))

        basic_model = pm.Model()

        with basic_model:
            cdelt = pm.Uniform(name='cdelt', lower=0., upper=10.)
            rdelt = pm.Uniform(name='rdelt', lower=200., upper=4000.)
            pmod = [cdelt, rdelt]
            gmodel, rm, ev = WLmodel(wldata, pmod)
            g_obs = pm.MvNormal('WL', mu=gmodel[ev], observed=wldata.gplus, cov=cov_mat)

        with basic_model:
            trace = pm.sample(draws=ndraws, tune=ntune)

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

    if unit == 'proper':
        m200_med = rdelt_to_mdelt(r200_med, z_p, cosmo)
        m200_perc_16 = rdelt_to_mdelt(r200_perc_16, z_p, cosmo)
        m200_perc_84 = rdelt_to_mdelt(r200_perc_84, z_p, cosmo)

    if unit == 'comoving':
        r200_proper_med = r200_med * (1/(1+z_p))
        r200_proper_perc_16 = r200_perc_16 * (1/(1+z_p))
        r200_proper_perc_84 = r200_perc_84 * (1/(1+z_p))
        m200_med = rdelt_to_mdelt(r200_proper_med, z_p, cosmo)
        m200_perc_16 = rdelt_to_mdelt(r200_proper_perc_16, z_p, cosmo)
        m200_perc_84 = rdelt_to_mdelt(r200_proper_perc_84, z_p, cosmo)



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
