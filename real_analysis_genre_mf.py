import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument('--genre', type=str)
parser.add_argument('--name', type=str)

import utils
from ML_IMDB_NMF import real_setup, res_to_pd

def gather_reps(folder_name):
    "combines into one dictionary each"
    res = {}
    res_pCLK = {}
    res_pH = {}
    for file_name in listdir(folder_name):
        if isfile(join(folder_name, file_name)) and ('pkl' in file_name):
            with open(join(folder_name, file_name), 'rb') as f:
                res_tmp, res_pCLK_tmp, res_pH_tmp = pickle.load(f)
                res      = _combine_res(res, res_tmp)
                res_pCLK = _combine_resP(res_pCLK, res_pCLK_tmp)
                res_pH   = _combine_resP(res_pH, res_pH_tmp)

    return res, res_pCLK, res_pH

def calc_tlim_dist(res, U0, Ob, alpha_H, alpha_NH, beta, c):
    Ulim, Slim = calc_Ulim_Slim(
        res, U0, Ob, 
        alpha_H, alpha_NH, beta, c)
    Ut, St = calc_Ut_St(
        res, U0, Ob,
        alpha_H, alpha_NH, beta, c,
        nsteps=30
        )
    dist = {}
    for alg in Ut.keys():
        dist[alg] = np.linalg.norm(Ut[alg] - Ulim[alg], axis=0)
    return dist

def dist_to_pd(res_dist):
    df = pd.DataFrame(
        np.zeros((4,2)),
        index=res_dist.keys(),
        columns=['mu','std']
    )
    for alg in res_dist.keys():
        df['mu'].loc[alg] = res_dist[alg].mean(axis=1).item()
        df['std'].loc[alg] = res_dist[alg].std(axis=1).item()
    return df

def _combine_metrics(d_metric):
    df = pd.DataFrame(
        np.zeros((4,2)),
        index=d_metric.columns,
        columns=['mu','std']
    )
    df['mu'] = d_metric.mean(axis=0)
    df['std'] = d_metric.std(axis=0)
    return df

def _helper_table(df, rd, col):
    """
    convert to string and mu (+/- std) string
    rd can be 2e, 2f, etc
    """
    format_str = "{:." + rd + "}"
    str_mu = df['mu'].map(lambda x: format_str.format(x))
    str_std = df['std'].map(lambda x: format_str.format(x))
    df_str = pd.DataFrame(['a']*4, index=['gradk','alt','u0','unif'], columns=[col])
    for alg in df.index:
        df_str.loc[alg] = str_mu[alg] + r' ($\pm$' + str_std[alg] + ')'
    return df_str

def make_table_row(res_obj, res_pCLK, res_pH, res_dist, rd_metric, rd_dist):
    df_obj = _helper_table(_combine_metrics(res_to_pd(res_obj, 4, 100).T), rd_metric, "obj")
    df_pCLK = _helper_table(_combine_metrics(res_to_pd(res_pCLK, 4, 100).T), rd_metric, "pCLK")
    df_pH = _helper_table(_combine_metrics(res_to_pd(res_pH, 4, 100).T), rd_metric, "pH")
    df_dist = _helper_table(dist_to_pd(res_dist), rd_dist, "dist")
    df_str = pd.concat([
        df_obj, df_pCLK, df_pH, df_dist
    ], axis=1)
    df_str.index = ['Grad', 'Alt', 'U0', 'Unif']
    return df_str

genre_dict = {
    'Action':   {'c' : None}, # WOULD NEED TO UPDATE THESE
    'Comedy':   {'c' : None},
    'Sci-Fi':   {'c' : None},
    'Fantasy':  {'c' : None},
    'Adventure':{'c' : None},
    'lam': 100,
    'aH': 0.50,
    'aNH': 0.25,
    'beta': 0.2
}

if __name__ == "__main__":
    load_name = f"path/to/policies"
    
    k   = 1        # rec set size
    lam = genre_dict['lam']        # harm penalty
    alpha_H  = genre_dict['aH'] # H addictiveness
    alpha_NH = genre_dict['aNH'] # NH addictiveness
    # beta   = 0.2   # influence of initial user state
    beta   = genre_dict['beta']   # influence of initial user state
    c      = genre_dict[genre]['c']     # influence of recs (smaller is more influence)
    nsteps_usersteady = 10    # number of steps to run F() for fixed point
    niter_alt = 10
    n_u = 100

    # setup data
    U0_filter, Ob_filter, _, _, E_vec = real_setup(
        k, n_u, None, None, Ob_rescale=1, 
        U0_path=f"data/{genre}_U_mf_df.csv",
        Ob_path=f"data/{genre}_Ob_mf_df.csv",
        user_idx=None, user_tot=None)
    S0 = utils.compute_pref_score(U0_filter, Ob_filter)

    # load policies
    res, res_pCLK, res_pH = gather_reps(load_name)

    # compute ||Ut-Ulim||
    res, _, _ = gather_reps(load_name)
    dist = calc_tlim_dist(res, U0_filter, Ob_filter, alpha_H, alpha_NH, beta, c)
    # table main results
    rd_metric="3f"
    df = make_table_row(res['obj'], res_pCLK, res_pH, dist, rd_metric, "1e")
    print(df.to_latex(escape=False))
