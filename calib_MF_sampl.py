from ML_IMDB_sampl import real_setup
from real_analysis import plot_ml_data 
import policy_sample

import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool, Manager

if __name__ == "__main__":
    # parameters
    k   = 5          # rec set size
    lam = 100        # harm penalty
    alpha_H  = 0.5   # H addictiveness (default)
    alpha_NH = 0.25  # NH addictiveness (default)
    beta   = 0.2     # influence of initial user state (default)
    niter_alt = 10
    n_u = 1
    # n_u = 100
    tr_no_samples = 256
    tr_eps        = 5e-2
    tr_max_iter   = 100
    ev_no_samples = 512
    ev_eps        = 5e-2 #1e-2
    ev_max_iter   = 100


    # setup data
    #genre = "Action"
    #genre = "Adventure"
    #genre = "Comedy"
    #genre = "Fantasy"
    genre = "Sci-Fi"
    c = 20 # influence of recs (smaller is more influence)
    Ob_rescale = 1 #NMF action default
    print(f'genre: {genre}, c: {c}')

    U0_filter, Ob_filter, = real_setup(
        U0_path=f"data/{genre}_U_mf_df.csv",
        Ob_path=f"data/{genre}_Ob_mf_df.csv",
        num_user=n_u, Ob_rescale=Ob_rescale)


    # copied from policy_sample
    # alternating optimization
    with Pool() as pool:
        out = pool.map(partial(
            policy_sample.alt_policy, 
            k=k, Ob=Ob_filter, U0=U0_filter, 
            no_samples=tr_no_samples, 
            alpha_H=alpha_H, alpha_NH=alpha_NH, 
            beta=beta, c=c, 
            eps=tr_eps, max_iter=tr_max_iter,
            nsteps=niter_alt
            ), U0_filter.columns)
    res_alt_dict = {u: pi for (u, pi) in out}
    # Unif
    with Pool() as pool:
        out = pool.map(partial(
        policy_sample.unif_policy,
        k=k, Ob=Ob_filter,
        ), U0_filter.columns)
    res_unif_dict = {u: pi for (u, pi) in out}
    res_policy = {
        'alt' : res_alt_dict,
        'unif' : res_unif_dict
    }
    met_columns = ['alt','unif']
    # compute metrics
    res_obj = pd.DataFrame(
        np.zeros((U0_filter.shape[1], len(met_columns))),
        index=U0_filter.columns,
        columns=met_columns
    )
    res_pCLK = pd.DataFrame(
        np.zeros((U0_filter.shape[1], len(met_columns))),
        index=U0_filter.columns,
        columns=met_columns
    )
    res_pH = pd.DataFrame(
        np.zeros((U0_filter.shape[1], len(met_columns))),
        index=U0_filter.columns,
        columns=met_columns
    )
    rsource_dict = None
    n_H = Ob_filter.filter(like='objH').shape[1]
    n_NH = Ob_filter.filter(like='objNH').shape[1]
    HM = list(range(n_NH, n_NH+n_H))
    E_set = None
    with Pool() as pool:
        out = pool.map(partial(
        policy_sample._helper_metric,
        Ob=Ob_filter, U0=U0_filter, 
        no_samples=ev_no_samples, HM=HM,
        alpha_H=alpha_H, alpha_NH=alpha_NH, 
        beta=beta, c=c, lam=lam, 
        eps=ev_eps, max_iter=ev_max_iter,
        res_policy=res_policy, E_set=E_set,
        rsource_dict=rsource_dict,
        ), U0_filter.columns)
    for u, (tmp_obj, tmp_pCLK, tmp_pH) in out:
        for alg in res_policy.keys():
            res_obj[alg].loc[u] = tmp_obj[alg]
            res_pCLK[alg].loc[u] = tmp_pCLK[alg]
            res_pH[alg].loc[u] = tmp_pH[alg]

    print(f"steady state alt : pCLK: {res_pCLK['alt'].mean():.3f}, pH: {res_pH['alt'].mean():.3f}")
    print(f"steady state unif: pCLK: {res_pCLK['unif'].mean():.3f}, pH: {res_pH['unif'].mean():.3f}")
