import os
import numpy as np
import pandas as pd
import itertools
import pickle

import utils
import grad_utils
import policy

import sklearn.decomposition
import matplotlib.pyplot as plt

def _analysis1(Ob_NH, Ob_H):
    NH_mean = Ob_NH.mean(axis=1)
    H_mean  = Ob_H.mean(axis=1)
    print("NH_mean")
    print(NH_mean)
    print("H_mean")
    print(H_mean)
    HNH_cos = np.dot(NH_mean, H_mean) / (np.linalg.norm(NH_mean) * np.linalg.norm(H_mean))
    print(f"cosine of NH_mean, H_mean: {HNH_cos}")
    NH_std  = Ob_NH.std(axis=1)
    H_std   = Ob_H.std(axis=1)
    print("overlap between NH, H means via their std")
    print(get_overlap(
        (NH_mean - NH_std, NH_mean + NH_std),
        (H_mean - H_std, H_mean + H_std),
    ))

def real_setup(
    k,
    n_u,
    n_NH, n_H,
    Ob_rescale,
    U0_path,
    Ob_path,
    user_idx=None,
    user_tot=None,
    include_Evec=True
):
    "if user_idx given, split up total n_u users"
    # load data
    U0 = pd.read_csv(U0_path, index_col=0)
    Ob = pd.read_csv(Ob_path, index_col=0)

    # reduce set of movies to n_NH, n_H
    if n_NH is not None:
        Ob_NH = Ob.filter(like='objNH').sample(n=n_NH, axis=1, random_state=42)
    else:
        Ob_NH = Ob.filter(like='objNH')
    if n_H is not None:
        Ob_H  = Ob.filter(like='objH').sample(n=n_H, axis=1, random_state=42)
    else:
        Ob_H = Ob.filter(like='objH')
    print(f"Ob_NH shape: {Ob_NH.shape}")
    print(f"Ob_H shape: {Ob_H.shape}")

    # quick data analysis
    _analysis1(Ob_NH, Ob_H)

    # increase magnitude of embeddings to increase diversity in preference
    Ob_filter = Ob_rescale * pd.concat((Ob_NH, Ob_H), axis=1)
    U0_filter = U0.sample(n=n_u, axis=1, random_state=42)
    print(f"U0_filter shape: {U0_filter.shape}")

    # enumerate all possible recommendations of size k
    E_vec=None
    if include_Evec:
        E_vec = [set(e) for e in 
                list(itertools.combinations(
                    Ob_filter.filter(like='objNH').columns, k
                ))]

    # optional for slurm: only take part of users
    assert (user_idx and user_tot) or (user_idx is None and user_tot is None)
    # 1-index!
    if user_idx:
        assert 0 < user_idx <= user_tot
        step = (1/user_tot) * n_u
        assert step.is_integer()
        start = int((user_idx-1) * step)
        end   = int((user_idx) * step)
        U0_filter = U0_filter.iloc[:,start:end]
        print(f"returning split of U0_filter: {start}-{end} with shape {U0_filter.shape}")

    return U0_filter, Ob_filter, U0, Ob, E_vec

def get_overlap(a, b):
    return np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]))

def res_to_pd(res, a, b):
    df = pd.DataFrame(
        np.zeros((a,b)),
        index=list(res.keys()),
        columns=list(res['gradk'].keys())
    )
    for alg in res.keys():
        df.loc[alg] = res[alg].loc[0]
    return df
