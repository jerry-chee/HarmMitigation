import os
import time
import numpy as np
import pandas as pd
import itertools
import pickle
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, Manager

import policy_sample
import model
import scipy.optimize


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
    print(f"{Ob_NH.shape[1]} NH movies, {Ob_H.shape[1]} H movies")

def real_setup(
    U0_path,
    Ob_path,
    Ob_rescale,
    num_user,
    num_movie=None,
    rand_state=42,
    par_idx=None,
    par_tot=None,
):
    "if user_idx given, split up total n_u users"
    # load data
    U0 = pd.read_csv(U0_path, index_col=0)
    Ob = pd.read_csv(Ob_path, index_col=0)

    # reduce set of movies to n_NH, n_H
    Ob_NH = Ob.filter(like='objNH')
    Ob_H  = Ob.filter(like='objH')
    print(f"Ob_NH shape: {Ob_NH.shape}")
    print(f"Ob_H shape: {Ob_H.shape}")
    # quick data analysis
    _analysis1(Ob_NH, Ob_H)

    # increase magnitude of embeddings to increase diversity in preference
    Ob_filter = Ob_rescale * pd.concat((Ob_NH, Ob_H), axis=1)
    # reduce sample of users, movies
    U0_filter = U0.sample(n=int(num_user), axis=1, random_state=rand_state)
    if num_movie: Ob_filter = Ob.sample(n=int(num_movie), axis=1, random_state=rand_state)
    print(f"U0_filter shape: {U0_filter.shape}")

    # optional for slurm: only take part of users
    assert (par_idx and par_tot) or (par_idx is None and par_tot is None)
    # 1-index!
    if par_idx:
        assert 0 < par_idx <= par_tot
        assert (num_user % par_tot == 0)
        step = (1/par_tot) * num_user
        assert step.is_integer()
        start = int((par_idx-1) * step)
        end   = int((par_idx) * step)
        U0_filter = U0_filter.iloc[:,start:end]
        print(f"returning split of U0_filter: {start}-{end-1} with shape {U0_filter.shape}")

    return U0_filter, Ob_filter

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