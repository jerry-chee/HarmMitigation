## fit and save various recommendation policies
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
import grad_utils

def _init(init_mode, user, U0, Ob, k, E_vec):
    # Initial guess: 
    if init_mode=="U0_topk":
        # topk over U0
        S = utils.compute_pref_score(U0, Ob)
        Pi0 = utils.rec_topk(S[user].to_frame(), k, E_vec)
        E0 = list(Pi0[user].keys())[0]
        E0_ind = E_vec.index(set(E0))
        Pi_vec0 = np.zeros(len(E_vec))
        Pi_vec0[E0_ind] = 1.0
        print(f"hm_minimize: initial guess for set E:{E0_ind}")
    elif init_mode=="rand":
        Pi_vec0 = np.random.rand(len(E_vec))
        Pi_vec0 /= sum(Pi_vec0)
        print("hm_minimize: initial guess random vector")
    elif init_mode=="unif":
        Pi_vec0 = np.ones(len(E_vec)) / len(E_vec)
        print("hm_minimize: initial guess unif vector")
    elif init_mode=="rand_int":
        Pi_vec0 = np.random.rand(len(E_vec))
        Pi_vec0 /= sum(Pi_vec0)
        Pi_vec0 /= 100
        print("hm_minimize: initial guess random vector interior/100")
    elif init_mode=="unif_int":
        Pi_vec0 = np.ones(len(E_vec)) / len(E_vec)
        Pi_vec0 /= 100
        print("hm_minimize: initial guess unif vector interior/100")
    else:
        raise NotImplementedError
    return Pi_vec0


def _init_alt(init_mode, U0, Ob, k, E_vec):
    # Initial guess: 
    if init_mode=="rand":
        Pi_vec0 = np.random.rand(len(E_vec))
        Pi_vec0 /= sum(Pi_vec0)
        print("hm_minimize: initial guess random vector")
    else:
        raise NotImplementedError
    return Pi_vec0

def grad_k_policy(
    k,init_mode,
    lam,
    U0, Ob,
    E_vec,
    alpha_H, alpha_NH, beta, c,
    nsteps,
    user,
    threshold=1e-3,
    ftol=1e-4,
    maxiter=20
):
    """
    calls grad_utils.hm_minimize() with initial guess, outputs final dict
    init 'rand' method minimizes objective bettern than U0_topk
    threshold=1e-3 to filter out essentially zero probabilities 
    """
    best_obj = np.inf
    res = None
    for im in ['U0_topk','unif','unif_int',\
        'rand','rand','rand','rand_int','rand_int','rand_int']:
        Pi_vec0 = _init(im, user, U0, Ob, k, E_vec)
        # run gradient algorithm
        res_tmp = grad_utils.hm_minimize(
            Pi_vec0,
            lam,
            U0,Ob,
            E_vec,
            alpha_H,alpha_NH,beta,c,
            nsteps,
            user,
            ftol=ftol,
            maxiter=maxiter
        )
        print(f"init_mode: {im}, obj: {res_tmp.fun}, success: {res_tmp.success}")
        if (res_tmp.fun < best_obj) and (res_tmp.success):
            print(f"new best init: {im}, obj: {res_tmp.fun}")
            best_obj = res_tmp.fun
            res = res_tmp

    # Pi_vec to Pi_dict. only store probabilities > 1e-3
    Pi_dict = {
        user: {
            tuple(E):prob for E,prob in zip(E_vec, res.x) if prob > threshold
        }
    }
    return Pi_dict, res

def alt_policy(
    k, init_mode,
    U0, Ob,
    E_vec,
    alpha_H, alpha_NH, beta, c,
    niter
):
    """
    calls utils.alg_alt_opt()
    runs over all users
    """
    Pi_vec0 = _init_alt(init_mode, U0, Ob, k, E_vec)
    Pi_dict0 = {
        user: {
            tuple(E):prob for E,prob in zip(E_vec, Pi_vec0)
        } for user in U0.columns
    }
    _, Pi_dict = utils.alg_alt_opt(U0, Ob, Pi_dict0, E_vec, k, alpha_H, alpha_NH, beta, c, niter)
    return Pi_dict

def unif_policy(
    U0, E_vec
):
    """
    uniform probability over all sets
    """
    p = 1 / len(E_vec)
    base_policy = {tuple(E):p for E in E_vec}
    Pi_dict = {
        user : base_policy for user in U0.columns
    }
    return Pi_dict

def u0topk_policy(
    S0, k, E_vec
):
    """
    choose topk entities over initial user state U0
    """
    return utils.rec_topk(S0, k, E_vec)

def _helper_objective(
    U0, Ob, Pi_dict,
    lam, alpha_H, alpha_NH, beta, c, 
    nsteps
):
    """
    computes objective pCLK - lam pH, over all users
    """
    # user steady state
    Ulim = utils.user_steady(
        U0, Ob, Pi_dict, 
        alpha_H, alpha_NH, beta, c, nsteps
    )
    Slim = utils.compute_pref_score(Ulim, Ob)
    # objective. Note: negatiion of obj for scipy.optimize
    obj = utils.prob_CLK(Slim, Pi_dict, c) - lam * utils.prob_H(Slim, Pi_dict, c)
    return obj

def _helper_policies(
    k, init,
    lam,
    U0, Ob,
    E_vec, 
    alpha_H, alpha_NH, beta, c,
    nsteps_usersteady, niter_alt, ftol=1e-4
):
    """
    for given parameter configuration
    computes policies and their respective objective values
    vectorized over all users
    """
    S0 = utils.compute_pref_score(U0, Ob)
    Pi_dict_gradk = {}
    res_dict_gradk = {}
    obj_dict = {}
    # Gradient algorithm for k cardinality
    for user in tqdm(U0.columns):
        Pi_dict_tmp, res_tmp = grad_k_policy(
            k, init, 
            lam, 
            U0, Ob, 
            E_vec, 
            alpha_H, alpha_NH, beta, c,
            nsteps_usersteady,
            user, ftol=ftol)
        Pi_dict_gradk[user] = Pi_dict_tmp[user]
        res_dict_gradk[user] = res_tmp
    # alternating optimization
    Pi_dict_alt = alt_policy(
        k, init, 
        U0, Ob, 
        E_vec, 
        alpha_H, alpha_NH, beta, c, 
        niter_alt)
    # uniform
    Pi_dict_unif = unif_policy(
        U0, E_vec)
    # topk over U0
    Pi_dict_u0 = u0topk_policy(
        S0, k, E_vec)
    # compute objective values
    for name, pidict in zip(
        ["gradk","alt","unif","u0"],
        [Pi_dict_gradk, Pi_dict_alt, Pi_dict_unif, Pi_dict_u0]
    ):
        obj_dict[name] = _helper_objective(
            U0, Ob, pidict, 
            lam, alpha_H, alpha_NH, beta, c, 
            nsteps_usersteady)
    return {
        "gradk": Pi_dict_gradk, 
        "res"  : res_dict_gradk, 
        "alt"  : Pi_dict_alt, 
        "unif" : Pi_dict_unif, 
        "u0"   : Pi_dict_u0,
        "obj"  : obj_dict
    }

def compute_lim_CLK_H(
    U0, Ob, res,
    alpha_H, alpha_NH, beta, c,
    nsteps_usersteady
):
    "computes pCLK, pH under Ulim, Pi_dict keys are different policies"
    results_pCLK = {}
    results_pH = {}
    for alg in res['obj'].keys():
        Ulim = utils.user_steady(
            U0, Ob, 
            res[alg], alpha_H, alpha_NH, beta, c,
            nsteps=nsteps_usersteady)
        Slim = utils.compute_pref_score(Ulim, Ob)
        results_pCLK[alg] = utils.prob_CLK(Slim, res[alg], c)
        results_pH[alg] = utils.prob_H(Slim, res[alg], c)
    return results_pCLK, results_pH