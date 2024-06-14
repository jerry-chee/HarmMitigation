import os
import copy
import random
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import scipy.optimize
from functools import partial

import utils
import model

# global for multiprocessing
use_multiproc = True

# successful finite difference check
def deriv_Gp_pv(U, Ob, S, Pi, v, alpha_H, alpha_NH, beta, c):
    """
    computes partial derivative \partial G_i(p) / \partial p_v
    v is str of object name
    vectorized over all users, and all coordiantes
    for a given v
    """
    p_V = utils.prob_V(S, Pi, c)
    H_ind = np.array(
        ["objH" in ob for ob in Ob.columns], dtype=int
    )
    alpha_V = H_ind * alpha_H + (1-H_ind) * alpha_NH
    numer = alpha_V[Ob.columns.get_loc(v)] * (
        Ob[v].to_numpy()[:,np.newaxis] -  Ob).dot(
            alpha_V[:,np.newaxis] * p_V)
    numer += alpha_V[Ob.columns.get_loc(v)] * beta * (
        Ob[v].to_numpy()[:,np.newaxis] - U)
    denom = np.square(beta + (alpha_V[:,np.newaxis] * p_V).sum(axis=0))
    return numer / denom

# successful finite difference check
def deriv_user_pvE_uj_a(user, Ob, S, E, v, c):
    """
    computes partial derivative for given user and E:
    \partial / \partial ulim 
    \frac{s_v}{\sum_{v' \in E} s_{v'} + c}
    v is str of object name
    c is the constant from g(S_E) = S_E / (S_E + c)
    """
    s_v_u = S[user].loc[v]
    Ob_E = Ob.filter(items=E)
    S_E_u  = S[user].filter(items=E, axis=0)
    numer = s_v_u * (
        (Ob[v].to_numpy()[:,np.newaxis] - Ob_E).dot(S_E_u) + \
            c * Ob[v].to_numpy()
    )
    denom = np.square((S_E_u).sum(axis=0) + c)
    return numer / denom

def fn_user_pvE_uj_a(user, S, E, v, c):
    """
    computes function 
    \frac{s_v}{\sum_{v'\in E} s_{v'} + c}
    """
    s_v_u = S[user].loc[v]
    S_E_u  = S[user].filter(items=E, axis=0)
    denom = S_E_u.sum(axis=0) + c
    return s_v_u / denom

# successful finite difference check
def deriv_pvE_uj_b(Ob, S, c):
    """
    computes partial derivative:
    /partial / partial ulim
    \frac{c}{\sum_{v' \in \catalog} s_{v'}}
    """
    numer = -c * Ob.dot(S)
    denom = np.square(S.sum(axis=0))
    return numer / denom

def fn_pvE_uj_b(S, c):
    """
    computes function
    c / \sum_{v' \in \catalog} s_{v'}
    """
    denom = S.sum(axis=0)
    return c / denom

# successful finite difference check
def deriv_user_pvE_uj(user, Ob, S, E, v, c):
    """
    computes partial derivative for given user and E:
    \partial p_{V|E} / \partial \overline{u}_j
    """
    sub_deriv_a = deriv_user_pvE_uj_a(user, Ob, S, E, v, c) 
    sub_deriv_b = deriv_pvE_uj_b(Ob, S[user], c)
    out =  fn_pvE_uj_b(S[user], c) * sub_deriv_a 
    out += fn_user_pvE_uj_a(user, S, E, v, c) * sub_deriv_b
    if v in E:
        out += sub_deriv_a
    return out

# successful finite difference check
def deriv_user_pv_uj(user, Ob, S, Pi, v, c):
    """
    computes partial derivative:
    \partial p_v / \partial \overline{u}_j
    """
    # for each E in pi
    out = pd.DataFrame(
        np.zeros((Ob.shape[0],1)),
        columns=['']
    )
    for E_u in Pi[user].keys():
        out += Pi[user][E_u] * deriv_user_pvE_uj(user, Ob, S, E_u, v, c).to_numpy()[:,np.newaxis]
    return out

# successful finite difference check
def deriv_F_U(U0, Ob, Slim, Pi, alpha_H, alpha_NH, beta, c):
    """
    computes jacobian of F wrt \overline{u}, for all users
    F_i along rows, u_j along columns
    note that U0 is the original user state, and Slim the preferences from Ulim
    """
    F_dict = {}
    for user in U0.columns:
        F_dict[user] = pd.DataFrame(
            np.zeros(
                (U0.shape[0], U0.shape[0])
            ),
            index   =[f"dF{i}" for i in range(U0.shape[0])],
            columns =[f"du{j}" for j in range(U0.shape[0])]
        )
    for v in Ob.columns:
        dGp_pv = deriv_Gp_pv(U0, Ob, Slim, Pi, v, alpha_H, alpha_NH, beta, c)
        for user in U0.columns:
            d_u_pv_uj = deriv_user_pv_uj(user, Ob, Slim, Pi, v, c).to_numpy()
            # for outer make sure F_i 1st input, u_j 2nd input 
            F_dict[user] +=  np.outer(dGp_pv[user].to_numpy(), d_u_pv_uj)
    return F_dict

# successful finite difference check
def deriv_user_pv_piE(user, S, E, v, c):
    """
    computes partial derivative:
    \partial p_v / \partial \pi_E
    """
    s_v_u = S[user].loc[v]
    S_E_u = S[user].filter(items=E, axis=0)
    numer_1 = c * s_v_u
    denom_1 = S_E_u.sum(axis=0) + c
    denom_2 = S[user].sum(axis=0)
    out = numer_1 / (denom_1 * denom_2)
    if v in E:
        out += s_v_u / denom_1
    return out

def _helper_dF_Pi(user, dGp_pv, E, Ob, S, v, c):
    d_u_pv_piE = deriv_user_pv_piE(user, S, E, v, c)
    return (dGp_pv[user] * d_u_pv_piE).to_numpy()

# successful finite difference check
def deriv_F_Pi(U0, Ob, Slim, Pi, E_vec, alpha_H, alpha_NH, beta, c):
    """
    optional parallelization via multiprocessing.pool
    computes jacobian of F wrt pi, for all users
    F_i along rows, pi_E along columns
    F_i \in R^d, and pi_E \in R^m (number of enumerations of E)
    E_vec: enumeration over all possible sets E
    """
    if use_multiproc: pool = multiprocessing.Pool(os.cpu_count())
    F_dict = {}
    for user in U0.columns:
        F_dict[user] = pd.DataFrame(
            np.zeros(
                (U0.shape[0], len(E_vec))
            ),
            index   =[f"dF{i}" for i in range(U0.shape[0])],
            columns =[f"dpiE{j}" for j in range(len(E_vec))]
        )
    for v in Ob.columns:
        dGp_pv = deriv_Gp_pv(U0, Ob, Slim, Pi, v, alpha_H, alpha_NH, beta, c)
        for user in U0.columns:
            if use_multiproc:
                out = pool.starmap(
                    _helper_dF_Pi,
                    zip(
                        itertools.repeat(user),
                        itertools.repeat(dGp_pv),
                        E_vec,
                        itertools.repeat(Ob),
                        itertools.repeat(Slim),
                        itertools.repeat(v),
                        itertools.repeat(c)
                    )
                )
                F_dict[user][:].loc[:] += np.array(out).T
            else:
                for j, E in enumerate(E_vec):
                    F_dict[user][f"dpiE{j}"] += _helper_dF_Pi(user, dGp_pv, E, Ob, Slim, v, c)
    return F_dict

# successful finite difference check
def deriv_Ulim_Pi(U0, Ob, Slim, Pi, E_vec, alpha_H, alpha_NH, beta, c):
    """
    computes gradient_pi ulim(pi)
    """
    dF_U  = deriv_F_U(U0, Ob, Slim, Pi, alpha_H, alpha_NH, beta, c)
    dF_Pi = deriv_F_Pi(U0, Ob, Slim, Pi, E_vec, alpha_H, alpha_NH, beta, c)
    dUlim = {}
    for user in U0.columns:
        A = -(dF_U[user] - np.identity(U0.shape[0]))
        B = dF_Pi[user]
        dUlim[user] = pd.DataFrame(
            np.linalg.solve(A,B),
            index   =[f"dulim{i}" for i in range(U0.shape[0])],
            columns =[f"dpiE{j}" for j in range(len(E_vec))]
        )
    return dUlim

def _helper_pCLK_Pi(S, E, c):
    "Note: this S_E is an aggregate score, not S filtered"
    S_E = utils.compute_agg_score(S, E)
    return utils.g(S_E, c)

# successful finite difference check
def deriv_pCLK_Pi(Slim, Ob, E_vec, c):
    """
    optional parallelization via multiprocessing
    computes gradient_pi pCLK
    """
    dpCLK = {}
    for user in Slim.columns:
        dpCLK[user] = pd.DataFrame(
            np.zeros(len(E_vec)),
            index =[f"dpiE{j}" for j in range(len(E_vec))],
            columns=['']
        )
    if use_multiproc: 
        pool = multiprocessing.Pool(os.cpu_count())
        out = pool.starmap(
            _helper_pCLK_Pi,
            zip(
                itertools.repeat(Slim),
                E_vec,
                itertools.repeat(c)
            )
        )
        for i, row in enumerate(out):
            for user in Slim.columns:
                dpCLK[user].loc[f"dpiE{i}"] = out[i][user]
    else:
        for i, E in enumerate(E_vec):
            tmp = _helper_pCLK_Pi(Slim, E, c)
            for user in Slim.columns:
                dpCLK[user].loc[f"dpiE{i}"] = tmp[user]
    return dpCLK

def _helper_pCLK_U(Ob, S, Pi, E, c):
    "just do for loops for simplicity"
    S_E = utils.compute_agg_score(S, E)
    dg = c / np.square(S_E + c)
    Ob_E_filter = Ob.filter(items=E, axis=1)
    S_E_filter = S.filter(items=E, axis=0)
    out = pd.DataFrame(
        np.zeros(
            (Ob.shape[0], S.shape[1])
        ),
        columns=S.columns,
        index=Ob.index
    )
    # dsE/duj
    for i in range(Ob.shape[0]):
        tmp = Ob_E_filter.loc[i].to_numpy()[:,np.newaxis] * S_E_filter
        tmp = tmp.sum(axis=0)
        out.loc[i] = tmp
    # g'
    out *= dg
    # pi_E probability
    for user in S.columns:
        if set(E) in utils.dict_keys_lookup(Pi[user]):
            try:
                out[user] *= Pi[user][E]
            except:
                out[user] *= Pi[user][tuple(E)]
        else:
            out[user] *= 0.0
    return out

# successful finite difference check
def deriv_pCLK_U(Slim, Ob, Pi, E_vec, c):
    """ 
    computes gradient_ulim pCLK
    """
    dpCLK = pd.DataFrame(
        np.zeros(
            (Ob.shape[0], Slim.shape[1])
        ),
        columns=Slim.columns,
        index=Ob.index
    )
    for E in E_vec:
        dpCLK += _helper_pCLK_U(Ob, Slim, Pi, E, c)
    return dpCLK

# successful finite difference check
def deriv_pCLK(U0, Slim, Ob, Pi, E_vec, alpha_H, alpha_NH, beta, c):
    """
    gradient_pi pCLK(pi, ulim(pi))
    """
    d_pCLK_pi = deriv_pCLK_Pi(Slim, Ob, E_vec, c)
    d_Ulim_pi = deriv_Ulim_Pi(U0, Ob, Slim, Pi, E_vec, alpha_H, alpha_NH, beta, c)
    d_pCLK_u  = deriv_pCLK_U(Slim, Ob, Pi, E_vec, c)
    dpCLK = {}
    for user in U0.columns:
        dpCLK[user] = d_pCLK_pi[user] + \
            np.matmul(d_pCLK_u[user].to_numpy(), d_Ulim_pi[user]).to_frame(name='')
    return dpCLK

# successful finite difference check
def deriv_pH_Pi(Slim, Ob, E_vec, c):
    """
    computes gradient_pi pH
    """
    S_H, S_NH = utils.compute_agg_score_HNH(Slim)
    S_Omega = S_H + S_NH
    d_pCLK_pi = deriv_pCLK_Pi(Slim, Ob, E_vec, c)
    dpH = {}
    tmp = - S_H / S_Omega
    for user in Slim.columns:
        dpH[user] = tmp[user] * d_pCLK_pi[user]
    return dpH

def _helper_pH_U(Slim, Ob, c):
    Ob_H = Ob.filter(like='objH', axis=1)
    denom = np.square(Slim.sum(axis=0) + c)
    out = pd.DataFrame(
        np.zeros(
            (Ob.shape[0], Slim.shape[1])
        ),
        columns=Slim.columns,
        index=Ob.index
    )
    for v in Ob_H.columns:
        numer = Slim.loc[v].to_numpy()[np.newaxis,:] * (
            (Ob_H[v].to_numpy()[:,np.newaxis] - Ob).dot(Slim) + 
            (c * Ob_H[v]).to_numpy()[:,np.newaxis]
        )
        out += numer
    out /= denom
    return out

# successful finite difference check
def deriv_pH_U(Slim, Ob, Pi, E_vec, c):
    """
    computes gradient_u pH
    """
    d_sh_so = _helper_pH_U(Slim, Ob, c)
    pCLK = utils.prob_CLK(Slim, Pi, c)
    d_pCLK_u = deriv_pCLK_U(Slim, Ob, Pi, E_vec, c)
    S_H, S_NH = utils.compute_agg_score_HNH(Slim)
    pH = d_sh_so * (1. - pCLK.to_numpy()) \
        - (S_H / (S_H+S_NH)) * d_pCLK_u
    return pH

# successful finite difference check
def deriv_pH(U0, Slim, Ob, Pi, E_vec, alpha_H, alpha_NH, beta, c):
    """
    computes gradient_pi pH(pi, ulim(pi))
    """
    d_pH_pi   = deriv_pH_Pi(Slim, Ob, E_vec, c)
    d_Ulim_pi = deriv_Ulim_Pi(U0, Ob, Slim, Pi, E_vec, alpha_H, alpha_NH, beta, c)
    d_pH_u    = deriv_pH_U(Slim, Ob, Pi, E_vec, c)
    dpH = {}
    for user in U0.columns:
        dpH[user] = d_pH_pi[user] + \
            np.matmul(d_pH_u[user].to_numpy(), d_Ulim_pi[user]).to_frame(name='')
    return dpH

def hm_objective(Pi_vec, *args):
    """
    Note: Pi_vec is a 1-dim vector Pi_dict is the dictionary representation
    Note: for a given user (thought previously could vectorize across users)
    pCLK - lambda pH
    """
    print("called hm_objective")
    # fixed parameters
    lam      = args[0]
    U0       = args[1]
    Ob       = args[2]
    E_vec    = args[3]
    alpha_H  = args[4]
    alpha_NH = args[5]
    beta     = args[6]
    c        = args[7]
    nsteps   = args[8]
    user     = args[9]
    assert len(Pi_vec) == len(E_vec)
    # Pi_vec to Pi_dict. only store nonzero probabilities
    Pi_dict = {
        user: {
            tuple(E):prob for E,prob in zip(E_vec, Pi_vec) if prob
        }
    }
    # user steady state
    Ulim = utils.user_steady(
        U0[user].to_frame(), Ob, Pi_dict, 
        alpha_H, alpha_NH, beta, c, nsteps
    )
    Slim = utils.compute_pref_score(Ulim, Ob)
    # objective. Note: take negation for minimization
    obj = - utils.prob_CLK(Slim, Pi_dict, c) + lam * utils.prob_H(Slim, Pi_dict, c)
    return obj[user].loc[0]

def hm_jacobian(Pi_vec, *args):
    """
    gradient of objective
    """
    print("called hm_jacobian")
    # fixed parameters
    lam      = args[0]
    U0       = args[1]
    Ob       = args[2]
    E_vec    = args[3]
    alpha_H  = args[4]
    alpha_NH = args[5]
    beta     = args[6]
    c        = args[7]
    nsteps   = args[8]
    user     = args[9]
    assert len(Pi_vec) == len(E_vec)

    # Pi_vec to Pi_dict. only store nonzero probabilities
    Pi_dict = {
        user: {
            tuple(E):prob for E,prob in zip(E_vec, Pi_vec) if prob
        }
    }
    # user steady state
    Ulim = utils.user_steady(
        U0[user].to_frame(), Ob, Pi_dict, 
        alpha_H, alpha_NH, beta, c, nsteps
    )
    Slim = utils.compute_pref_score(Ulim, Ob)
    # jacobian
    d_pCLK = deriv_pCLK(
        U0[user].to_frame(), Slim, Ob, 
        Pi_dict, E_vec, 
        alpha_H, alpha_NH, beta, c
    )
    d_pH = deriv_pH(
        U0[user].to_frame(), Slim, Ob, 
        Pi_dict, E_vec, 
        alpha_H, alpha_NH, beta, c
    )
    jac = - d_pCLK[user] + lam * d_pH[user]
    return jac.to_numpy().flatten()

def hm_minimize(Pi_vec0, *args, ftol=1e-4, maxiter=20):
    """
    runs scipy.optimize.minimize for constrained optimization
    method SLSQP with ftol=1e-4 stopping condition
    init: 
    - U0_topk : topk from U0
    - rand : random initialization
    bounds: nonzero
    constrains: sum to 1
    ftol for convergence criteria
    """
    # fixed parameters
    lam      = args[0]
    U0       = args[1]
    Ob       = args[2]
    E_vec    = args[3]
    alpha_H  = args[4]
    alpha_NH = args[5]
    beta     = args[6]
    c        = args[7]
    nsteps   = args[8]
    user     = args[9]
    # bounds: nonzero
    bnds = [(0,None) for _ in E_vec] 
    # constraints: sum to 1
    cons = (
        {'type': 'eq', 'fun': lambda x:  sum(x) - 1}
    )
    # run SLSQP
    res = scipy.optimize.minimize(
        fun=hm_objective,
        x0=Pi_vec0,
        args=args,
        jac=hm_jacobian,
        bounds=bnds,
        constraints=cons,
        method='SLSQP',
        options={
            "maxiter":maxiter,
            'ftol':ftol
        }
    )
    return res


def hm_minimize_bound(Pi_vec0, loss, k, ftol=1e-4, user=''):
    """
    runs scipy.optimize.minimize for the bounded cardinality case, with Stratis' code
    """
    bnds = [(0,None) for _ in range(len(Pi_vec0))]
    cons = (
        {'type': 'eq', 'fun': lambda x:  sum(x) - 1},
    )
    def callbackF(x):
        print(f"bound {user} obj: {loss(x, grad=False):.3f}, sum: {np.sum(x):.3f}")
        model.gradTester(loss, x, eps=0.1, delta=0.1)
    # run SLSQP
    res = scipy.optimize.minimize(
        fun=partial(loss, grad=True),
        callback=callbackF,
        x0=Pi_vec0,
        bounds=bnds,
        jac=True,
        constraints=cons,
        method='SLSQP',
        options={
            'ftol':ftol
        }
    )
    return res


def hm_minimize_sampl(Pi_vec0, loss, k, ftol=1e-4, user=''):
    """
    runs scipy.optimize.minimize for the independent sampling case
    """
    bnds = [(0,1) for _ in range(len(Pi_vec0))]
    cons = (
        #{'type': 'eq', 'fun': lambda x:  sum(x) - k},
        {'type': 'ineq', 'fun': lambda x:  k - sum(x)},
        {'type': 'eq', 'fun': lambda x:  x[loss.HM]}
    )
    def callbackF(x):
        print(f"sampl {user} obj: {loss(x, grad=False):.3f}, sum: {np.sum(x):.3f}")
        model.gradTester(loss, x, eps=0.1, delta=0.1)

    # run SLSQP
    res = scipy.optimize.minimize(
        fun=partial(loss, grad=True),
        callback=callbackF,
        x0=Pi_vec0,
        bounds=bnds,
        jac=True,
        constraints=cons,
        method='SLSQP',
        options={
            # 'ftol':ftol
        }
    )
    return res
