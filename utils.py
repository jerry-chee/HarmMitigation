import copy
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_pref_score(U, Ob):
    """
    computers users preference towards each object 
    """
    ip = np.exp(
        np.matmul(Ob.to_numpy().T, U.to_numpy())
    )
    S = pd.DataFrame(ip, columns=U.columns, index=Ob.columns)
    return S

def compute_agg_score_HNH(S):
    """
    computes aggregate scores for H, NH
    """
    S_H = S.filter(like="objH",axis=0).sum(axis=0)
    S_NH = S.filter(like="objNH",axis=0).sum(axis=0)
    return S_H, S_NH

def compute_agg_score(S, E):
    """
    computes aggregate scores for a given set E
    note: same E set applied to all users (columns)
    """
    S_E = S.filter(items=E, axis=0).sum(axis=0)
    return S_E

def compute_g_c(S, k, batch):
    """
    estimates c < s_\omega - s_E for all E \in supp(\pi)
    note that s\_oemga and s_E differ per user. 
    for now, I am just choosing 0.5*min(s_omega - s_E) over all users
    this is over rec-k policies
    batch: number of E esimates to calc c
    """
    E_batch = pd.DataFrame(
        np.array(random.choices(S.index, k=k*batch)).reshape((k, batch)),
        columns=["batch"+str(i) for i in range(batch)],
        index=["rec"+str(i) for i in range(k)]
    )
    S_batch = E_batch.apply(lambda col: compute_agg_score(S,col)).max(axis=1)
    S_omega = S.sum(axis=0)
    c_est = int(0.5 * min(S_omega - S_batch))
    return c_est

def g(S_E, c=1):
    """
    computes non-decreasing g function for p_{CLK|E}
    g(S_E) = S_E / (S_E + c)
    """
    return S_E / (S_E + c)

def prob_user_v_condE(user, S, E_u, S_omega, c):
    """
    computes probability a user selects v | E, forall v
    """
    # compute S_E
    S_E_u = compute_agg_score(S[user], E_u)
    # if v \in E or not
    v_ind = np.array(
        [s in E_u for s in S.index], dtype=int
    )
    # computes g(S_E)
    g_S_E_u = g(S_E_u, c)
    # common to both v \in E and v \notin E
    out = (1-g_S_E_u) * (S[user] / S_omega[user])
    # for only v \in E
    out += v_ind * g_S_E_u * (S[user] / S_E_u)
    return out

def prob_V(S, Pi, c):
    """
    computes probability users selects v \in \Omega, forall v
    pi: dict of dicts. user: {E : prob}
    """
    assert set(S.columns.to_list()) == set(list(Pi.keys()))
    pV = pd.DataFrame(
        np.zeros(S.shape),
        columns=S.columns,
        index=S.index
        )
    S_omega = S.sum(axis=0)
    # for each user
    for user in Pi.keys():
        # for each E in pi
        for E_u in Pi[user].keys():
            pV[user] += Pi[user][E_u] * prob_user_v_condE(user, S, E_u, S_omega, c)

    return pV

def prob_CLK(S, Pi, c):
    """
    probability a user will CLK a recommendation
    """
    assert set(S.columns.to_list()) == set(list(Pi.keys()))
    pCLK = pd.DataFrame(
        np.zeros((1,S.shape[1])),
        columns=S.columns
    )
    # for each user
    for user in Pi.keys():
        # for each E in pi
        for E_u in Pi[user].keys():
            # compute S_E
            S_E_u = compute_agg_score(S[user], E_u)
            pCLK[user] += Pi[user][E_u] * g(S_E_u, c)

    return pCLK

def prob_ORG(S, Pi, c):
    """
    probability a user will have an ORG interaction
    """
    assert set(S.columns.to_list()) == set(list(Pi.keys()))
    pORG = pd.DataFrame(
        np.zeros((1,S.shape[1])),
        columns=S.columns
    )
    # for each user
    for user in Pi.keys():
        # for each E in pi
        for E_u in Pi[user].keys():
            # compute S_E
            S_E_u = compute_agg_score(S[user], E_u)
            pORG[user] += Pi[user][E_u] * (1 - g(S_E_u, c))

    return pORG

def prob_H(S, Pi, c):
    """
    probability a user will interact with a harmful item
    note: p_NH = 1 - p_H
    want (S_H/S_Omega) * (1-pCLK) formulation for gradient consistency
    """
    pCLK = prob_CLK(S, Pi, c)
    S_H = S.filter(like="objH",axis=0).sum(axis=0)
    S_Omega = S.sum(axis=0)
    return (S_H / S_Omega) * (1. - pCLK)

def rec_topk_optimized(S, k, E_vec):
    """
    computes deterministic rec policy based on top scores in \Omega
    for each user
    note: E as a tuple here, but when checking convert to set
    map to E_vec, total list of enumerations of E
    """
    S_NH = S.filter(like="objNH", axis=0)
    def foo(user):
        tmp = S_NH[user].nlargest(k).index
        return tuple(tmp)
    Pi = {
        user: {foo(user) : 1.}
        for user in tqdm(S_NH.columns)
    }
    return Pi


def rec_topk(S, k, E_vec):
    """
    computes deterministic rec policy based on top scores in \Omega
    for each user
    note: E as a tuple here, but when checking convert to set
    map to E_vec, total list of enumerations of E
    """
    S_NH = S.filter(like="objNH", axis=0)
    Pi = {}
    for user in tqdm(S_NH.columns):
        tmp = S_NH[user].nlargest(k).index
        tmp_set = E_vec[E_vec.index(set(tmp))]
        Pi[user] = {
            tuple(tmp_set) : 1.
        }
    return Pi


def dict_keys_lookup(d):
    "turns dict keys which are tuples to sets for lookup"
    out = list(d.keys())
    out = [set(e) for e in out]
    return out

def G(U0, Ob, p_V, alpha_H, alpha_NH, beta):
    """
    implements G function for user steady state:
    (\sum_v \alpha_v p_v v) / (\sum_v \alpha_v p_v)
    note: U0 is initial user state, and p_V is updated from Ulim
    """
    H_ind = np.array(
        ["objH" in ob for ob in Ob.columns], dtype=int
    )
    alpha_V = H_ind * alpha_H + (1-H_ind) * alpha_NH
    numer = Ob.dot(alpha_V[:,np.newaxis] * p_V)
    denom = (alpha_V[:,np.newaxis] * p_V).sum(axis=0)
    return (beta*U0 + numer) / (beta + denom)

def user_steady(U0, Ob, Pi, alpha_H, alpha_NH, beta, c, nsteps=1, tol=1e-3):
    """
    comptues steady user state after n steps
    note: need to keep updating U, and dependents: S, p_V
    note: U0 argument is for beta term!
    """
    assert set(U0.columns.to_list()) == set(list(Pi.keys()))
    Ulim = copy.deepcopy(U0)
    Slim   = compute_pref_score(Ulim, Ob)
    p_Vlim = prob_V(Slim, Pi, c)
    for _ in tqdm(range(nsteps)):
        Ulim_old = Ulim
        Ulim = G(
            # note U0 is for beta term!
            U0=U0, Ob=Ob,
            p_V=p_Vlim,
            alpha_H=alpha_H, alpha_NH=alpha_NH, beta=beta
            )
        Slim   = compute_pref_score(Ulim, Ob)
        p_Vlim = prob_V(Slim, Pi, c)
    if np.any(np.abs(Ulim - Ulim_old) > tol):
        print(f"WARNING: Ulim-Ulim_old>{tol}")
    return Ulim

def _helper_timestep(U0, Ut, Ob, Pi, alpha_H, alpha_NH, beta, c):
    assert np.all(U0.columns == Ut.columns)
    H_ind = np.array( ["objH" in ob for ob in Ob.columns], dtype=int )
    St = compute_pref_score(Ut, Ob)
    p_V = prob_V(St, Pi, c)
    p_H = prob_H(St, Pi, c)
    p_NH = 1 - p_H
    Uout = pd.DataFrame(
        np.zeros(Ut.shape),
        columns=Ut.columns
    )
    alpha_V = H_ind * alpha_H + (1-H_ind) * alpha_NH
    for user in U0.columns:
        numer1 = Ob.dot(alpha_V * p_V[user])
        numer2 = (
            p_H[user].item() * (1 - alpha_H - beta) * Ut[user]
            ) + (
            p_NH[user].item() * (1 - alpha_NH - beta) * Ut[user]
            )
        Uout[user] = numer1 + numer2
    Uout += beta * U0
    return Uout 

def user_timestep(U0, Ob, Pi, alpha_H, alpha_NH, beta, c, nsteps=1, tol=1e-3):
    """
    computes user timestep iterations u(t+1) = ...
    note: need to keep updating U, and dependents: S, p_V
    note: U0 argument is for beta term!
    """
    assert set(U0.columns.to_list()) == set(list(Pi.keys()))
    Ut = copy.deepcopy(U0)
    for _ in range(nsteps):
        Ut_old = Ut
        Ut = _helper_timestep(U0, Ut, Ob, Pi, alpha_H, alpha_NH, beta, c)
        print(f"U absolute difference: {np.linalg.norm(Ut - Ut_old, axis=0).mean():.2e}")
        print(f"U norm mean: {np.linalg.norm(Ut,axis=0).mean()}")
        if np.all(np.abs((Ut - Ut_old)) < tol):
            break
    if np.any(np.abs(Ut - Ut_old) > tol):
        print(f"WARNING: Ulim-Ulim_old>{tol}")
    return Ut

def alg_alt_opt(U_0, Ob, Pi_0, E_vec, k, alpha_H, alpha_NH, beta, c, niter):
    """
    alternative optimization to solve for user steady state and policy
    """
    Pi = Pi_0
    U = U_0
    # initial guess for policy is top-k rec given current scores
    # S = compute_pref_score(U, Ob)
    # Pi = rec_topk(S, k, E_vec)
    for _ in range(niter):
        U_old  = U
        Pi_old = Pi
        U  = user_steady(U, Ob, Pi, alpha_H, alpha_NH, beta, c, nsteps=10)
        S  = compute_pref_score(U, Ob)
        Pi = rec_topk(S, k, E_vec)
        # TODO turn off if use multiproc
        print(f"U absolute difference: {np.linalg.norm(U - U_old, axis=0).mean():.2e}")
        print(f"Pi absolute difference: {_Pi_diff(Pi, Pi_old)}")

    return U, Pi

def _Pi_diff(Pi, Pi_old):
    """
    counts average number of mistmatched objects in Pi recommended set
    assumes deterministic top-k selection
    """
    assert Pi.keys() == Pi_old.keys()
    first_key = list(Pi.keys())[0]
    k = len(list(Pi[first_key]))
    # k = len(list(Pi['user0']))
    diff = np.zeros(len(Pi.keys()))
    for i, user in enumerate(Pi.keys()):
        num_mismatch = len(
            set(list(Pi[user])[0]).symmetric_difference(
            set(list(Pi_old[user])[0])
        ))
        diff[i] = num_mismatch / k
    return np.mean(diff)

def gen_Evec(k, Ob):
    """
    generates all possible enumerations of the recommendation set: E_vec
    """
    E_vec = [set(e) for e in 
            list(itertools.combinations(
                Ob.filter(like='objNH').columns, k
            ))]
    return E_vec