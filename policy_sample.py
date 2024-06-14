## fit and save various recommendation policies, independent sampling
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, Manager

import utils
import grad_utils
import model

import scipy.optimize

def _init(init_mode, user, U0, Ob, k, nH, nNH):
    # Initial guess: 
    if init_mode=="U0_topk":
        s0 = utils.compute_pref_score(U0[user].to_frame(), Ob)
        Pi_vec0 = s0.to_numpy().flatten()
        Pi_vec0[nNH:] = 0
        Pi_vec0 = k * Pi_vec0 / np.sum(Pi_vec0)
        # print(f"hm_minimize_smpl: initial guess u0_topk")
    elif init_mode=="rand":
        Pi_vec0 = np.random.rand(nNH)
        Pi_vec0 /= sum(Pi_vec0)
        Pi_vec0 *= k
        Pi_vec0 = np.concatenate( (Pi_vec0, np.zeros(nH)) )
        # print("hm_minimize: initial guess random vector")
    elif init_mode=="unif":
        Pi_vec0 = np.zeros(nNH+nH)
        Pi_vec0[list(range(nNH))] = k / nNH
        # print("hm_minimize: initial guess unif vector")
    elif init_mode=="rand_int":
        Pi_vec0 = np.random.rand(nNH)
        Pi_vec0 /= sum(Pi_vec0)
        Pi_vec0 *= k
        Pi_vec0 = np.concatenate( (Pi_vec0, np.zeros(nH)) )
        Pi_vec0 /= 100
        # print("hm_minimize: initial guess random vector interior/100")
    elif init_mode=="unif_int":
        Pi_vec0 = np.zeros(nNH+nH)
        Pi_vec0[list(range(nNH))] = k / nNH
        Pi_vec0 /= 100
        # print("hm_minimize: initial guess unif vector interior/100")
    else:
        raise NotImplementedError
    return Pi_vec0


def reshape_Ob(Ob):
    Ob_NH = Ob.filter(like='objNH')
    Ob_H = Ob.filter(like='objH')
    V = pd.concat([Ob_NH, Ob_H], axis=1)
    return V

def grad_bound_policy(
    u,
    k, Ob, U0,
    alpha_H, alpha_NH,
    beta, c, lam,
    eps, max_iter, threshold=1e-5,
    print_prob=1
):
    """
    calls grad_utils.hm_minimize() with initial guess, outputs final dict
    init 'rand' method minimizes objective bettern than U0_topk
    threshold=1e-3 to filter out essentially zero probabilities 
    requires V to have NH first, then H, and to transpose from how I generate
    """
    # assert k == 1
    nH = Ob.filter(like='objH').shape[1]
    nNH = Ob.filter(like='objNH').shape[1]
    HM = list(range(nNH, nNH+nH))
    E_set = list(itertools.combinations(range(nNH),k))
    best_obj = np.inf
    res = None
    for im in ['U0_topk','unif','unif_int',\
        'rand','rand','rand','rand_int','rand_int','rand_int']:
        Pi_vec0 = _init(im, u, U0, Ob, k, nH=nH, nNH=len(E_set))
        loss = model.BoundedCardinalityModelLoss(
            V=Ob.T.to_numpy(), u0=U0[u].to_numpy(), u0_id=u,
            E_set=E_set,
            HM=HM,
            a_HM=alpha_H, a_NH=alpha_NH,
            beta=beta, c=c, lam=lam,
            eps=eps, max_iter=max_iter)
        # run gradient algorithm
        res_tmp = grad_utils.hm_minimize_bound(
            Pi_vec0,
            loss, k, 
            ftol=1e-3, user=u)

        print(f"init_mode: {im}, obj: {res_tmp.fun}, success: {res_tmp.success}")
        if (res_tmp.fun < best_obj) and (res_tmp.success):
            print(f"new best init: {im}, obj: {res_tmp.fun}")
            best_obj = res_tmp.fun
            res = res_tmp

    # threhsold small numbers to zero
    res.x[abs(res.x) < threshold] = 0
    # print
    if np.random.rand(1) < print_prob:
        print(f"grad_bound_policy {u} completed")
    return u, res


def grad_sampl_helper(init_method,
    u, k, Ob, U0,
    no_samples, 
    alpha_H, alpha_NH, nH, nNH, HM,
    beta, c, lam,
    eps, max_iter,
    version='default'
):
    Pi_vec0 = _init(init_method, u, U0, Ob, k, nH=nH, nNH=nNH)
    loss = model.SampledModelLoss(
        V=Ob.T.to_numpy(), u0=U0[u].to_numpy(), u0_id=u,
        no_samples=no_samples,
        HM=HM,
        a_HM=alpha_H, a_NH=alpha_NH,
        beta=beta, c=c, lam=lam,
        eps=eps, max_iter=max_iter, restart_umap=True)

    # run gradient algorithm
    if version == 'default':
        res_tmp = grad_utils.hm_minimize_sampl(
            Pi_vec0,
            loss, k,
            ftol=1e-3, user=u)
    elif version == 'nograd':
        res_tmp = grad_utils.hm_minimize_sampl_nograd(
            Pi_vec0,
            loss, k,
            ftol=1e-3, user=u)
    return res_tmp, loss


def grad_sampl_policy(
    u, 
    k, Ob, U0,
    no_samples, 
    alpha_H, alpha_NH,
    beta, c, lam,
    eps, max_iter, threshold=1e-5,
    print_prob=1, version='default'
):
    """
    calls grad_utils.hm_minimize() with initial guess, outputs final dict
    init 'rand' method minimizes objective bettern than U0_topk
    threshold=1e-3 to filter out essentially zero probabilities 
    requires V to have NH first, then H, and to transpose from how I generate
    """
    nH = Ob.filter(like='objH').shape[1]
    nNH = Ob.filter(like='objNH').shape[1]
    HM = list(range(nNH, nNH+nH))
    best_obj = np.inf
    res = None

    im_ls = ['U0_topk','unif','unif_int', \
            'rand','rand','rand','rand_int','rand_int','rand_int']

    with Pool() as pool:
        out = pool.map(partial(
            grad_sampl_helper,
            u=u, k=k, Ob=Ob, U0=U0,
            no_samples=no_samples, alpha_H=alpha_H, alpha_NH=alpha_NH,
            nH=nH, nNH=nNH, HM=HM,
            beta=beta, c=c, lam=lam, eps=eps, max_iter=max_iter, version=version
            ), im_ls)
    
    for (res_tmp, loss_tmp), im in zip(out, im_ls):
        print(f"init_mode: {im}, obj: {res_tmp.fun}, success: {res_tmp.success}")
        if (res_tmp.fun < best_obj) and (res_tmp.success):
            print(f"new best init: {im}, obj: {res_tmp.fun}")
            best_obj = res_tmp.fun
            res = res_tmp
            loss = loss_tmp

    # threhsold small numbers to zero
    res.x[abs(res.x) < threshold] = 0
    # print
    if np.random.rand(1) < print_prob:
        print(f"grad_sampl_policy {u} completed with solver {version}")
    return u, res, loss.rsource

def alt_policy(
    u,
    k, Ob, U0, 
    no_samples,
    alpha_H, alpha_NH,
    beta, c,
    eps, max_iter,
    nsteps,
    print_prob=1
):
    """
    runs over all users
    """
    nH = Ob.filter(like='objH').shape[1]
    nNH = Ob.filter(like='objNH').shape[1]
    HM = list(range(nNH, nNH+nH))
    # Pi_vec = _init_unif(k=k, nH=nH, nNH=nNH)
    Pi_vec = _init('unif', u, U0, Ob, k, nH, nNH)

    user_score_map = model.SampledModelMap(
        V=Ob.T.to_numpy(), u0=U0[u].to_numpy(),
        no_samples=no_samples,
        HM=HM,
        a_HM=alpha_H, a_NH=alpha_NH,
        beta=beta, c=c,
        eps=eps, max_iter=max_iter)

    ulim = U0[u].to_numpy()
    for _ in range(nsteps):
        u_old = ulim
        pi_old = Pi_vec
        ulim, slim = user_score_map(Pi_vec)
        slim_NH = np.delete(slim, HM)
        topk_idx = np.argpartition(slim_NH, -k)[-k:]
        Pi_vec = np.zeros(nNH+nH)
        Pi_vec[topk_idx] = 1
        user_score_map.set_u_start(ulim)
    # print
    if np.random.rand(1) < print_prob:
        print(f"alt_policy {u} completed")
    return u, (Pi_vec, ulim)


def unif_policy(
    u,
    k, Ob,
    print_prob=1
):
    """
    uniform probability over all sets
    """
    nH = Ob.filter(like='objH').shape[1]
    nNH = Ob.filter(like='objNH').shape[1]
    p = k / nNH
    Pi_vec = p * np.ones(nNH)
    Pi_vec = np.concatenate( (Pi_vec, np.zeros(nH)) )
    # print
    if np.random.rand(1) < print_prob:
        print(f"unif_policy {u} completed")
    return u, Pi_vec

def u0topk_policy(
    u,
    k, Ob, U0,
    print_prob=1
):
    """
    choose topk entities over initial user state U0
    """
    s0 = utils.compute_pref_score(U0[u].to_frame(), Ob)
    s0_NH = s0.filter(like='objNH', axis=0).to_numpy().flatten()
    topk_idx = np.argpartition(s0_NH, -k)[-k:]
    Pi_vec = np.zeros(s0.shape[0])
    Pi_vec[topk_idx] = 1
    # print
    if np.random.rand(1) < print_prob:
        print(f"u0topk_policy {u} completed")
    return u, Pi_vec

def _helper_metric(
    u,
    Ob, U0, no_samples, HM,
    alpha_H, alpha_NH, beta, c, lam,
    eps, max_iter,
    res_policy, E_set=None, print_prob=1, rsource_dict=None
):
    rsource=None
    if rsource_dict: rsource = rsource_dict[u]
    metrics_sampl = model.SampledModelMetrics(
        V=Ob.T.to_numpy(), u0=U0[u].to_numpy(),
        no_samples=no_samples,
        HM=HM,
        a_HM=alpha_H, a_NH=alpha_NH,
        beta=beta, c=c,
        eps=eps, max_iter=max_iter, rsource=rsource)
    if 'gradk_bound' in res_policy.keys():
        assert E_set is not None
        metrics_bound = model.BoundedModelMetrics(
            V=Ob.T.to_numpy(), u0=U0[u].to_numpy(),
            E_set=E_set, HM=HM,
            a_HM=alpha_H, a_NH=alpha_NH,
            beta=beta, c=c,
            eps=eps, max_iter=max_iter)
    res_obj, res_pCLK, res_pH = {}, {}, {}
    for alg in res_policy.keys():
        if alg == 'gradk':
            tmp_obj, tmp_pCLK, tmp_pH = metrics_sampl(
                res_policy[alg][u].x, lam)
        elif alg == 'gradk_bound':
            tmp_obj, tmp_pCLK, tmp_pH = metrics_bound(
                res_policy[alg][u].x, lam)
        elif alg == 'alt':
            tmp_obj, tmp_pCLK, tmp_pH = metrics_sampl(
                res_policy[alg][u][0], lam)
        else:
            tmp_obj, tmp_pCLK, tmp_pH = metrics_sampl(
                res_policy[alg][u], lam)
        res_obj[alg] = tmp_obj
        res_pCLK[alg] = tmp_pCLK
        res_pH[alg] = tmp_pH
    # print
    if np.random.rand(1) < print_prob:
        print(f"metric {u} completed")
    return u, (res_obj, res_pCLK, res_pH)


def _helper_policies_sample(
    k, lam, Ob, U0,
    alpha_H, alpha_NH, beta, c,
    tr_no_samples, tr_eps, tr_max_iter, 
    ev_no_samples, ev_eps, ev_max_iter, 
    niter_alt,
    include_bounded=False
):
    """
    for given parameter configuration
    computes policies and their respective objective values
    vectorized over all users
    """
    res_grad_dict = {}
    rsource_dict = {}
    for user in tqdm(U0.columns):
        u, pi, r = grad_sampl_policy(
            u=user, k=k, Ob=Ob, U0=U0, 
            no_samples=tr_no_samples, 
            alpha_H=alpha_H, alpha_NH=alpha_NH, 
            beta=beta, c=c, lam=lam, 
            eps=tr_eps, max_iter=tr_max_iter, version='default')
        res_grad_dict[u] = pi
        rsource_dict[u] = r
    # ------------------------
    # alternating optimization
    with Pool() as pool:
        out = pool.map(partial(
                alt_policy, 
                k=k, Ob=Ob, U0=U0, 
                no_samples=tr_no_samples, 
                alpha_H=alpha_H, alpha_NH=alpha_NH, 
                beta=beta, c=c, 
                eps=tr_eps, max_iter=tr_max_iter,
                nsteps=niter_alt
                ), U0.columns)
    res_alt_dict = {u: pi for (u, pi) in out}
    # -------------------------
    # U0
    with Pool() as pool:
        out = pool.map(partial(
            u0topk_policy,
            k=k, Ob=Ob, U0=U0
        ), U0.columns)
    res_u0_dict = {u: pi for (u, pi) in out}
    # -------------------------
    # Unif
    with Pool() as pool:
        out = pool.map(partial(
            unif_policy,
            k=k, Ob=Ob,
        ), U0.columns)
    res_unif_dict = {u: pi for (u, pi) in out}
    res_policy = {
        'gradk' : res_grad_dict,
        'alt' : res_alt_dict,
        'u0' : res_u0_dict,
        'unif' : res_unif_dict
    }
    met_columns = ['gradk','alt','u0','unif']
    if include_bounded:
        res_policy['gradk_bound'] = res_gradBD_dict
        met_columns = ['gradk','gradk_bound','alt','u0','unif']
    # compute metrics
    res_obj = pd.DataFrame(
        np.zeros((U0.shape[1], len(met_columns))),
        index=U0.columns,
        columns=met_columns
    )
    res_pCLK = pd.DataFrame(
        np.zeros((U0.shape[1], len(met_columns))),
        index=U0.columns,
        columns=met_columns
    )
    res_pH = pd.DataFrame(
        np.zeros((U0.shape[1], len(met_columns))),
        index=U0.columns,
        columns=met_columns
    )
    n_H = Ob.filter(like='objH').shape[1]
    n_NH = Ob.filter(like='objNH').shape[1]
    HM = list(range(n_NH, n_NH+n_H))
    E_set = None
    if include_bounded:
        E_set = list(itertools.combinations(range(n_NH),k))
    with Pool() as pool:
        out = pool.map(partial(
            _helper_metric,
            Ob=Ob, U0=U0, 
            no_samples=ev_no_samples, HM=HM,
            alpha_H=alpha_H, alpha_NH=alpha_NH, 
            beta=beta, c=c, lam=lam, 
            eps=ev_eps, max_iter=ev_max_iter,
            res_policy=res_policy, E_set=E_set,
            rsource_dict=rsource_dict,
        ), U0.columns)
    for u, (tmp_obj, tmp_pCLK, tmp_pH) in out:
        for alg in res_policy.keys():
            res_obj[alg].loc[u] = tmp_obj[alg]
            res_pCLK[alg].loc[u] = tmp_pCLK[alg]
            res_pH[alg].loc[u] = tmp_pH[alg]
    return res_policy, res_obj, res_pCLK, res_pH
