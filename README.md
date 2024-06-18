# Harm Mitigation in Recommender Systems under User Preference Dynamics KDD'24

This repository contains code for the KDD'24 paper: https://arxiv.org/abs/2406.09882v1.

## Data
Download the IMDB parental guidelines and MovieLens25m datasets from:
- https://www.kaggle.com/datasets/barryhaworth/imdb-parental-guide
- https://grouplens.org/datasets/movielens/25m/

## Learning Embeddings
Run `MovieLens/explore_MF.py` to join IMDB and ML25m data, filter on genre, and create learned Matrix Factorization embeddings. At the top of the script specify the genre, saved output locations. MF bias terms are learned; we incorporate these into the resulting user and item embeddings.

## Calibrating c
Run `calib_MF.py` or `calib_MF_sampl.py` with a given genre selected, at various values of `c`. Outputs pCLK, pH under Unif and Alt policies at steady state, on a set of 10 users. We select `c` to enter a regime where pH is sufficiently high and therefore harmful outcomes are not unlikely, but also where pCLK > 0.5 and therefore our policies have some impact. 

## Learning Policies
To learn the policies given a set of user and item embeddings at `U0_path` and `Ob_path` respectively, run the following code. The following simulation parameters must also be given:
- `n_u`: total number of users
- `k`: number of recommended items
- `lam`: harm penalty
- `a_H`: harmful content coefficient
- `a_NH`: non-harmful content coefficient
- `beta`: inherent profile coefficient
- `c`: multinomial selection constant
- `nsteps_usersteady`: number of iterations for fixed point
- `savename`: where to save output

**Bounded Cardinality Setting**
```
import os
import pickle
import policy
import ML_IMDB
U0_filter, Ob_filter, _, _, E_vec = ML_IMDB.real_setup(
  k=k, n_u=n_u, n_NH=None, n_H=None,
  Ob_rescale=1, 
  U0_path=U0_path, Ob_path=Ob_path,
  user_idx=None, user_tot=None, 
  )
res = policy._helper_policies(
  k=k, init="rand", lam=lam, 
  U0=U0_filter, Ob=Ob_filter, 
  E_vec=E_vec, alpha_H=a_H, alpha_NH=a_NH,
  beta=beta, c=c, 
  nsteps_usersteady=nsteps_usersteady, niter_alt=niter_alt)
results_pCLK, results_pH = policy.compute_lim_CLK_H(
  U0=U0_filter, Ob=Ob_filter, res=res, 
  alpha_H=a_H, alpha_NH=a_NH,
  beta=beta, c=c, 
  nsteps_usersteady=nsteps_usersteady)
tmp = (res, results_pCLK, results_pH)
with open(savename, "wb") as foo: pickle.dump(tmp, foo)
```

**Independent Sampling Setting**
Additional parameters:
- `tr_no_samples`: number of samples for gradient estimation for learning the policy
- `tr_eps`: tolerance of fixed point convergence for learning the policy
- `tr_max_iter`: maximum number of iterations of the fixed point convergence for learning the policy
- `ev_no_samples`: number of samples for gradient estimation for evaluating the policy
- `ev_eps`: tolerance of fixed point convergence for evaluating the policy
- `ev_max_iter`: maximum number of iterations of the fixed point convergence for evaluating the policy

 
```
import os
import pickle
imppolrt policy_sampl
import ML_IMDB_sampl
U0_filter, Ob_filter = ML_IMDB_sampl.real_setup(
  U0_path=U0_path, Ob_path=Ob_path,
  num_user=n_u, Ob_rescale=1,
  par_idx=None, par_tot=None,
)
res_policy, res_obj, res_pCLK, res_pH = policy_sample._helper_policies_sample(
  k=k, lam=lam, 
  U0=U0_filter, Ob=Ob_filter, 
  alpha_H=a_H, alpha_NH=a_NH,
  tr_no_samples=tr_no_samples, tr_eps=tr_eps, tr_max_iter=tr_max_iter,
  ev_no_samples=ev_no_samples, ev_eps=ev_eps, ev_max_iter=ev_max_iter,
  beta=beta, c="c, niter_alt=niter_alt)
tmp = (res_policy, res_obj, res_pCLK, res_pH)
with open(savename, "wb") as foo: pickle.dump(tmp, foo)
```

## Results Output
To output the results of the paper, run `real_analysis_genre_mf.py`. In the script one needs to set the folder where policies have been saved, and modify the experiment params.
