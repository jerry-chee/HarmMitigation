from ML_IMDB import *
from real_analysis import plot_ml_data 

if __name__ == "__main__":
    # parameters
    k   = 1          # rec set size
    lam = 100        # harm penalty
    alpha_H  = 0.5   # H addictiveness (default)
    alpha_NH = 0.25  # NH addictiveness (default)
    beta   = 0.2     # influence of initial user state (default)
    nsteps_usersteady = 10    # number of steps to run F() for fixed point
    niter_alt = 10
    n_u = 10
    #n_u = 100

    # setup data
    #genre = "Action"
    #genre = "Adventure"
    #genre = "Comedy"
    #genre = "Fantasy"
    genre = "Sci-Fi"
    c = 6 # influence of recs (smaller is more influence)
    Ob_rescale = 1 #NMF action default

    U0_filter, Ob_filter, _, _, E_vec = real_setup(
        k, n_u, None, None, Ob_rescale, 
        U0_path=f"data/{genre}_U_mf_df.csv",
        Ob_path=f"data/{genre}_Ob_mf_df.csv",
        user_idx=None, user_tot=None)

    # # learn policies
    init="rand"

    res_alt = policy.alt_policy(k, init, 
            U0_filter, Ob_filter, E_vec, alpha_H, alpha_NH, beta, c, niter_alt)
    res_unif = policy.unif_policy(U0_filter, E_vec)
    Ulim_unif = utils.user_steady(U0_filter, Ob_filter, res_unif, 
            alpha_H, alpha_NH, beta, c, 10)
    Slim_unif = utils.compute_pref_score(Ulim_unif, Ob_filter)
    Ulim_alt = utils.user_steady(U0_filter, Ob_filter, res_alt, 
            alpha_H, alpha_NH, beta, c, 10)
    Slim_alt = utils.compute_pref_score(Ulim_alt, Ob_filter)
    unif_pCLK = utils.prob_CLK(Slim_unif, res_unif, c)
    unif_pH = utils.prob_H(Slim_unif, res_unif, c)
    alt_pCLK = utils.prob_CLK(Slim_alt, res_alt, c)
    alt_pH = utils.prob_H(Slim_alt, res_alt, c)
    print("steady state alt")
    print(alt_pCLK.mean(axis=1))
    print(alt_pH.mean(axis=1))
    print("steady state unif")
    print(unif_pCLK.mean(axis=1))
    print(unif_pH.mean(axis=1))