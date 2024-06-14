import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import scipy.sparse.linalg
# import sklearn.decomposition
from mf import MF

def RMSE(A, B):
    return np.sqrt(np.mean(np.square(
        A.to_numpy() - B.to_numpy()
    )))

if __name__ == "__main__":
    # params
    #genre = "Action"
    #genre = "Adventure"
    #genre = "Comedy"
    #genre = "Fantasy"
    genre = "Sci-Fi"
    use_bias=True
    save_name_U  = f"../data/{genre}_bias{use_bias}_U_mf_df.csv"
    save_name_Ob = f"../data/{genre}_bias{use_bias}_Ob_mf_df.csv"
    save_name_top= f"../data/movie_{genre}_bias{use_bias}_tophits.csv"


    print(f"genre: {genre}")
    # Load ratings dataset
    rating = pd.read_csv('ml-25m/ratings.csv')
    # Load movies dataset
    movie = pd.read_csv('ml-25m/movies.csv')
    # Link to IMDB
    link = pd.read_csv("ml-25m/links.csv") 

    # IMDB_parental
    imdb = pd.read_csv("IMDB_parental_detail_guide.csv")
    imdb['tconst'] = imdb['tconst'].apply(lambda x: int(x.replace('tt','')))

    # join datasets
    df = pd.merge(rating, movie, on='movieId')
    df = pd.merge(df, link, on='movieId')
    df = pd.merge(df, imdb, left_on='imdbId', right_on='tconst')

    # some analysis
    # let's take severe category 4 as ground truth
    print(f"total number of user-movie ratings: {df.shape[0]}")
    for category in ["sex", "violence", "profanity", "drug", "intense"]:
        print(f"total number of moview with {category}_code=4: {df[f'{category}_code'].loc[df[f'{category}_code']==4].shape[0]}")

    # filter on genre
    df_action = df.loc[df['genres_x'].apply(lambda x: genre in x)]
    for category in ["sex", "violence", "profanity", "drug", "intense"]:
        print(f"total number of moview with {category}_code=4: {df_action[f'{category}_code'].loc[df_action[f'{category}_code']==4].shape[0]}")
    print(df_action.shape)

    # our Harm column
    df_action['severe_any'] = (df_action['sex_code'] == 4) | \
        (df_action['violence_code'] == 4) | \
        (df_action['profanity_code'] == 4) | \
        (df_action['drug_code'] == 4) | \
        (df_action['intense_code'] == 4)
    print(f"action movies have severe: {np.sum(df_action['severe_any']==True)/df_action.shape[0]}")

    action_lookup = df_action[['movieId', 'severe_any']].drop_duplicates()
    action_severe_set = set(df_action['movieId'].loc[df_action['severe_any']==True])

    # compile ratings matrix
    R_df =df_action.pivot(index='userId', columns='movieId', values='rating').fillna(0) 
    # filter to top 100 movies, and top 1000 users (most num ratings)
    movie_sort = R_df.astype(bool).sum(axis=0).sort_values(ascending=False, inplace=False)
    R_df_filter = R_df[movie_sort.index[:100]]
    user_sort = R_df_filter.astype(bool).sum(axis=1).sort_values(ascending=False, inplace=False)
    R_df_filter = R_df_filter.loc[user_sort.index[:1000]]

    num_severe_filter = len(np.intersect1d(
        R_df_filter.columns.to_numpy(),
        action_lookup.loc[action_lookup['severe_any']==True, "movieId"].to_numpy()
        ))
    print(f"post filter, we have {num_severe_filter} / {R_df_filter.shape[1]} severe movies")

    # movie info on filetered set: action, and movies with many ratings
    movie_filter_set = set(R_df_filter.columns)
    movie_filter_idx = movie['movieId'].apply(
        lambda x: x in movie_filter_set)
    movie_filter = movie.loc[movie_filter_idx]
    movie_filter = movie_filter.merge(action_lookup, on='movieId')

    R = R_df_filter.to_numpy() 

    k=10
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    mf = MF(R_demeaned, K=k, alpha=0.01, beta=0.01, 
            iterations=100, use_bias=use_bias)
    mf.train()
    pred = mf.full_matrix()
    rmse = RMSE(pd.DataFrame(pred), pd.DataFrame(R_demeaned))
    print(f"MF RMSE with {k} dim: {rmse}")

    # modify for saving
    if use_bias:
        U = np.hstack(
            (mf.P, mf.b_u.reshape(-1,1), np.ones((1000,1)))
        ).T
        Ob = np.hstack(
            (mf.Q, np.ones((100,1)), mf.b_i.reshape(-1,1))
        ).T
    else:
        U = mf.P.T
        Ob = mf.Q.T
    U_df = pd.DataFrame(
        U, 
        columns=["user"+str(x) for x in list(R_df_filter.index)]
    ) 
    Ob_df = pd.DataFrame(
        Ob,
        columns=["objH"+str(x) if x in action_severe_set \
            else "objNH"+str(x) for x in list(R_df_filter.columns)]
    )
    assert np.allclose(pred, U_df.T @ Ob_df)

    U_df.to_csv(save_name_U)
    Ob_df.to_csv(save_name_Ob)
    movie_filter.to_csv(save_name_top)