"""
NMFRecommender:
    A recommender system based on Non-Negative Matrix Factorization using scikit-learn
    Wraps the sklearn NMF to handle user/item mapping and provide prediction/recommendation methods

    WHY CUSTOM_MODELS CLASS WAS ADDED:
        - is needed because sklearn.decomposition.NMF only performs the matrix factorization (R≈W×H); it gives you the W (user factors) and H (item factors) matrices
        - sklearn.decomposition.NMF doesn't provide methods for:
            - Calculating the predicted rating for a specific user-item pair (predict_for_user)
            - Generating a list of top-N recommendations for a user (recommend)
        - NMFRecommender class adds these crucial recommender system methods by utilizing the learned W and H matrices
        - LensKit has a specific interface for recommender algorithms: 
            - algorithms are expected to have methods like fit (to train on data) and recommend or predict (to generate recommendations or predictions)
        - By creating the NMFRecommender class with fit, predict_for_user, and recommend methods,
          it makes the scikit-learn NMF algorithm compatible with LensKit's evaluation pipelines
        - The Recommender.adapt(fittable) call in gridsearch.py and main.py is specifically designed to work with objects that have these methods.
"""
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd

class NMFRecommender: #Non-Negative Matrix Factorization
    def __init__(self, n_components=20, random_state=42, max_iter=500): #n_components may be changed according to dataset size
        self.n_components = n_components
        self.model = NMF(n_components=n_components, init='nndsvda', random_state=random_state, max_iter=max_iter) #creates the actual NMF engine from the sklearn library
        self.user_map = {}
        self.item_map = {}
        self.user_inv = {}
        self.item_inv = {}


    def fit(self, ratings): #Trains the NMF model on the provided ratings data
        # Map user/item IDs to indices
        users = ratings['user'].unique()
        items = ratings['item'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.user_inv = {i: u for u, i in self.user_map.items()}
        self.item_inv = {j: i for i, j in self.item_map.items()}

        # Create user-item matrix
        R = np.zeros((len(users), len(items)))
        for _, row in ratings.iterrows():
            R[self.user_map[row['user']], self.item_map[row['item']]] = row['rating']

        # Apply NMF
        self.W = self.model.fit_transform(R)
        self.H = self.model.components_

        return self

    def predict_for_user(self, user, items, ratings=None): #Predicts ratings for a list of items for a specific user
        if user not in self.user_map:
            return pd.Series(np.nan, index=items)
        uid = self.user_map[user]
        preds = {}
        for item in items:
            if item in self.item_map:
                iid = self.item_map[item]
                preds[item] = np.dot(self.W[uid], self.H[:, iid])
            else:
                preds[item] = np.nan
        return pd.Series(preds)

    def recommend(self, user, n=10, candidates=None, ratings=None): #Generates top-N recommendations for a specific user
        if candidates is None:
            candidates = list(self.item_map.keys())

        scores = self.predict_for_user(user, candidates)
        scores = scores.dropna()
        top_scores = scores.nlargest(n)

        # Setze explizit den Spaltennamen
        top_scores.name = 'score'
        return top_scores.reset_index().rename(columns={'index': 'item'})



