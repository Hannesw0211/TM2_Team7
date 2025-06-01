"""
custom_models.py

algorithms: NMF, PMF

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
from collections import defaultdict
from lenskit.algorithms import Recommender


class NMFRecommender: #Non-Negative Matrix Factorization
    def __init__(self, n_components=20, random_state=42, max_iter=500): #n_components may be changed according to dataset size
        # Erstellt die NMF-Engine von scikit-learn
        # n_components: Anzahl der latenten Faktoren, die gelernt werden sollen (hyperparameter)
        # init='nndsvda': Initialisierungsmethode für die Matrizen W und H
        # random_state: für Reproduzierbarkeit der Ergebnisse
        # max_iter: Maximale Anzahl von Iterationen für den Optimierungsalgorithmus
        self.n_components = n_components
        self.model = NMF(n_components=n_components, init='nndsvda', random_state=random_state, max_iter=max_iter) #NMF-Modell von scikit-learn instanziiert
        self.user_map = {} #Zuordnung Benutzer-IDs zu Integer-Indizes (Initialisierung: noch leer)
        self.item_map = {} #Zuordnung von Artikel-IDs zu Integer-Indizes
        self.user_inv = {} #inverese Zuordnung
        self.item_inv = {}


    def fit(self, ratings): #Trains the NMF model on the provided ratings data  # ratings: Pandas DataFrame, das Spalten wie 'user', 'item' und 'rating' enthält
        # Map user/item IDs to indices (mit Werten)
        users = ratings['user'].unique() # einzigartigen Benutzer- und Artikel-IDs
        items = ratings['item'].unique()
        self.user_map = {u: i for i, u in enumerate(users)} #ID-Mapping
        self.item_map = {i: j for j, i in enumerate(items)}
        self.user_inv = {i: u for u, i in self.user_map.items()} #inverse
        self.item_inv = {j: i for i, j in self.item_map.items()}

        # Create user-item matrix R
        R = np.zeros((len(users), len(items))) # Dimensionen: (Anzahl der einzigartigen Benutzer) x (Anzahl der einzigartigen Artikel)
        for _, row in ratings.iterrows():
            R[self.user_map[row['user']], self.item_map[row['item']]] = row['rating'] # Matrix mit den expliziten Bewertungen füllen

        # Apply NMF to R: finden von optimalen Benutzer-Faktor-Matrix(W) und Artikel-Faktor-Matrix (H)
        self.W = self.model.fit_transform(R)
        self.H = self.model.components_

        return self

    def predict_for_user(self, user, items, ratings=None): #Predicts ratings for a list of items for a specific user
        # Überprüft, ob der Benutzer im Modell bekannt ist: wenn nicht, werden NaN-Werte zurückgegeben
        if user not in self.user_map:
            return pd.Series(np.nan, index=items)
        uid = self.user_map[user] # holt den internen Index des Benutzers
        preds = {}  # Dictionary zum Speichern der Vorhersagen
        # Überprüft, ob Artikel im Modell bekannt ist
        for item in items:
            if item in self.item_map:
                iid = self.item_map[item] # Index des Artikels holen
                preds[item] = np.dot(self.W[uid], self.H[:, iid]) # Berechneung der vorhergesagten Bewertung durch Skalarprodukt von Benutzer-Faktor-Vektors (der entsprechenden Zeile aus self.W)
                # & Artikel-Faktor-Vektors (der entsprechenden Spalte aus self.H) (ist die Rekonstruktion eines EIntrags der ursprünglichen Matrix)
            else: #atrikel unbekannt: NaN gesetzt
                preds[item] = np.nan
        return pd.Series(preds) # Vorhersagen als Pandas Series

    def recommend(self, user, n=10, candidates=None, ratings=None): #Generates top-N recommendations for a specific user  # n: Anzahl der Top-Empfehlungen  #canidates: optionale Liste von Artikeln, aus denen Empfehlungen ausgewählt werden sollen

        if candidates is None:
            candidates = list(self.item_map.keys()) #alle im Modell bekannten Artikel als Kandidaten betrachtet

        scores = self.predict_for_user(user, candidates) # um die vorhergesagten Bewertungen für alle Kandidatenartikel für den gegebenen Benutzer zu erhalten
        scores = scores.dropna() # Artikel, für die keine gültige Vorhersage gemacht werden konnte, entfernen
        top_scores = scores.nlargest(n) # Artikel mit den höchsten vorhergesagten Bewertungen auswählen

        top_scores.name = 'score'
        return top_scores.reset_index().rename(columns={'index': 'item'}) #Ergebnisse werden in Pandas DataFrame umgewandelt ( Spalten = 'Artikel-ID, 'score' = vorhergesagte Bewertung)



class PMFRecommender:

    def __init__(self, n_factors=20, lr=0.015, reg=0.02, n_iters=100, #Initialisiert den PMF-Recommender
                 batch_size=10000, random_state=42):
        # n_factors (int): Anzahl der latenten Faktoren, die das Modell lernen soll (Hyperparameter)
        # lr (float): Lernrate (Learning Rate) für den Gradientenabstieg. Bestimmt die Schrittgröße der Parameterupdates
        # reg (float): Regularisierungsstärke (Regularization) zur Vermeidung von Overfitting
        # n_iters (int): Maximale Anzahl von Iterationen (Epochen) über den gesamten Datensatz während des Trainings
        # batch_size (int): Größe der Mini-Batches, die pro Update-Schritt verarbeitet werden
        #random_state (int): Startwert für den Zufallszahlengenerator zur Reproduzierbarkeit der Initialisierung der latenten Faktoren
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.random_state = random_state
        # Dictionaries zur Abbildung von Benutzer- und Artikel-IDs auf interne Indizes
        self.u_map = {}
        self.i_map = {}
        self.u_inv = {}
        self.i_inv = {}

    def fit(self, ratings): # trainiert das PMF-Modell auf den bereitgestellten Bewertungsdaten

        # Map IDs to indices
        users = ratings['user'].unique()
        items = ratings['item'].unique()
        self.u_map = {u: i for i, u in enumerate(users)}
        self.i_map = {i: j for j, i in enumerate(items)}
        self.u_inv = {i: u for u, i in self.u_map.items()}
        self.i_inv = {j: i for i, j in self.i_map.items()}

        n_u, n_i = len(users), len(items)

        # Initialisierung der latenten Faktor-Matrizen (kleine zufällige Werte)
        # P für Benutzer (users x n_factors) und Q für Artikel (items x n_factors)
        rng = np.random.RandomState(self.random_state)
        P = 0.1 * rng.randn(n_u, self.n_factors).astype(np.float32)
        Q = 0.1 * rng.randn(n_i, self.n_factors).astype(np.float32)

        # Pre-compute arrays: externen IDs einmalig in interne Indizes umgewandelt und als numpy arrays gespeichert
        user_idx = ratings['user'].map(self.u_map).values.astype(np.int32)
        item_idx = ratings['item'].map(self.i_map).values.astype(np.int32)
        rating_vals = ratings['rating'].values.astype(np.float32)
        n_ratings = len(ratings)

        # Training loop with batch processing: iteriert über eine festgelegte Anzahl von Epochen
        for epoch in range(self.n_iters):
            # Shuffle all data: Für jede Epoche werden die Reihenfolge der Ratings zufällig gemischt (konvergenz von sgd und lokale minima vermeiden)
            shuffle_idx = rng.permutation(n_ratings)

            # Process in batches: gemischten Ratings werden in Mini-Batches aufgeteilt
            n_batches = (n_ratings + self.batch_size - 1) // self.batch_size

            for batch_start in range(0, n_ratings, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_ratings)
                batch_indices = shuffle_idx[batch_start:batch_end]

                # Extract batch data: Extrahieren der Benutzer-, Artikel- und Bewertungsdaten
                batch_users = user_idx[batch_indices]
                batch_items = item_idx[batch_indices]
                batch_ratings = rating_vals[batch_indices]

                # _sgd_batch aufrufen, die die eigentlichen Gradientenabstiegs-Updates für aktuellen Mini-Batch durchführt
                self._sgd_batch(P, Q, batch_users, batch_items, batch_ratings)

        # optimierten Benutzer- (P) und Artikel-Faktoren (Q) gespeichert
        self.P, self.Q = P, Q
        return self

    def _sgd_batch(self, P, Q, batch_users, batch_items, batch_ratings):
        # Aktualisiert die latenten Faktoren P und Q basierend auf den Vorhersagefehlern

        # Iteriert über jedes Rating im aktuellen Mini-Batch
        batch_size = len(batch_users)
        for i in range(batch_size):
            u = batch_users[i]    # Aktueller Benutzer-Index
            it = batch_items[i]   # Aktueller Artikel-Index
            r = batch_ratings[i]  # Aktueller Bewertungswert

            # vorhergesagte Bewertung für das aktuelle Paar (u, it) als Skalarprodukt der latenten Faktoren von Benutzer und Artikel
            pred = np.dot(P[u], Q[it])
            # Vorhersagefehler
            err = r - pred

            # Store current factors (needed for simultaneous update)
            Pu = P[u].copy()
            Qi = Q[it].copy()

            # Aktualisierung der latenten Faktoren von Benutzer und Artikel
            # Die Faktoren werden proportional zum Fehler (err) und zur Lernrate (self.lr) angepasst
            P[u] += self.lr * (err * Qi - self.reg * Pu)
            Q[it] += self.lr * (err * Pu - self.reg * Qi)


    def predict_for_user(self, user, items, ratings=None): # Berechnet die vorhergesagten Bewertungen für eine Liste von Artikeln für einen bestimmten Benutzer

        #Berechnet die vorhergesagten Bewertungen für eine Liste von Artikeln für einen bestimmten Benutzer.

        # Benutzer nicht im Modell bekannt: NaN
        if user not in self.u_map:
            return pd.Series(np.nan, index=items)

        u = self.u_map[user] # Index Benutzer holen

        preds = {}

        # Berechnung von Vorhersagen für bekannte Artikel
        valid_items = [it for it in items if it in self.i_map]
        if valid_items:
            item_indices = [self.i_map[it] for it in valid_items]
            # Berechnet das Skalarprodukt des Benutzer-Faktor-Vektors (P[u]) mit allen gültigen Artikel-Faktor-Vektoren (Q[item_indices])
            scores = np.dot(self.P[u], self.Q[item_indices].T)

            # Ordnet die berechneten Scores den ursprünglichen Artikel-IDs zu
            for it, score in zip(valid_items, scores):
                preds[it] = score

        # Handle invalid items: Setzt NaN für Artikel, die im Modell unbekannt sind
        for it in items:
            if it not in preds:
                preds[it] = np.nan

        return pd.Series(preds) # Vorhersagen als Pandas Series

    def recommend(self, user, n=10, candidates=None, ratings=None): #Generiert Top-N Empfehlungen für einen bestimmten Benutzer
        # Wenn keine Kandidaten angegeben sind, werden alle im Modell bekannten Artikel als Kandidaten betrachtet
        if candidates is None:
            candidates = list(self.i_map.keys())

        # Ruft die vorhergesagten Bewertungen für alle Kandidatenartikel für den gegebenen Benutzer ab.
        scores = self.predict_for_user(user, candidates).dropna()
        # Wählt die Top-N Artikel mit den höchsten vorhergesagten Bewertungen aus.
        top = scores.nlargest(n).reset_index()
        top.columns = ['item', 'score']
        return top




# @derivable() # Uncomment if you intend to use LensKit's derivable decorator for parameter saving
class SlopeOneRecommender:
    """
    A recommender system based on the Slope One algorithm.

    Slope One is a simple yet effective item-based collaborative filtering
    algorithm that predicts a user's rating for an item based on the average
    difference in ratings of other items rated by that user.
    """

    # Static methods for defaultdict factories to ensure picklability
    @staticmethod
    def _float_defaultdict_factory():
        return defaultdict(float)

    @staticmethod
    def _int_defaultdict_factory():
        return defaultdict(int)

    def __init__(self):
        self.item_map = {}  # Maps original item IDs to internal integer indices
        self.item_inv = {}  # Maps internal integer indices back to original item IDs
        self.user_map = {}  # Maps original user IDs to internal integer indices
        self.user_inv = {}  # Maps internal integer indices back to original user IDs

        # Use the static methods as factories for defaultdict
        self.item_ratings_sum_diff = defaultdict(SlopeOneRecommender._float_defaultdict_factory)
        self.item_freq_diff = defaultdict(SlopeOneRecommender._int_defaultdict_factory)

        # Stores all ratings for each user, useful for prediction logic
        # Structure: self.user_ratings[user_id] = {item_id_1: rating_1, item_id_2: rating_2}
        self.user_ratings = defaultdict(dict)

    def fit(self, ratings_df, **kwargs):  # Added **kwargs for Recommender.adapt compatibility
        """
        Trains the Slope One model.
        This involves calculating the average rating differences between all pairs of items
        that were co-rated by users.

        Args:
            ratings_df (pd.DataFrame): DataFrame with 'user', 'item', 'rating' columns.
        """
        users = ratings_df['user'].unique()
        items = ratings_df['item'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}
        self.user_inv = {i: u for u, i in self.user_map.items()}
        self.item_inv = {j: i for i, j in self.item_map.items()}

        self.user_ratings.clear()
        self.item_ratings_sum_diff.clear()
        self.item_freq_diff.clear()

        for _, row in ratings_df.iterrows():
            user_id, item_id, rating_val = row['user'], row['item'], float(row['rating'])
            self.user_ratings[user_id][item_id] = rating_val

        for user_id, current_user_ratings_dict in self.user_ratings.items():
            rated_item_ids = list(current_user_ratings_dict.keys())
            for i in range(len(rated_item_ids)):
                for j in range(i + 1, len(rated_item_ids)):
                    item1_id = rated_item_ids[i]
                    item2_id = rated_item_ids[j]
                    rating1 = current_user_ratings_dict[item1_id]
                    rating2 = current_user_ratings_dict[item2_id]

                    self.item_ratings_sum_diff[item1_id][item2_id] += (rating1 - rating2)
                    self.item_freq_diff[item1_id][item2_id] += 1

                    self.item_ratings_sum_diff[item2_id][item1_id] += (rating2 - rating1)
                    self.item_freq_diff[item2_id][item1_id] += 1
        return self

    def predict_for_user(self, user, items_to_predict, ratings=None):
        """
        Predicts ratings for a list of items for a specific user.

        Args:
            user: The ID of the user for whom to predict ratings.
            items_to_predict (list): A list of item IDs for which to predict ratings.
            ratings (pd.Series, optional): The user's historical ratings.

        Returns:
            pd.Series: Predicted ratings, indexed by item ID. NaN for items
                       that cannot be predicted or are unknown.
        """
        predictions = {}

        if user not in self.user_ratings:
            return pd.Series(np.nan, index=items_to_predict)

        user_rated_items_dict = self.user_ratings.get(user, {})

        for target_item_id in items_to_predict:
            if target_item_id not in self.item_map:
                predictions[target_item_id] = np.nan
                continue

            numerator = 0.0
            denominator = 0.0

            for rated_item_id, user_rating_for_rated_item in user_rated_items_dict.items():
                if target_item_id in self.item_freq_diff and \
                        rated_item_id in self.item_freq_diff[target_item_id]:

                    freq_co_occurrence = self.item_freq_diff[target_item_id][rated_item_id]

                    if freq_co_occurrence > 0:
                        avg_diff = self.item_ratings_sum_diff[target_item_id][rated_item_id] / freq_co_occurrence
                        numerator += (user_rating_for_rated_item + avg_diff) * freq_co_occurrence
                        denominator += freq_co_occurrence

            if denominator > 0:
                predictions[target_item_id] = numerator / denominator
            else:
                predictions[target_item_id] = np.nan

        return pd.Series(predictions).reindex(items_to_predict)

    def recommend(self, user, n=10, candidates=None, ratings=None):
        """
        Generates top-N recommendations for a specific user.

        Args:
            user: The ID of the user for whom to generate recommendations.
            n (int): The number of recommendations to return.
            candidates (list, optional): A list of candidate item IDs to score.
            ratings (pd.Series, optional): The user's historical ratings.

        Returns:
            pd.DataFrame: A DataFrame with 'item' and 'score' columns.
        """
        if user not in self.user_ratings and candidates is None:
            return pd.DataFrame({'item': [], 'score': []})

        user_rated_item_ids = set(self.user_ratings.get(user, {}).keys())

        if candidates is None:
            all_model_items = list(self.item_map.keys())
            candidates_to_score = [item_id for item_id in all_model_items if item_id not in user_rated_item_ids]
        else:
            candidates_to_score = [item_id for item_id in candidates if item_id not in user_rated_item_ids]

        if not candidates_to_score:
            return pd.DataFrame({'item': [], 'score': []})

        scores_series = self.predict_for_user(user, candidates_to_score, ratings=ratings)
        scores_series = scores_series.dropna()

        if scores_series.empty:
            return pd.DataFrame({'item': [], 'score': []})

        top_n_items = scores_series.nlargest(n)

        recommendations_df = top_n_items.reset_index()
        recommendations_df.columns = ['item', 'score']

        return recommendations_df

import pandas as pd
import numpy as np
from lenskit import topn
from sklearn.metrics.pairwise import cosine_similarity

class UUCustom: # User-Item Scoring per Cosine Similarity + Ranking
    def __init__(self, param=0, top_k=None):
        self.param = param
        self.top_k = top_k  # Optional: Nur die k ähnlichsten Nutzer verwenden

    def fit(self, train):
        self.train = train.copy()
        self.user_item_matrix = train.pivot(index='user', columns='item', values='rating').fillna(0)
        self.user_sim = cosine_similarity(self.user_item_matrix)
        self.user_sim_df = pd.DataFrame(self.user_sim,
                                        index=self.user_item_matrix.index,
                                        columns=self.user_item_matrix.index)

    def predict_for_user(self, user, items, ratings=None):
        if user not in self.user_item_matrix.index:
            return pd.Series(0.0, index=items)

        scores = {}
        user_sims = self.user_sim_df.loc[user]

        # Optional: nur Top-K ähnlichste Nutzer verwenden (außer sich selbst)
        if self.top_k is not None:
            top_users = user_sims.drop(user).nlargest(self.top_k).index
            user_sims = user_sims[top_users]

        for item in items:
            if item not in self.user_item_matrix.columns:
                scores[item] = 0.0
                continue

            ratings_for_item = self.user_item_matrix[item]

            # Nur Nutzer betrachten, die das Item bewertet haben
            valid_users = ratings_for_item[ratings_for_item > 0].index
            valid_users = valid_users.intersection(user_sims.index)

            if len(valid_users) == 0:
                scores[item] = 0.0
                continue

            sims = user_sims[valid_users]
            ratings = ratings_for_item[valid_users]

            sim_sum = sims.sum()
            score = np.dot(sims, ratings)

            scores[item] = score / sim_sum if sim_sum > 0 else 0.0

        return pd.Series(scores)

    def recommend(self, user, k=10):
        if user not in self.user_item_matrix.index:
            return []

        all_items = set(self.user_item_matrix.columns)
        known_items = set(self.user_item_matrix.columns[self.user_item_matrix.loc[user] > 0])
        unknown_items = list(all_items - known_items)

        scores = self.predict_for_user(user, unknown_items)
        return scores.sort_values(ascending=False).head(k).index.tolist()