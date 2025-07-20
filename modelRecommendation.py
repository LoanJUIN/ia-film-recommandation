import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class LightMovieRecommendationSystem:
    def __init__(self, model_type='linear_regression'):

        self.model_type = model_type
        self.genre_encoder = LabelEncoder()

        # Choix du modèle
        if model_type == 'linear_regression':
            self.model = LinearRegression()
        elif model_type == 'decision_tree':
            self.model = DecisionTreeRegressor()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=10,
                min_samples_leaf=4
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor()
        elif model_type == 'mlp':
            self.model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
        else:
            raise ValueError(f"Model type '{model_type}' not recognized.")

    def prepare_data(self, df, sample_size=None):
        # Échantillonnage si nécessaire
        if sample_size:
            df = df.sample(n=sample_size)

        # Encodage des genres
        df['genres_encoded'] = self.genre_encoder.fit_transform(df['genres'])

        return df

    def train_recommendation_model(self, df_combined, df_separate):

        # Préparation des données
        df_processed = self.prepare_data(df_separate, sample_size=10000)

        # Préparation des features d'entrées et de sorties
        X = df_processed[['userId', 'movieId', 'genres_encoded']]
        y = df_processed['rating']

        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        # Conversion des features pour les modèles nécessitant un scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if self.model_type in ['linear_regression', 'mlp', 'gradient_boosting', 'random_forest', 'xgboost', 'lightgbm']:
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
        elif self.model_type == 'decision_tree':
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
        else:
            raise ValueError("Unsupported model type.")

        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, label="Prédictions vs Réel")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Parfaite correspondance")
        plt.xlabel("Valeurs réelles (y_test)")
        plt.ylabel("Prédictions (y_pred)")
        plt.title("Prédictions vs Valeurs Réelles")
        plt.legend()
        plt.grid(True)
        plt.show()

    def recommend_by_genre(self, df, user_id):
        """
        Recommande un film basé sur le genre préféré de l'utilisateur

        """
        # Filtrer les films vus par l'utilisateur
        user_ratings = df[df['userId'] == user_id]

        if user_ratings.empty:
            return None, None

        # Trouver le genre préféré de l'utilisateur
        top_genre = user_ratings.groupby('genres')['rating'].mean().idxmax()

        # Filtrer les films du genre préféré que l'utilisateur n'a pas encore vus
        unseen_movies = df[(df['genres'] == top_genre) & (~df['movieId'].isin(user_ratings['movieId']))]

        if unseen_movies.empty:
            return None, top_genre

        # Recommander le film le mieux noté dans ce genre
        recommended_movie = unseen_movies.loc[unseen_movies['rating'].idxmax()]

        return recommended_movie['title'], top_genre
