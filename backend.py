import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from surprise import KNNBasic
from surprise import NMF
from surprise import accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import Dataset, Reader
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_course_genres():
    df = pd.read_csv("course_genre.csv")
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def load_user_profiles():
    return pd.read_csv("user_profile.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        global user_ratings_glob
        user_ratings_glob = updated_ratings
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def process_dataset(raw_data):
    encoded_data = raw_data.copy()  # Make a copy of the raw dataset to avoid modifying the original data.

    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()  # Get unique user IDs from the dataset.
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}  # Create a dictionary mapping user IDs to indices.
    user_idx2id_dict = {i: x for i, x in
                        enumerate(user_list)}  # Create a dictionary mapping user indices back to original user IDs.

    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()  # Get unique item (course) IDs from the dataset.
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}  # Create a dictionary mapping item IDs to indices.
    course_idx2id_dict = {i: x for i, x in
                          enumerate(course_list)}  # Create a dictionary mapping item indices back to original item IDs.

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_idx2id_dict, course_idx2id_dict  # Return the processed dataset and dictionaries mapping indices to original IDs.


@tf.keras.utils.register_keras_serializable()
class RecommenderNet(keras.Model):
    """
        Neural network model for recommendation.

        This model learns embeddings for users and items, and computes the dot product
        of the user and item embeddings to predict ratings or preferences.

        Attributes:
        - num_users (int): Number of users.
        - num_items (int): Number of items.
        - embedding_size (int): Size of embedding vectors for users and items.
    """

    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        """
            Constructor.

            Args:
            - num_users (int): Number of users.
            - num_items (int): Number of items.
            - embedding_size (int): Size of embedding vectors for users and items.
         """
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # Define a user_embedding vector
        # Input dimension is the num_users
        # Output dimension is the embedding size
        # A name for the layer, which helps in identifying the layer within the model.

        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define a user bias layer
        # Bias is applied per user, hence output_dim is set to 1.
        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name="user_bias")

        # Define an item_embedding vector
        # Input dimension is the num_items
        # Output dimension is the embedding size
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define an item bias layer
        # Bias is applied per item, hence output_dim is set to 1.
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name="item_bias")

        self.dense1 = layers.Dense(128, activation='relu', name='dense1')
        self.dense2 = layers.Dense(64, activation='relu', name='dense2')
        self.dense3 = layers.Dense(32, activation='relu', name='dense3')
        self.final_dense = layers.Dense(embedding_size, activation='relu', name='final_dense')

    def get_config(self):
        config = super(RecommenderNet, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_size': self.embedding_size
        })
        return config

    def call(self, inputs):
        """
            Method called during model fitting.

            Args:
            - inputs (tf.Tensor): Input tensor containing user and item one-hot vectors.

            Returns:
            - tf.Tensor: Output tensor containing predictions.
        """
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_vector = self.dense1(user_vector)
        user_vector = self.dense2(user_vector)
        user_vector = self.dense3(user_vector)
        user_vector = self.final_dense(user_vector)
        # Compute the user bias
        user_bias = self.user_bias(inputs[:, 0])
        # Compute the item embedding vector
        item_vector = self.item_embedding_layer(inputs[:, 1])
        # Compute the item bias
        item_bias = self.item_bias(inputs[:, 1])
        # Compute dot product of user and item embeddings
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # Apply ReLU activation function
        return tf.nn.relu(x)


def process_dataset(raw_data):
    encoded_data = raw_data.copy()  # Make a copy of the raw dataset to avoid modifying the original data.

    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()  # Get unique user IDs from the dataset.
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}  # Create a dictionary mapping user IDs to indices.
    user_idx2id_dict = {i: x for i, x in
                        enumerate(user_list)}  # Create a dictionary mapping user indices back to original user IDs.

    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()  # Get unique item (course) IDs from the dataset.
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}  # Create a dictionary mapping item IDs to indices.
    course_idx2id_dict = {i: x for i, x in
                          enumerate(course_list)}  # Create a dictionary mapping item indices back to original item IDs.

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_idx2id_dict, user_id2idx_dict, course_idx2id_dict, course_id2idx_dict  # Return the processed dataset and dictionaries mapping indices to original IDs.


# Model training
def train(model_name, params):
    # TODO: Add model training code here
    if model_name == models[0]:
        pass
    if model_name == models[1]:
        pass
    if model_name == models[2]:
        pass
    if model_name == models[3]:
        pass
    if model_name == models[4]:
        pass
    if model_name == models[5]:
        pass
    if model_name == models[6]:
        pass
    pass


def combine_cluster_labels(user_ids, labels):
    # Convert labels to a DataFrame
    labels_df = pd.DataFrame(labels)
    # Merge user_ids DataFrame with labels DataFrame based on index
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    # Rename columns to 'user' and 'cluster'
    cluster_df.columns = ['user', 'cluster']
    return cluster_df


def pca(x_data, n_components):
    pca = PCA(n_components=n_components)
    fit_pca = pca.fit(x_data)
    print("Variance explained with {0} components:".format(n_components),
          round(sum(fit_pca.explained_variance_ratio_), 2))
    return fit_pca.transform(x_data)


def predict_ratings_for_user(model, user_id, unseen_courses, user_id2idx_dict, course_id2idx_dict, course_idx2id_dict):
    # Check if the user is new or existing
    if user_id in user_id2idx_dict:
        encoded_user_id = user_id2idx_dict[user_id]
    else:
        # For a new user, assign a temporary user ID and add it to the dictionary
        encoded_user_id = max(user_id2idx_dict.values()) + 1
        user_id2idx_dict[user_id] = encoded_user_id

    # Encode unseen courses
    encoded_courses = [course_id2idx_dict[course_id] for course_id in unseen_courses if course_id in course_id2idx_dict]

    # Prepare input tensor
    user_array = np.array([encoded_user_id] * len(encoded_courses))
    courses_array = np.array(encoded_courses)
    input_tensor = np.vstack((user_array, courses_array)).T

    # Predict ratings
    predictions = model.predict(input_tensor)

    # Decode predicted ratings back to original course IDs
    decoded_predictions = [(course_idx2id_dict[course_idx], pred[0]) for course_idx, pred in
                           zip(encoded_courses, predictions)]

    # Create DataFrame for better readability
    predictions_df = pd.DataFrame(decoded_predictions, columns=["item", "rating"])

    return predictions_df


def load_nn_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'RecommenderNet': RecommenderNet})
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"Failed to load model: {e}")
        return None


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    if "profile_sim_threshold" in params:
        profile_sim_threshold = params["profile_sim_threshold"]
    if "popularity" in params:
        popularity = params['popularity']
    if "cluster_no" in params:
        cluster_no = params['cluster_no']
    if "feature_no" in params:
        feature_no = params['feature_no']
    if "nn_threshold" in params:
        nn_threshold = params["nn_threshold"]
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        # TODO: Add prediction model code here
        if model_name == models[1]:
            course_genres_df = load_course_genres()
            ratings_df = load_ratings()
            all_courses = set(course_genres_df['COURSE_ID'].values)
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_courses = user_ratings['item'].to_list()
            # Trying to create a user vector here
            courses_of_user = course_genres_df[course_genres_df['COURSE_ID'].isin(enrolled_courses)]
            user_vector = courses_of_user.drop(columns=['COURSE_ID', 'TITLE']).sum(axis=0)
            user_vector_df = user_vector * 3
            user_vector_df = pd.DataFrame(user_vector_df).T
            # Finishing with user vector
            test_user_vector = user_vector_df.iloc[0, :].values
            unknown_courses = all_courses.difference(enrolled_courses)
            unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
            unknown_course_ids = unknown_course_df['COURSE_ID'].values
            recommendation_scores = np.dot(unknown_course_df.iloc[:, 2:].values, test_user_vector)
            for i in range(0, len(unknown_course_ids)):
                score = recommendation_scores[i]
                if score >= profile_sim_threshold:
                    users.append(user_id)
                    courses.append(unknown_course_ids[i])
                    scores.append(recommendation_scores[i])
        if model_name == models[2]:
            user_profile_df = load_user_profiles()
            # Generating current user's vector
            course_genres_df = load_course_genres()
            ratings_df = load_ratings()
            all_courses = set(course_genres_df['COURSE_ID'].values)
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_courses = user_ratings['item'].to_list()
            # Trying to create a user vector here
            courses_of_user = course_genres_df[course_genres_df['COURSE_ID'].isin(enrolled_courses)]
            user_vector = courses_of_user.drop(columns=['COURSE_ID', 'TITLE']).sum(axis=0)
            user_vector_df = user_vector * 3
            user_vector_df = pd.DataFrame(user_vector_df).T
            user_vector_df.insert(0, 'user', user_id)
            test_user_vector = user_vector_df.iloc[0, :].values
            # Getting all users profiles
            # user_profile_df = load_user_profiles()
            feature_names = list(user_profile_df.columns[1:])
            # Adding user vector to the rest
            user_profile_df2 = pd.concat([user_vector_df, user_profile_df], ignore_index=True)
            # Standardize the features data for clustering
            scaler = StandardScaler()
            user_profile_df2[feature_names] = scaler.fit_transform(user_profile_df2[feature_names])
            features = user_profile_df2.loc[:, user_profile_df2.columns != 'user']
            user_ids = user_profile_df2.loc[:, user_profile_df2.columns == 'user']
            kmeans = KMeans(n_clusters=cluster_no, random_state=42)
            kmeans.fit(features)
            cluster_labels = kmeans.labels_
            result_df = combine_cluster_labels(user_ids, cluster_labels)
            ratings_df_labelled = pd.merge(ratings_df, result_df, left_on='user', right_on='user')
            # Extracting the 'item' and 'cluster' columns from the test_users_labelled DataFrame
            courses_cluster = ratings_df_labelled[['item', 'cluster']]
            # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
            courses_cluster['count'] = [1] * len(courses_cluster)
            # Grouping the DataFrame by 'cluster' and 'item', aggregating the 'count' column with the sum function,
            # and resetting the index to make the result more readable
            courses_cluster_grouped = courses_cluster.groupby(['cluster', 'item']).agg(
                enrollments=('count', 'sum')).reset_index()
            # Getting user's visited courses, user's cluster number
            user_courses_df = ratings_df_labelled[ratings_df_labelled['user'] == user_id]
            user_courses = user_courses_df['item'].tolist()
            user_cluster = user_courses_df['cluster'].iloc[0]
            # Defining popular courses of this cluster
            cluster_courses = courses_cluster_grouped[courses_cluster_grouped['cluster'] == user_cluster]
            popular_courses = cluster_courses[cluster_courses['enrollments'] > popularity].sort_values(by='enrollments',
                                                                                                       ascending=False)
            # Checking through list of populars. If course does not belong to user courses, adding to final result
            for course, enrollment in zip(popular_courses['item'].values, popular_courses['enrollments'].values):
                if course not in user_courses:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(enrollment)
        if model_name == models[3]:
            user_profile_df = load_user_profiles()
            # Generating current user's vector
            course_genres_df = load_course_genres()
            ratings_df = load_ratings()
            all_courses = set(course_genres_df['COURSE_ID'].values)
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_courses = user_ratings['item'].to_list()
            # Trying to create a user vector here
            courses_of_user = course_genres_df[course_genres_df['COURSE_ID'].isin(enrolled_courses)]
            user_vector = courses_of_user.drop(columns=['COURSE_ID', 'TITLE']).sum(axis=0)
            user_vector_df = user_vector * 3
            user_vector_df = pd.DataFrame(user_vector_df).T
            user_vector_df.insert(0, 'user', user_id)
            test_user_vector = user_vector_df.iloc[0, :].values
            # Getting all users profiles
            # user_profile_df = load_user_profiles()
            feature_names = list(user_profile_df.columns[1:])
            # Adding user vector to the rest
            user_profile_df2 = pd.concat([user_vector_df, user_profile_df], ignore_index=True)
            # Standardize the features data for clustering
            scaler = StandardScaler()
            user_profile_df2[feature_names] = scaler.fit_transform(user_profile_df2[feature_names])
            features = user_profile_df2.loc[:, user_profile_df2.columns != 'user']
            user_ids = user_profile_df2.loc[:, user_profile_df2.columns == 'user']
            # Applying PCA to features
            features_array = pca(features, feature_no)
            features_df = pd.DataFrame(features_array)
            df_combined = pd.concat([user_profile_df2['user'], features_df], axis=1)
            df_combined.set_index('user', inplace=True)
            kmeans = KMeans(n_clusters=cluster_no, random_state=42)
            kmeans.fit(features_df)
            cluster_labels = kmeans.labels_
            result_df = combine_cluster_labels(user_ids, cluster_labels)
            ratings_df_labelled = pd.merge(ratings_df, result_df, left_on='user', right_on='user')
            # Extracting the 'item' and 'cluster' columns from the test_users_labelled DataFrame
            courses_cluster = ratings_df_labelled[['item', 'cluster']]
            # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
            courses_cluster['count'] = [1] * len(courses_cluster)
            # Grouping the DataFrame by 'cluster' and 'item', aggregating the 'count' column with the sum function,
            # and resetting the index to make the result more readable
            courses_cluster_grouped = courses_cluster.groupby(['cluster', 'item']).agg(
                enrollments=('count', 'sum')).reset_index()
            # Getting user's visited courses, user's cluster number
            user_courses_df = ratings_df_labelled[ratings_df_labelled['user'] == user_id]
            user_courses = user_courses_df['item'].tolist()
            user_cluster = user_courses_df['cluster'].iloc[0]
            # Defining popular courses of this cluster
            cluster_courses = courses_cluster_grouped[courses_cluster_grouped['cluster'] == user_cluster]
            popular_courses = cluster_courses[cluster_courses['enrollments'] > popularity].sort_values(by='enrollments',
                                                                                                       ascending=False)
            # Checking through list of populars. If course does not belong to user courses, adding to final result
            for course, enrollment in zip(popular_courses['item'].values, popular_courses['enrollments'].values):
                if course not in user_courses:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(enrollment)
        if model_name == models[4]:
            # Training
            ratings_df = load_ratings()
            reader = Reader(line_format='user item rating', rating_scale=(2, 3))
            course_dataset = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader=reader)
            trainset = course_dataset.build_full_trainset()
            sim_options = {'name': 'pearson', 'user_based': False}
            model_surprise_knn = KNNBasic(sim_options=sim_options)
            model_surprise_knn.fit(trainset)
            # Prediction
            course_genres_df = load_course_genres()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_courses = user_ratings['item'].to_list()
            all_courses = set(course_genres_df['COURSE_ID'].values)
            unknown_courses = all_courses.difference(enrolled_courses)
            filtered_unknown_courses = [course_id for course_id in unknown_courses if
                                        course_id in ratings_df['item'].values]

            def predict_ratings(algo, unknown_courses):
                predictions = []
                for item in unknown_courses:
                    pred = algo.predict(user_id, item)
                    predictions.append(pred.est)
                    if pred.est == 3:
                        users.append(user_id)
                        courses.append(item)
                        scores.append(pred.est)

            predict_ratings(model_surprise_knn, filtered_unknown_courses)
        if model_name == models[5]:
            # Training the model
            ratings_df = load_ratings()
            reader = Reader(line_format='user item rating', rating_scale=(2, 3))
            course_dataset = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader=reader)
            model_surprise_nmf = NMF(verbose=True, random_state=123, init_low=0.5, init_high=5.0,
                                     n_factors=params["factor_no"])
            trainset = course_dataset.build_full_trainset()
            model_surprise_nmf.fit(trainset)
            # Prediction
            course_genres_df = load_course_genres()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_courses = user_ratings['item'].to_list()
            all_courses = set(course_genres_df['COURSE_ID'].values)
            unknown_courses = all_courses.difference(enrolled_courses)
            filtered_unknown_courses = [course_id for course_id in unknown_courses if
                                        course_id in ratings_df['item'].values]

            def predict_ratings(algo, unknown_courses):
                predictions = []
                for course in unknown_courses:
                    pred = algo.predict(user_id, course)
                    predictions.append(pred.est)
                    print(pred)
                    if pred.est == 3:
                        users.append(user_id)
                        courses.append(course)
                        scores.append(pred.est)

            # Predict ratings for the specific user
            predict_ratings(model_surprise_nmf, filtered_unknown_courses)
        if model_name == models[6]:
            # Training
            ratings_df = load_ratings()
            encoded_data, user_idx2id_dict, user_id2idx_dict, course_idx2id_dict, course_id2idx_dict = process_dataset(
                ratings_df)
            num_users = len(ratings_df['user'].unique())
            num_items = len(ratings_df['item'].unique())
            embedding_size = 32
            model = RecommenderNet(num_users, num_items, embedding_size)
            x = encoded_data[['user', 'item']]
            y = encoded_data['rating']
            model.compile(optimizer='adam', loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
            history = model.fit(x, y, validation_split=0.2, epochs=params["epochs"], batch_size=64, verbose=1)
            st.info("Model has been trained. Predicting...")
            # Predicting
            course_genres_df = pd.read_csv('course_genre.csv')
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_courses = user_ratings['item'].to_list()
            all_courses = set(course_genres_df['COURSE_ID'].values)
            unknown_courses = all_courses.difference(enrolled_courses)
            filtered_unknown_courses = [course_id for course_id in unknown_courses if
                                        course_id in ratings_df['item'].values]

            try:
                results_df = predict_ratings_for_user(model, user_id, filtered_unknown_courses, user_id2idx_dict,
                                                      course_id2idx_dict, course_idx2id_dict)
                st.info("Predictions:", results_df)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                st.error(f"Failed to load model: {e}")
            # results_df = predict_ratings_for_user(nn_model, user_id, filtered_unknown_courses, gl_user_id2idx_dict,
            #                                       gl_course_id2idx_dict, gl_course_idx2id_dict)
            for index, row in results_df.iterrows():
                if row['predicted_rating'] > nn_threshold:
                    users.append(user_id)
                    courses.append(row['course_id'])
                    scores.append(row['predicted_rating'])

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
