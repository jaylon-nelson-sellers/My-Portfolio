import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS
from sklearn.decomposition import FastICA, PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler


class LoadStockDataset:
    """
    A class to load, preprocess, normalize, and select features from a dataset.
    """

    def __init__(self, dataset_index, normalize=1, verbose=0):
        """
        Initializes the dataset by loading files, normalizing features, and selecting features based on the parameters.
        """
        # Print loading message if verbose
        if verbose:
            print("Loading File")
        # need to change
        if normalize:
            self.feats = pd.read_csv("feats.csv")
            self.normalize()
            self.feats.to_csv("feats_n.csv", index=False)
            return

        if dataset_index == 1:
            self.observed = 1
            self.feats = pd.read_csv("feats_n.csv")
            self.targets = pd.read_csv("regress.csv")
        elif dataset_index == 10:
            self.observed = 10
        else:
            self.observed = 100


        # Read classification, features, and regression targets
        # change back after vae

        # Select targets based on the target index


        # Replace missing and infinite values with zero
        self.feats.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
        self.targets.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

        def convert_to_numeric(df):
            return df.apply(pd.to_numeric, errors='coerce').fillna(0)

        self.feats = convert_to_numeric(self.feats)
        self.targets = convert_to_numeric(self.targets)

        # Normalize features if requested

        # Replace missing and infinite values with zero
        self.feats.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
        self.targets.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
        #
        #print("Dataset Loaded")

    def get_train_test_split(self, split=0.2):
        """
        Splits the features and targets into training and testing sets.
        """
        return train_test_split(self.feats, self.targets, test_size=split, random_state=1)

    def is_power_of_two(self, n):
        return (n != 0) and (n & (n - 1) == 0)

    def next_smallest_power_of_two(self, n):
        if n <= 1:
            return 0
        return 2 ** ((n - 1).bit_length() - 1)

    def confirm_power_of_two(self, feats_3d):
        num_columns = feats_3d.shape[2]

        if not self.is_power_of_two(num_columns):
            num_columns = feats_3d.shape[2]
            next_pow2 = self.next_smallest_power_of_two(num_columns)
            additional_cols = next_pow2 - num_columns
            random_array = np.random.rand(feats_3d.shape[0], feats_3d.shape[1], additional_cols)
            feats_3d = np.concatenate((feats_3d, random_array), axis=2)
        return feats_3d


    def get_3d(self, version=1, split=.2):
        if version == 0:
            # "1D" Convolution
            # [n_samples, 1, n_features]
            new_shape = (self.feats.shape[0], 1, self.feats.shape[1])
            self.feats = self.feats.values.reshape(new_shape)
            return train_test_split(self.feats, self.targets, test_size=split, random_state=1)

        if version == 1:
            # 2D Method 1: Stacking Observed Days
            # [n_samples,observed_days,n_features/observed_days]
            if self.observed == 1:

                new_shape = (self.feats.shape[0], 1, self.feats.shape[1])
                self.feats = self.feats.values.reshape(new_shape)
                return train_test_split(self.feats, self.targets, test_size=split, random_state=1)
            else:
                feats = self.feats
                observed = self.observed + 1
                # Add empty columns if necessary
                num_columns = feats.shape[1]
                remainder = num_columns % observed
                if remainder != 0:
                    num_extra_columns = observed - remainder
                    print(f"Adding {num_extra_columns} to fill Dataframe")
                    # Perform FastICA with num_extra_columns components
                    ica = FastICA(n_components=num_extra_columns, random_state=0)
                    ica_components = ica.fit_transform(feats)

                    # Add the FastICA components as new columns to feats
                    for i in range(num_extra_columns):
                        feats[f'ica_{i}'] = ica_components[:, i]
                self.feats = feats
                # self.feats.to_csv(f"feats_days-{self.observed}_comp-{self.feats.shape[1]}-v2.csv", index=False)

                # Reshape the dataframe to 3D
                new_depth = observed
                new_shape = (feats.shape[0], feats.shape[1] // new_depth, new_depth)
                feats_3d = feats.values.reshape(new_shape)
                feats_3d = np.swapaxes(feats_3d, 1, 2)
                self.feats = feats_3d
                print(f"New Shape:{self.feats.shape}")
                return train_test_split(self.feats, self.targets, test_size=split, random_state=1)
        if version == 2:
            # need to finish this part
            self.IGTD()

    from sklearn.preprocessing import StandardScaler

    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    import numpy as np

    def normalize(self):
        # Create a copy of the original dataframe
        normalized_df = self.feats

        # Initialize RobustScaler
        scaler = StandardScaler()

        normalized_data = scaler.fit_transform(normalized_df)

        self.feats = pd.DataFrame(normalized_data)


    def apply_clustering(self):
        """
        Apply multiple clustering methods with varying numbers of clusters
        and append the results to the original dataframe.

        :param df: Original dataframe
        :param feature_columns: List of column names to use for clustering
        :return: Dataframe with cluster labels appended
        """
        # Extract features for clustering
        features = self.feats


        cluster_labels = {}

        # Apply K-Means and HAC for 2-5 clusters
        for n_clusters in range(2,11):
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(features)
            cluster_labels[f'kmeans_{n_clusters}'] = kmeans_labels



        # Append cluster labels to the original dataframe
        for method, labels in cluster_labels.items():
            features[f'cluster_{method}'] = labels
        self.feats = features

#ld = LoadStockDataset(1,1)