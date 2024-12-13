import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def pca_dim_reduction(data, column, max_components=None):
    """
        Perform Principal Component Analysis (PCA) on the given column of the
        given DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame to reduce.
        column : str
            The column to reduce.
        max_components : int, optional
            The maximum number of components to keep. If unspecified, will use as
            many as necessary to capture all of the variance in the data, up to the
            number of samples.

        Returns
        -------
        pandas.DataFrame
            The reduced DataFrame.

        Note
        ----
        The number of components to keep is determined by taking the minimum of
        the number of samples and the number of features, unless max_components is
        specified, in which case it will use that instead.

        The reduced DataFrame is created by replacing the original column with the
        projected data.
        """
    data_copy = data.copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_copy[[column]])

    # Dynamically determine n_components
    n_samples, n_features = scaled_data.shape[0], scaled_data.shape[1]
    n_components = min(max_components or n_samples, n_samples, n_features)

    if n_components < 1:
        raise ValueError("PCA cannot be performed: insufficient data.")
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)
    data_copy[column] = reduced_data
    return data_copy


def isometric_mapping(data, n_components, column, n_neighbors=10):
    """
    Perform Diffusion Maps on a given dataset of time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be reduced.
    n_components : int
        The number of components to keep.
    column : str
        The column to reduce.
    n_neighbors : int, optional
        The number of neighbors to consider. Defaults to 10.

    Returns
    -------
    pandas.DataFrame
        The reduced DataFrame.

    Note
    ----
    Diffusion Maps is a dimensionality reduction technique that applies a Markov
    transition matrix to the data, which is then used to reduce the dimensionality
    of the data. In this implementation, we use Isomap as an approximation of
    diffusion maps, which constructs a graph of the data and then computes the
    shortest path between all points. The reduced data is computed by computing
    the eigenvectors of the graph Laplacian, and then selecting the top n
    eigenvectors as the reduced data.
    """
    data_copy = data.copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_copy[[column]])

    # Apply Isomap as an approximation of diffusion maps
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    reduced_data = isomap.fit_transform(scaled_data)
    data_copy[column] = reduced_data

    return data_copy


def pvqa(data, num_segments=100):
    """
    Perform Piecewise Vector Quantized Approximation (PVQA) on a given dataset of time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be reduced.
    num_segments : int, optional
        The number of segments to divide the data into. Defaults to 10.

    Returns
    -------
    data : pandas.DataFrame
        The reduced dataset.

    Notes
    -----
    PVQA is a lossy compression technique for time series data.
    It works by dividing the data into segments and then applying Vector Quantization to each segment.
    The result is a dataset with the same shape as the original but with reduced dimensionality.

    The process can be thought of as follows:

    1. Divide the data into segments of equal size.
    2. Calculate the mean of each segment.
    3. Replace each segment with its mean.
    """
    data_copy = data.copy()
    segment_size = len(data_copy) // num_segments
    segments = [data_copy[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]
    codebook = [segment.mean() for segment in segments]
    for i, segment in enumerate(segments):
        data_copy.iloc[i * segment_size:(i + 1) * segment_size, :] = codebook[i]
    return data_copy


def autoencoder_reduction(data, column, n_lag):
    """
    Perform Autoencoder-based dimensionality reduction on a given dataset of time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be reduced.
    column : str
        The column of the dataset to be reduced.
    n_lag : int
        The number of lagged features to create.

    Returns
    -------
    data : pandas.DataFrame
        The reduced dataset.

    Notes
    -----
    This function works by creating lagged features from the given column, and then using an Autoencoder to reduce the dimensionality of the data.
    The Autoencoder is trained on the lagged features, and then used to predict the reduced representation of the data.
    The reduced representation is then merged back into the original DataFrame, and the original column is dropped.
    """
    data_copy = data.copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_copy[[column]])

    # Create lagged features
    lagged_data = np.column_stack([scaled_data[i:len(scaled_data) - n_lag + i + 1] for i in range(n_lag)])

    # Autoencoder setup
    hidden_layer_size = max(1, n_lag // 2)  # Ensure at least one neuron
    autoencoder = MLPRegressor(hidden_layer_sizes=(hidden_layer_size,), max_iter=200, random_state=42)
    autoencoder.fit(lagged_data, lagged_data)

    # Get the reduced representation
    reduced_data = autoencoder.predict(lagged_data)

    # Create a new DataFrame with the reduced data
    if reduced_data.ndim == 1:
        reduced_df = pd.DataFrame(reduced_data, columns=[f'{column}_reduced'])
    else:
        reduced_df = pd.DataFrame(reduced_data[:, 0], columns=[f'{column}_reduced'])
    reduced_df.index = data_copy.index[n_lag - 1:]

    # Merge the reduced data back into the original DataFrame
    data_copy = data_copy.join(reduced_df)
    data_copy[column] = data_copy[f'{column}_reduced'].fillna(data_copy[column])
    data_copy = data_copy.drop(columns=[f'{column}_reduced'])

    return data_copy

# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# pca_df = pca_dim_reduction(df, 'value')
# print(pca_df.head(20))

# iso_df = isometric_mapping(df, 2, 'value')
# print(iso_df.head(20))

# diffusion_df = diffusion_maps(df, 2, 'value', n_neighbors=5)
# print(diffusion_df.head(20))

# pvqa_df = pvqa(df, 500)
# print(pvqa_df.head(50))

# autoencoder_df = autoencoder_reduction(df, 'value', 4)
# print(autoencoder_df.head(50))
