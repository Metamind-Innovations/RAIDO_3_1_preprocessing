import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def pca_dim_reduction(data, max_components=None):
    """
    Perform PCA on a given dataset of time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be reduced.
    max_components : int or None, optional
        The maximum number of components to consider. If None, the number of components is determined
        dynamically depending on the size of the dataset.

    Returns
    -------
    reduced_data : numpy.ndarray
        The reduced dataset.
    """

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['value']])

    # Dynamically determine n_components
    n_samples, n_features = scaled_data.shape[0], scaled_data.shape[1]
    n_components = min(max_components or n_samples, n_samples, n_features)

    if n_components < 1:
        raise ValueError("PCA cannot be performed: insufficient data.")
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)
    return reduced_data

# Diffusion Maps
def diffusion_maps(data, n_components, n_neighbors=10):
    """
    Apply diffusion maps to a given dataset of time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be reduced.
    n_components : int
        The number of components to consider.
    n_neighbors : int, optional
        The number of neighbors to consider for the diffusion map.

    Returns
    -------
    reduced_data : numpy.ndarray
        The reduced dataset.
    """

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['value']])

    # Apply Isomap as an approximation of diffusion maps
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    reduced_data = isomap.fit_transform(scaled_data)

    return reduced_data

# Piecewise Vector Quantized Approximation (PVQA)
def pvqa(data, num_segments):
    """
    Apply PVQA to a given dataset of time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be reduced.
    num_segments : int
        The number of segments to divide the dataset into.

    Returns
    -------
    codebook : list
        The codebook of the PVQA.
    """

    segment_size = len(data) // num_segments
    segments = [data[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]
    codebook = [segment.mean() for segment in segments]
    return codebook


# Autoencoder-based Reduction
def autoencoder_reduction(data, n_lag):
    """
    Perform autoencoder-based reduction on a given dataset of time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be reduced.
    n_lag : int
        The number of lagged features to create.

    Returns
    -------
    reduced_data : numpy.ndarray
        The reduced dataset.
    """

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['value']])

    # Create lagged features
    lagged_data = np.column_stack([scaled_data[i:len(scaled_data) - n_lag + i] for i in range(n_lag)])

    # Autoencoder setup
    autoencoder = MLPRegressor(hidden_layer_sizes=(n_lag // 2,), max_iter=200, random_state=42)
    autoencoder.fit(lagged_data, lagged_data)

    reduced_data = autoencoder.predict(lagged_data)
    return reduced_data
