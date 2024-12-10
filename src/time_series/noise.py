import numpy as np
import pandas as pd


def ema(df: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """
    This function will remove the noise from a given dataframe by implementing the exponential moving average algorithm.

    Parameters:
    df (pd.DataFrame): The dataframe to remove the noise from.
    alpha (float): The smoothing factor. Default value is 0.2.

    Returns:
    pd.DataFrame: The dataframe with new columns added to the right of the existing ones containing the exponential moving average values.
    """
    df_copy = df.copy()
    for column in df_copy.columns[1:]:
        df_copy[f'{column}_ema'] = df_copy[column].ewm(alpha=alpha).mean()
    return df_copy

def fourier_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function will calculate the fourier transform for all columns in a given dataframe except the first one.

    Parameters:
    df (pd.DataFrame): The dataframe to calculate the fourier transform for.

    Returns:
    pd.DataFrame: The dataframe with new columns added to the right of the existing ones containing the fourier transform values.
    """
    df_copy = df.copy()
    for column in df_copy.columns[1:]:
        df_copy[f'{column}_fourier'] = np.abs(np.fft.rfft(df_copy[column]))
    return df_copy

df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
noise_removed = ema(df)
print(f'Noise removal: {noise_removed.head(20)}')

transformed = fourier_transform(df)
print(f'Fourier Transform: {transformed.head(20)}')