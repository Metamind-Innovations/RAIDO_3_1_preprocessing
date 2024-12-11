import numpy as np
import pandas as pd


def ema(df: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """
    Calculate the Exponential Moving Average (EMA) for each column in the DataFrame,
    except the first one which is assumed to be a datetime column.

    :param df: The DataFrame containing the data for which the EMA should be calculated.
    :type df: pd.DataFrame
    :param alpha: The smoothing factor, a value between 0 and 1. Default is 0.2.
    :type alpha: float, optional
    :return: A DataFrame with the original data and additional columns for each EMA calculated.
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()
    for column in df_copy.columns[1:]:
        df_copy[f'{column}_ema'] = df_copy[column].ewm(alpha=alpha).mean()
    return df_copy


def fourier_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Fast Fourier Transform (FFT) for each column in the DataFrame,
    except the first one which is assumed to be a datetime column.

    :param df: The DataFrame containing the data for which the FFT should be calculated.
    :type df: pd.DataFrame
    :return: A DataFrame with the original data and additional columns for each FFT calculated.
    :rtype: pd.DataFrame
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
