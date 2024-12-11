import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import pywt


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
        fft_values = np.fft.fft(df_copy[column])
        df_copy[f'{column}_fourier'] = np.abs(fft_values)
    return df_copy


def savitzky_golay(df: pd.DataFrame, window_length: int = 51, poly_order: int = 3) -> pd.DataFrame:
    """
    Apply a Savitzky-Golay filter to each column in the DataFrame, except the first one which is assumed to be a datetime column.

    The Savitzky-Golay filter is a digital filter that can be used to smooth and differentiate a signal. It is a particular type of low-pass filter that convolves the signal with a Gaussian and then applies a least squares fit to the convolved signal. It is often used to remove noise from a signal.

    :param df: The DataFrame containing the data for which the Savitzky-Golay filter should be applied.
    :type df: pd.DataFrame
    :param window_length: The length of the filter window. Must be an odd positive integer. Default is 51.
    :type window_length: int, optional
    :param poly_order: The order of the polynomial used to fit the samples. Must be less than the window_length. Default is 3.
    :type poly_order: int, optional
    :return: A DataFrame with the original data and additional columns for each filtered column.
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()
    for column in df_copy.columns[1:]:
        df_copy[f'{column}_sg'] = savgol_filter(
            df_copy[column],
            window_length=window_length,
            polyorder=poly_order
        )
    return df_copy


def wavelet_denoising(df: pd.DataFrame, wavelet: str = 'db4', level: int = 1) -> pd.DataFrame:
    """
    Apply a wavelet denoising filter to each column in the DataFrame, except the first one which is assumed to be a datetime column.

    :param df: The DataFrame containing the data for which the wavelet denoising should be applied.
    :type df: pd.DataFrame
    :param wavelet: The type of wavelet to use. Default is 'db4'.
    :type wavelet: str, optional
    :param level: The level of the wavelet decomposition. Default is 1.
    :type level: int, optional
    :return: A DataFrame with the original data and additional columns for each filtered column.
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()
    for column in df_copy.columns[1:]:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(df_copy[column], wavelet, level=level)

        # Threshold the coefficients
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(df_copy[column])))
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]

        # Reconstruct the signal
        df_copy[f'{column}_wavelet'] = pywt.waverec(coeffs, wavelet)

    return df_copy

# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# noise_removed = ema(df)
# print(f'Noise removal: {noise_removed.head(20)}')
#
# transformed = fourier_transform(df)
# print(f'Fourier Transform: {transformed.head(20)}')
#
# savitzky = savitzky_golay(df)
# print(f'Savitzky: {savitzky.head(20)}')
#
# wavelet = wavelet_denoising(df)
# print(f'Wavelet: {wavelet.head(20)}')
