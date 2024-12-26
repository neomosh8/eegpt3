#utils.py
import numpy as np
import pywt



def wavelet_decompose_window(window, wavelet='db2', level=4):
    """
    Apply wavelet decomposition to each channel in the window.

    :param window: 2D numpy array (channels x samples)
    :param wavelet: Wavelet type
    :param level: Decomposition level
    :return: decomposed_channels, coeffs_lengths, original_signal_length, normalized_data
    """
    num_channels, num_samples = window.shape
    decomposed_channels = []
    coeffs_lengths = []
    normalized_data = []

    for channel_index in range(num_channels):
        # Extract single channel data
        channel_data = window[channel_index, :]

        # Normalize the channel data (z-score normalization)
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std == 0:
            std = 1  # Prevent division by zero
        channel_data_normalized = (channel_data)
        normalized_data.append(channel_data_normalized)

        # Perform wavelet decomposition on normalized data
        coeffs = pywt.wavedec(channel_data_normalized, wavelet, level=level)

        # Flatten the coefficients
        flattened_coeffs = np.hstack([comp.flatten() for comp in coeffs])

        # Store the lengths of each coefficient array
        lengths = np.array([len(comp) for comp in coeffs])

        decomposed_channels.append(flattened_coeffs)
        coeffs_lengths.append(lengths)

    decomposed_channels = np.array(decomposed_channels)
    coeffs_lengths = np.array(coeffs_lengths)  # Shape: (channels x num_levels+1)
    normalized_data = np.array(normalized_data)
    return decomposed_channels, coeffs_lengths, num_samples, normalized_data

def wavelet_reconstruct_window(decomposed_channels, coeffs_lengths, num_samples, wavelet='db2'):
    """
    Reconstruct the normalized signal from decomposed channels.

    :param decomposed_channels: 2D numpy array (channels x flattened_coefficients)
    :param coeffs_lengths: 2D numpy array (channels x num_levels+1)
    :param num_samples: Original signal length
    :param wavelet: Wavelet type
    :return: Reconstructed window (channels x samples)
    """
    num_channels = decomposed_channels.shape[0]
    reconstructed_window = []

    for channel_index in range(num_channels):
        flattened_coeffs = decomposed_channels[channel_index]
        lengths = coeffs_lengths[channel_index]
        # Split the flattened coefficients back into list of arrays
        coeffs = []
        idx = 0
        for length in lengths:
            coeff = flattened_coeffs[idx:idx+length]
            coeffs.append(coeff)
            idx += length
        # Reconstruct the signal using waverec
        channel_data_normalized = pywt.waverec(coeffs, wavelet)[:num_samples]
        reconstructed_window.append(channel_data_normalized)

    reconstructed_window = np.array(reconstructed_window)
    return reconstructed_window

def quantize_number(num):
    """
    Quantizes a z-score number and returns a unique string identifier.
    """
    # Clamping the input number to the range -10 to +10
    if num < -10:
        num = -10
    elif num > 10:
        num = 10

    # Determining step size based on the absolute value of the number
    abs_num = abs(num)
    if abs_num <= 1:
        step = 0.01  # High precision near 0
    elif abs_num <= 2:
        step = 0.05
    elif abs_num <= 3:
        step = 0.1
    elif abs_num <= 5:
        step = 0.5
    else:
        step = 1  # Coarser precision for outliers

    # Quantizing the number
    quantized_num = round(num / step) * step

    # Generating a unique string identifier
    # Using letters to represent the integer part and digits for the fractional part
    letters = 'LMNQRUVWXY'
    integer_part = int(abs(quantized_num))
    if integer_part > 9:
        integer_part = 9  # Cap at 9 since we have letters A-J

    if quantized_num >= 0:
        letter = letters[integer_part]
    else:
        letter = letters[integer_part].lower()

    # Calculating the fractional part as a two-digit number
    fractional_part = int(round((abs(quantized_num) - integer_part) / step))
    # Ensure the fractional part fits in two digits
    max_fractional = int(1 / step)
    if fractional_part >= max_fractional:
        fractional_part = max_fractional - 1

    # Constructing the unique identifier
    unique_identifier = f"{letter}{fractional_part:02d}"

    return unique_identifier

def dequantize_identifier(identifier):
    """
    Dequantizes a unique string identifier back to the original number.
    """
    letters = 'LMNQRUVWXY'
    letters_lower = letters.lower()

    # Extract the letter and digits from the identifier
    letter = ''.join(filter(str.isalpha, identifier))
    digits = ''.join(filter(str.isdigit, identifier))

    # Determine the integer part from the letter
    if letter in letters:
        integer_part = letters.index(letter)
        sign = 1
    elif letter in letters_lower:
        integer_part = letters_lower.index(letter)
        sign = -1
    else:
        print(letter)
        raise ValueError("Invalid identifier")

    # Determine step size based on the absolute integer part
    abs_integer = integer_part
    if abs_integer <= 1:
        step = 0.01
    elif abs_integer <= 2:
        step = 0.05
    elif abs_integer <= 3:
        step = 0.1
    elif abs_integer <= 5:
        step = 0.5
    else:
        step = 1

    # Reconstruct the fractional part from the digits
    fractional_part = int(digits)
    quantized_fraction = fractional_part * step

    # Combine integer and fractional parts
    quantized_num = sign * (integer_part + quantized_fraction)

    return quantized_num