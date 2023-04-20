from api import MagRetraceData
import numpy as np
import scipy
import scipy.signal as signal
from scipy.spatial.distance import euclidean
from dtw import dtw
import scipy.fftpack as fftpack

def dtw_analysis(template, sequence):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.
    Returns the DTW distance and the matching part of the sequence.
    """
    # Ensure that both arrays are 1-D
    template = np.reshape(template, (-1, 1))
    sequence = np.reshape(sequence, (-1, 1))

    # Define the distance function to use (Euclidean distance)
    distance = lambda x, y: euclidean(x, y)

    # Compute the DTW distance between the template and the sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(template, sequence, dist=distance)

    # Extract the matching part of the sequence based on the DTW path
    start, end = path[0][1], path[-1][1]
    matched_sequence = sequence[start:end+1]

    return d, matched_sequence


def downsample(signal_time_seq: np.ndarray, signal: np.ndarray, downsample_factor) -> tuple[np.ndarray]:
    """
    Downsample the signal by the given factor.
    """
    # Compute the downsampled signal.
    signal_downsampled = signal[::downsample_factor]

    # Compute the downsampled time sequence.
    signal_time_seq_downsampled = signal_time_seq[::downsample_factor]

    return signal_time_seq_downsampled, signal_downsampled


def sliding_window(sequence: np.ndarray, window_length: int, step: int) -> list[np.ndarray]:
    """
    Split the sequence into overlapping windows.
    """
    windows = []
    for i in range(0, len(sequence) - window_length + 1, step):
        window = sequence[i:i+window_length]
        windows.append(window)
    return windows


def gaussian_filter(signal: np.ndarray, sigma: float):
    """
    Apply a Gaussian filter to the signal.
    """
    return scipy.ndimage.gaussian_filter(signal, sigma=sigma)


def get_dtw_distance(template: MagRetraceData,
    traversal: MagRetraceData,
    analysis_target: str,
    downsampling_coefficient=10,
    windowing_coefficient=1,
    step_coefficient=0.25,
    perform_gaussian_filtering=False,
    gaussian_filter_sigma=1.0):
    """
    An analysis function designed for the MagRetraceData class.
    Returns the DTW distance between the template and the traversal.
    Supports extra signalprocessing functionalities such as downsampling and Gaussian filtering.
    """
    tem_timeseq = template.time_seq
    tra_timeseq = traversal.time_seq
    if (analysis_target == 'abs'):
        tem_signal = template.mag_abs
        tra_signal = traversal.mag_abs
    elif (analysis_target == 'x'):
        tem_signal = template.mag_x
        tra_signal = traversal.mag_x
    elif (analysis_target == 'y'):
        tem_signal = template.mag_y
        tra_signal = traversal.mag_y
    elif (analysis_target == 'z'):
        tem_signal = template.mag_z
        tra_signal = traversal.mag_z
    else:
        raise ValueError("Analysis target should be one of \'abs\', \'x\', \'y\', \'z\'")
    
    # Filtering
    if perform_gaussian_filtering:
        tem_signal = gaussian_filter(tem_signal, gaussian_filter_sigma)
        tra_signal = gaussian_filter(tra_signal, gaussian_filter_sigma)
    
    # Downsampling
    tem_timeseq, tem_signal = downsample(tem_timeseq, tem_signal, downsampling_coefficient)
    tra_timeseq, tra_signal = downsample(tra_timeseq, tra_signal, downsampling_coefficient)

    tem_length = tem_signal.shape[0]
    window_length = round(tem_length * windowing_coefficient)
    window_step = round(tem_length * step_coefficient)
    windows = sliding_window(tra_signal, window_length, window_step)
    ds = []
    for window in windows:
        d, matched_sequence = dtw_analysis(tem_signal, window)
        ds.append(d)

    time_seq_idx = np.array([idx * window_step for idx in range(len(windows))])
    sample_interval = tra_timeseq[1] - tra_timeseq[0]
    time_seq = tra_timeseq[time_seq_idx]
    #print(f"Window length: {window_length * sample_interval: .2f} s")
    time_seq = np.array(time_seq)
    ds = np.array(ds)
    return (time_seq, ds)


def perform_fft_analysis(timeseq, signal, low_freq_cutoff=0, high_freq_cutoff=10):
    """
    Perform FFT analysis on the signal.
    Returns the frequency spectrum and the frequency sequence.
    """
    # Compute the FFT of the signal
    signal_fft = fftpack.fft(signal)

    # Compute the frequency sequence
    freq_seq = fftpack.fftfreq(signal.size, d=timeseq[1] - timeseq[0])

    # Shift the frequency sequence
    freq_seq = fftpack.fftshift(freq_seq)
    signal_fft = fftpack.fftshift(signal_fft)

    # Apply the frequency cutoff
    signal_fft = signal_fft[freq_seq >= low_freq_cutoff]
    freq_seq = freq_seq[freq_seq >= low_freq_cutoff]
    signal_fft = signal_fft[freq_seq <= high_freq_cutoff]
    freq_seq = freq_seq[freq_seq <= high_freq_cutoff]

    # Compute the frequency spectrum
    freq_spectrum = np.abs(signal_fft)

    return freq_seq, freq_spectrum

def get_frequency_spectrum_similarity(template: MagRetraceData,
                                        traversal: MagRetraceData,
                                        analysis_target: str,
                                        windowing_coefficient=1,
                                        step_coefficient=0.25):
    """
    Measures the similarity between two 1d signals via the frequency spectrum, in euclidean distance.
    """
    tem_timeseq = template.time_seq
    tra_timeseq = traversal.time_seq
    if (analysis_target == 'abs'):
        tem_signal = template.mag_abs
        tra_signal = traversal.mag_abs
    elif (analysis_target == 'x'):
        tem_signal = template.mag_x
        tra_signal = traversal.mag_x
    elif (analysis_target == 'y'):
        tem_signal = template.mag_y
        tra_signal = traversal.mag_y
    elif (analysis_target == 'z'):
        tem_signal = template.mag_z
        tra_signal = traversal.mag_z
    else:
        raise ValueError("Analysis target should be one of \'abs\', \'x\', \'y\', \'z\'")

    tem_length = tem_signal.shape[0]
    window_length = round(tem_length * windowing_coefficient)
    window_step = round(tem_length * step_coefficient)
    windows = sliding_window(tra_signal, window_length, window_step)
    ds = []
    for window in windows:
        template_freq_seq, template_freq_spectrum = perform_fft_analysis(tem_timeseq, tem_signal)
        traversal_freq_seq, traversal_freq_spectrum = perform_fft_analysis(tra_timeseq, window)
        assert template_freq_seq.shape == traversal_freq_seq.shape
        d = euclidean(template_freq_spectrum, traversal_freq_spectrum)
        ds.append(d)

    time_seq_idx = np.array([idx * window_step for idx in range(len(windows))])
    sample_interval = tra_timeseq[1] - tra_timeseq[0]
    time_seq = tra_timeseq[time_seq_idx]
    #print(f"Window length: {window_length * sample_interval: .2f} s")
    time_seq = np.array(time_seq)
    ds = np.array(ds)
    return (time_seq, ds)