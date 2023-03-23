"""
Created: 1/17/19
© Denisa Qori 2019 All Rights Reserved
"""
import numpy as np


class Spectrogram_Filterer:
    def __init__(self, spectrogram):
        self.spectrogram = spectrogram

    def group_into_freq_bands(self, spectrogram, num_bands=6):
        """
        Groups a spectrum input into frequency bands based on the num_bands input.

        Args:
            spectrogram (ndarray): the spectrum representation of the signal
            num_bands (int): the number of groups of frequencies to produce

        Returns:
            band_dict (dict): a dictionary mapping a tuple of frequency values to the values of those frequencies
            from the spectrum.

        """
        num_freq = spectrogram.shape[0]
        band_dict = {}
        start_freq = 0

        base_num_freq = int(num_freq / num_bands)
        extra_freq = num_freq % num_bands

        for i in range(0, num_bands):
            if extra_freq > 0:
                stop_freq = start_freq + base_num_freq + 1
                extra_freq -= 1
            else:
                stop_freq = start_freq + base_num_freq

            band_dict[(start_freq, stop_freq - 1)] = spectrogram[start_freq: (stop_freq - 1), :]
            start_freq = stop_freq

        return band_dict

    def get_strongest_freq(self, populated_bands, num_strongest=10):
        """
        Identifies and returns the highest intensity points over each frequency band of the signal over time.

        Args:
            populated_bands (dict): a dictionary mapping a tuple of start and end frequency for a specific frequency
                band to intensity values of those frequencies wrt. time.
            num_strongest (int): the number of maximum intensity points to keep in each band

        Returns:
            index_list (list): list of tuples of two arrays each indicating the coordinates of the maximum values in
            all the bands of populated_bands. The first index is that of frequency and the second is time.

        """
        strongest_freq = {}

        for band, bins in populated_bands.items():
            assert num_strongest < bins.size
            max_ampl = [0] * num_strongest

            for freq in range(0, bins.shape[0]):
                for time in range(0, bins.shape[1]):
                    if bins[freq, time] > max_ampl[0]:
                        max_ampl[0] = bins[freq, time]
                        max_ampl.sort()
            strongest_freq[band] = max_ampl

        mean = self.__compute_mean(strongest_freq)

        index_list = []
        for band, ampl_list in strongest_freq.items():
            for ampl in ampl_list:
                i, j = np.where(populated_bands[band] == ampl)
                index_list.append((i + band[0], j))

        time_axis, freq_axis = self.__get_strongest_freq_coord(index_list)
        return time_axis, freq_axis

    def __get_strongest_freq_coord(self, index_list):
        """
        Converts the list of maximum intensity points to two lists, respectively corresponding to time and frequency.

        Args:
            index_list (list): list of tuples of two arrays each indicating the coordinates of the maximum values in
            all the bands of populated_bands. The first index is that of frequency and the second is time.

        Returns:
            time_axis (list): list of time coordinates
            freq_axis (list): list of frequency coordinates

        """
        freq_axis = []
        time_axis = []
        for freq, time in index_list:
            assert freq.size == time.size
            for i in range(0, len(freq)):
                freq_axis.append(freq[i])
                time_axis.append(time[i])

        return time_axis, freq_axis

    def __compute_mean(self, strongest_freq):
        s = 0
        total_len = 0
        for band, ampl_list in strongest_freq.items():
            s += sum(ampl_list)
            total_len += len(ampl_list)

        mean = s / total_len
        return mean
