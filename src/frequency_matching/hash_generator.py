"""
Created: 1/24/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import numpy as np


class HashGenerator:
    def __init__(self, time_points, freq_points, F):

        # combine time and frequency points to an array with first column being time and second frequency
        self.time_freq = np.transpose(np.array([time_points, freq_points]))
        self.__hashes = self.__generate_hashes(self.time_freq, F)

    @property
    def hashes(self):
        return self.__hashes

    def __generate_hashes(self, time_freq, F):
        """
        Generates hashes of time-frequency point combinations (2 points at a time).

        Args:
            time_freq (ndarray): a 2-D numpy array, where the first column contains the timing information and the
                second frequency information.
            F: the number of points in a target zone

        Returns:
            all_hashes (ndarray): a 2-D numpy array containing hashes created from all target zones.

        """
        assert isinstance(time_freq, np.ndarray)
        assert time_freq.ndim == 2 and time_freq.shape[1] == 2
        assert F <= time_freq.shape[0]

        # sorting the array containing time-freq coordinates according to frequency (axis=1), and time(axis=0). To
        #  do that, first sort the least significant axis (time) then the most significant one (freq). The kind of
        # algorithm selected is mergesort, since it is stable (unlike quicksort or heapsort).
        freq_time_sorted = time_freq[time_freq[:, 0].argsort(kind='mergesort')]
        freq_time_sorted = freq_time_sorted[freq_time_sorted[:, 1].argsort(kind='mergesort')]

        hash_list = []
        for i in range(0, len(freq_time_sorted) - 1):
            if i + F <= freq_time_sorted.shape[0]:
                target_zone = freq_time_sorted[i:i + F, :]
            else:
                target_zone = freq_time_sorted[i:, :]

            zone_hashes = self.__get_zone_hashes(target_zone)
            hashes = np.stack(zone_hashes, axis=0)
            hash_list.append(hashes)

        all_hashes = np.concatenate(hash_list, axis=0)
        return all_hashes

    def __get_zone_hashes(self, target_zone):
        """
        Get hashes created from one target zone.

        Args:
            target_zone (ndarray): a collection of points (2-D array, first column time and second frequency).

        Returns:
            zone_hashes (tuple): hashes generated from one target zone.

        """
        zone_hashes = ()
        anchor = target_zone[0, :]
        for i in range(1, len(target_zone)):
            point = target_zone[i, :]
            hash_data = self.__compute_hash(point_pair=(anchor, point))
            zone_hashes = zone_hashes + (hash_data,)

        return zone_hashes

    def __compute_hash(self, point_pair):
        """
        Computes hash between two points, composed of: the frequency of the anchor's (first point), the frequency of
        the second point, time difference between two points, absolute time of anchor point.

        Args:
            point_pair (ndarray): Numpy ndarray of two ndarrays, each representing a point whose hash to compute

        Returns:
            hash (ndarray): hash between two points

        """
        assert isinstance(point_pair, tuple)
        assert (len(point_pair) == 2)

        anchor = point_pair[0]
        point = point_pair[1]

        # get the absolute time of the anchor point (from the beginning of signal acq.)
        t0 = anchor[0]
        # get time difference between the two points
        dt = point[0] - anchor[0]

        f0 = anchor[1]
        f1 = point[1]

        hash = np.array([f0, f1, dt, t0])

        return hash
