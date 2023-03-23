import math
import os
from abc import abstractmethod

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utilities import get_root_path, save_cpickle, load_cpickle
from numpy import fft
from datetime import timedelta


def znorm(x):
    normalized_x = (x - x.mean()) / (x.std() + 1e-16)
    return normalized_x


def predict_by_model(x, kmeans_model):
    normalized_x = znorm(x).reshape(1, -1)
    predicted_cluster = kmeans_model.predict(normalized_x)

    return predicted_cluster[0]


def perform_fft(w_data_ls, num_top_freq):
    data_fft_ls = []
    for i, wind_samp in enumerate(w_data_ls):
        chunk_fft_ls = []
        for j in range(0, wind_samp.shape[0]):
            all_angle_freq_ls = []
            for k in range(0, wind_samp.shape[2]):
                single_signal_chunk = wind_samp[j, :, k]

                fft_single_signal = fft.fft(single_signal_chunk)
                fft_ampl = np.abs(fft_single_signal)
                fft_ampl_filt = fft_ampl[1:num_top_freq+1]
                all_angle_freq_ls.append(fft_ampl_filt)
            all_angle_freq = np.stack(all_angle_freq_ls, axis=1)
            chunk_fft_ls.append(all_angle_freq)
        chunk_fft = np.stack(chunk_fft_ls, axis=0)
        data_fft_ls.append(chunk_fft)
    return data_fft_ls


class Clusterer:
    def __init__(self, step, common_FPS, rand_seed, cache):
        # self.__Tw = Tw
        self.__step = step
        self.__common_FPS = common_FPS
        self.__rand_seed = rand_seed

        self.cache = cache
        # self.K = K

        self.model_root_dir = os.path.join(get_root_path("cache"), "clustered_models")
        if not os.path.exists(self.model_root_dir):
            os.mkdir(self.model_root_dir)

    @property
    def rand_seed(self):
        return self.__rand_seed

    @property
    @abstractmethod
    def model_path(self):
        pass

    @property
    def common_fps(self):
        return self.__common_FPS

    def convert_to_timestamp(self, frame_id):
        td = timedelta(seconds=(frame_id / self.common_fps))
        minutes = int(td.seconds / 60)
        seconds = td.seconds % 60

        return minutes, seconds

    def window_single(self, data_sample, Tw, percent_keep=0.7):
        window_ls = []
        wind_id_ls = []

        # standard windows size defined by number of seconds per window (self.Tw) and frame rate (self.common FPS)
        standard_win_size = Tw * self.__common_FPS
        num_value_win = math.ceil(data_sample.shape[0] / self.__step)  # number of windows that will not be empty,
        # and have even one element from the sample
        num_partial_win = math.ceil(
            standard_win_size / self.__step)  # number of windows that will not have standard size
        num_full_win = num_value_win - num_partial_win  # number of windows that will be the full standard window length

        # split windows with number of overlap instances defined by self.step
        for i in range(0, num_value_win):
            start_idx = self.__step * i
            end_idx = start_idx + standard_win_size
            window = data_sample[start_idx:end_idx, :]
            window_ls.append(window)

            start_min, start_sec = self.convert_to_timestamp(start_idx)
            end_min, end_sec = self.convert_to_timestamp(end_idx)
            wind_id_ls.append([start_min, start_sec, end_min, end_sec])

        # keep only windows that are <percent_keep>% full
        j = num_full_win
        while j < len(window_ls):
            cur_win_size = window_ls[j].shape[0]
            if cur_win_size != standard_win_size:
                # if its length is less than <percent_keep>% of the standard window length, break out of the loop
                if cur_win_size < standard_win_size * percent_keep:
                    break
                # otherwise keep it and pad it with its mean
                else:
                    len_diff = standard_win_size - cur_win_size
                    padded_window = np.pad(window_ls[j], [(0, len_diff), (0, 0)], mode='mean')
                    window_ls[j] = padded_window
            j += 1
        window_ls = window_ls[0:j]  # keep only values that passed the above test, discard the ones whose window
        wind_id_ls = wind_id_ls[0:j]
        # size is less than <percent_keep> % of the standard

        windowed_array = np.stack(window_ls, axis=0)
        return windowed_array, wind_id_ls

    def window_data(self, data_ls, Tw):
        windowed_data_ls = []
        windows_idx_ls = []
        for i, data in enumerate(data_ls):
            windowed_array, wind_idx_ls = self.window_single(data, Tw)
            # in the list of timing ids - add video id in the beginning of the row
            for wind in wind_idx_ls:
                wind.insert(0, i)

            windowed_data_ls.append(windowed_array)
            windows_idx_ls.append(wind_idx_ls)
        return windowed_data_ls, windows_idx_ls

    @staticmethod
    def std_scale_matrix(matrix):
        # for each angle separately, standard scales within each 120-value window individually
        scaler = StandardScaler()

        scaled_ls = []
        # split by third dimension (three different angles)
        split_arr = np.split(matrix, 3, axis=2)
        for arr in split_arr:
            reshaped_slice = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])  # third dimension should be 1

            # this version would standardize each row (instance) independently
            # scaled_slice_inst = scaler.fit_transform(reshaped_slice.T).T
            # scaled_ls.append(scaled_slice_inst)

            # this version standardizes each column (feature) independently to 0 mean and 1 std dev
            scaled_slice_feat = scaler.fit_transform(reshaped_slice)
            scaled_ls.append(scaled_slice_feat)

        std_scaled = np.stack(scaled_ls, axis=2)
        # return std_scaled
        return std_scaled

    def compute_fit(self, kmeans, matrix, ext):

        kmeans_fit = kmeans.fit(matrix)
        self.save_model(kmeans_fit, ext)
        print(f"Saved model {self.model_path + ext}")
        return kmeans_fit

    def save_model(self, model, name_ext=""):
        return save_cpickle(self.model_path + name_ext, model)

    def load_model(self, name_ext=""):
        kmeans = load_cpickle(self.model_path + name_ext)
        return kmeans
