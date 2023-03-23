"""
Created: 1/31/19
© Denisa Qori 2019 All Rights Reserved
"""

import numpy as np
import matplotlib.pyplot as plt
from src.frequency_matching.hash_generator import HashGenerator
from src.frequency_matching.spectrogram_filterer import Spectrogram_Filterer
from collections import Counter
import seaborn as sns


class HashMatcher:
    def __init__(self, data_ls, vid_id, fs, nfft):

        self.hashes, self.categories = self.create_hash_dataset(data_ls, vid_id, fs, nfft)
        # self.score_all_signals(hashes, categories)

    # TODO: do this with every head angle - right now it is only with the first one.
    def create_hash_dataset(self, data_ls, vid_id, fs, nfft):
        # list of lists
        all_hashes_cat = []
        hashes_data_list = []

        print(f"Generating the spectrogram for video: {vid_id}")
        for i, signal in enumerate(data_ls.T):
            plt.figure(1)
            # spectrum, freqs, t, im = plt.specgram(cat_data[:, 0], NFFT=nfft, Fs=fs, noverlap=0)
            # signal = cat_data[:, 0]
            spectrum, freqs, t, im = plt.specgram(signal, NFFT=nfft, Fs=fs, noverlap=0)

            plt.colorbar(im).set_label('Amplitude (dB)')
            plt.xlabel("Time (sec)", fontsize=10, labelpad=10)
            plt.ylabel("Frequency (Hz)", fontsize=10, labelpad=10)
            plt.title(f"{vid_id}: Angle {i + 1}")

            sf = Spectrogram_Filterer(spectrum)
            bands = sf.group_into_freq_bands(spectrum, num_bands=5)
            time_axis, freq_axis = sf.get_strongest_freq(bands, num_strongest=10)

            plt.figure(2)
            filtered_spec = sns.scatterplot(x=time_axis, y=freq_axis)
            plt.show()

            #TODO: check and redo possibly
            hash_gen = HashGenerator(time_axis, freq_axis, F=10)
            hashes = hash_gen.hashes

            hashes_data_list.append(hashes)

        all_hashes = np.stack(hashes_data_list, axis=0)
        return all_hashes, all_hashes_cat

    def score_single_signal(self, test_signal, index_dataset):
        hashes = index_dataset[0]
        index_hashes = np.reshape(hashes, (hashes.shape[0] * hashes.shape[1], hashes.shape[2]))
        categories = [cat for subcat in index_dataset[1] for cat in subcat]

        test_hashes = test_signal[0]
        test_cat = test_signal[1][0]  # category is the same all over

        # unique_hashes = np.unique(hashes, axis=0)
        # num_unique = unique_hashes.shape[0]

        db_cat = []
        for i in range(1, test_hashes.shape[0]):
            # db_idx = []
            for j in range(0, index_hashes.shape[0]):
                if np.array_equal(test_hashes[i, :], index_hashes[j, :]):
                    # db_idx.append(j)
                    print("length of categories: ", len(categories))
                    print("j: ", j)
                    db_cat.append(categories[j])
        cat_count = Counter(db_cat)

        print(test_cat, " - ", cat_count)
        return test_cat, cat_count

    def score_all_signals(self, hashes, categories):

        real_cat = []
        retured_count = []
        for i in range(0, hashes.shape[0]):
            print("i: ", i)
            test_hashes = hashes[i, :, :]
            test_cat = categories[i]

            index_dataset = np.delete(hashes, obj=i, axis=0)
            del categories[i]

            # index_dataset_hashes = hashes[1:, :, :]
            # index_dataset_categories = categories[1:]
            test_cat, cat_count = self.score_single_signal(test_signal=[test_hashes, test_cat],
                                                           index_dataset=[index_dataset, categories])
            real_cat.append(test_cat)
            retured_count.append(cat_count)
        print("")
