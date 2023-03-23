import json
import math
import os
import statistics

import numpy as np
from sklearn import metrics, manifold
from sklearn.cluster import DBSCAN, KMeans

from clip_creator import ClipCreator
from head_pose_parser import HeadPoseParser
from src.clusterer import Clusterer, perform_fft

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import plotly.express as px

from dtaidistance import dtw, dtw_ndim
import array
from dtaidistance import dtw_visualisation as dtwvis
from template_matching import TemplateMatching, get_time_match_results, plot_matched_signals
from scipy import stats

from utilities import get_root_path


def create_chunks(signal_ls, clusterer, tw, num_top_freq):
    time_wx_ls, window_id_ls = clusterer.window_data(signal_ls, tw)

    # standard scale time windows by instance (row)
    feat_scaled_w_ls = []
    for w in time_wx_ls:
        feat_scaled_w = clusterer.std_scale_matrix(w)
        feat_scaled_w_ls.append(feat_scaled_w)

    # convert data list to fft
    wx_fft_ls = perform_fft(time_wx_ls, num_top_freq)

    # standard scale fft windows by feature
    scaled_fft_ls = []
    for f in wx_fft_ls:
        feat_scaled_f = clusterer.std_scale_matrix(f)
        scaled_fft_ls.append(feat_scaled_f)

    # concatenate information from each video in each of the 4 representations
    time_wx = np.concatenate(time_wx_ls, axis=0)
    time_wx_r = convert_to_arr(time_wx_ls)

    scaled_time_wx = np.concatenate(feat_scaled_w_ls, axis=0)
    scaled_time_wx_r = convert_to_arr(feat_scaled_w_ls)

    wx_fft = np.concatenate(wx_fft_ls, axis=0)
    wx_fft_r = convert_to_arr(wx_fft_ls)

    scaled_fft = np.concatenate(scaled_fft_ls, axis=0)
    scaled_fft_r = convert_to_arr(scaled_fft_ls)

    window_ids = np.concatenate(window_id_ls, axis=0)

    # return window_ids, time_wx_r, scaled_time_wx_r, wx_fft_r, scaled_fft_r
    return window_ids, time_wx, scaled_time_wx, wx_fft, scaled_fft


def convert_to_arr(signal_ls):
    concat = np.concatenate(signal_ls, axis=0)
    concat_transp = concat.transpose(0, 2, 1)
    arr = concat_transp.reshape(concat_transp.shape[0], -1)
    return arr


def toy_dbscan():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
    X = StandardScaler().fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    print("")

    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14, )
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6, )
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()
    print("")


def toy_fft():
    t = np.arange(256)
    sp = np.fft.fft(np.sin(t))
    freq = np.fft.fftfreq(t.shape[-1])
    r = sp.real
    i = sp.imag
    plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()


def get_time_segments(core_sample_indices, all_window_ids, video_ids):
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Potentially relevant samples:")
    for sample_idx in core_sample_indices:
        sample_coord = all_window_ids[sample_idx, :]

        vid_id, start_min, start_sec, end_min, end_sec = sample_coord

        print(f"'{video_ids[vid_id]}' : {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


def find_optimal_dbscan_params(data_arr, cluster_min=1, eps_start=0.05, eps_end=1, eps_step=0.05, min_samp_st=10,
                               min_samp_end=2000,
                               min_samp_step=10):
    best_score = -1
    best_eps = 0
    best_min_sampl = 0
    for eps in np.arange(eps_start, eps_end, step=eps_step):
        for min_samp in range(min_samp_st, min_samp_end, min_samp_step):
            db = DBSCAN(eps=eps, min_samples=min_samp).fit(data_arr)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_ < cluster_min:
                pass
                # print(f"Exclude eps: {eps} and min_sample: {min_samp} combination.")
            else:
                # Number of clusters in labels, ignoring noise if present.
                # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                # n_noise_ = list(labels).count(-1)
                sil_score = metrics.silhouette_score(data_arr, labels)
                if sil_score > best_score:
                    print(f"Best score so far: eps: {eps}, min_samples: {min_samp}, num_clusters: {n_clusters_},"
                          f" score: {sil_score}")
                    best_score = sil_score
                    best_eps = eps
                    best_min_sampl = min_samp
    return best_score, best_eps, best_min_sampl


def dbscan_cluster(fft_data_arr, eps, min_samples):
    dist_ls = []
    for i in range(0, fft_data_arr.shape[0] - 1):
        dist_example = math.dist(fft_data_arr[-1, :], fft_data_arr[i, :])
        dist_ls.append(dist_example)
    mean = statistics.mean(dist_ls)
    std_dev = statistics.stdev(dist_ls)
    median = statistics.median(dist_ls)
    minimum = min(dist_ls)
    maximum = max(dist_ls)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(fft_data_arr)
    # db = DBSCAN(eps=300, min_samples=100).fit(fft_data_arr)
    # db = DBSCAN(eps=0.4, min_samples=100).fit(fft_data_arr)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print(f"Silhouette Coefficient: {metrics.silhouette_score(fft_data_arr, labels):.3f}")

    unique_labels = set(labels)
    print(f"Unique Labels: {unique_labels}")
    idx = db.core_sample_indices_
    return idx


def kmeans_get_ids(all_data_arr, kmeans, all_window_ids, video_ids):
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Potentially relevant samples:")

    # Compute clustering and transform X to cluster-distance space from each label (cluster center)
    data_dist = kmeans.fit_transform(all_data_arr)
    # Get the distance from the particular label assigned by clustering
    data_cluster_dist = data_dist[np.arange(all_data_arr.shape[0]), kmeans.labels_]

    label_id_dict = {}
    # For each cluster, keep the top percentile_closest
    percentile_closest = 90
    for i in range(kmeans.n_clusters):
        in_cluster = (kmeans.labels_ == i)
        cluster_dist = data_cluster_dist[in_cluster]
        cutoff_dist = np.percentile(cluster_dist, percentile_closest)
        above_cutoff = (data_cluster_dist > cutoff_dist)

        label_id_dict[i] = [j for j, x in enumerate(above_cutoff) if x]
        data_cluster_dist[in_cluster & above_cutoff] = -1
    # part_prop = (data_cluster_dist != -1)
    # inst = all_data_arr[part_prop]

    for cluster_id, label_ids in label_id_dict.items():
        print(f"Cluster ID: {cluster_id}")
        get_time_segments(label_ids, all_window_ids, video_ids)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    return


def viz(data_arr):
    t_sne = manifold.TSNE(
        n_components=3,
        perplexity=20,
        init="random",
        n_iter=250,
        random_state=0,
    )
    projected_data = t_sne.fit_transform(data_arr)

    fig = px.scatter_3d(
        projected_data, x=0, y=1, z=2,
    )
    fig.update_traces(marker_size=8)
    fig.show()
    # plot_3d(projected_data, "T-SNE")
    print("")


def kmeans_cluster(data_arr, rand_state=1907):
    K_ls = [3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
    best_K = -1
    best_score = -1
    for K in K_ls:
        kmeans = KMeans(n_clusters=K, random_state=rand_state, n_init="auto").fit(data_arr)
        sil_sc = metrics.silhouette_score(data_arr, kmeans.labels_)
        print(f"Silhouette Coefficient: {sil_sc:.3f} for K = {K}")
        if sil_sc > best_score:
            best_score = sil_sc
            best_K = K
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Maximum Silhouette Coefficient: {best_score:.3f} for K = {best_K}")
    return best_K, best_score


def compute_template_dtw(time_signals, window_ids, dataset_vid_ids, template_ls, template_vid_ids, percent_keep_ls):
    matches = {}
    # all_templ_dist = []
    for i, templ in enumerate(template_ls):
        vid_dict = {}
        templ_vid_id = template_vid_ids[i]
        movement_name = templ_vid_id[9:]
        dist_ls = []
        for j, inst in enumerate(time_signals):
            # templ_reshaped = templ.reshape(-1, 1, order='F').astype(np.double)  # reshape by column
            # inst_reshaped = inst.reshape(inst.shape[0], 1).astype(np.double)

            # to focus on the shape rather than the absolute difference and offset, z-score normalize
            # templ_z = stats.zscore(templ_reshaped)
            # inst_z = stats.zscore(inst_reshaped)
            # distance = dtw.distance(inst_z, templ_z)

            templ_z = stats.zscore(templ)
            inst_z = stats.zscore(inst)
            distance = dtw_ndim.distance(inst_z, templ_z)
            dist_ls.append(distance)

        print(f"The maximum DTW distance with this template is {max(dist_ls)}")
        print(f"The minimum DTW distance with this template is {min(dist_ls)}")
        print(f"The mean DTW distance with this template is {statistics.mean(dist_ls)}")
        percentile = percent_keep_ls[i]
        threshold = np.percentile(dist_ls, percentile)
        print(f"The threshold ({percentile}-th percentile) inclusion for with this template is {threshold}\n")

        # after computing the all distances, select the lower ones
        closest_ids = []
        vid_names = []
        for d, dist in enumerate(dist_ls):
            if dist < threshold:
                # obtain the id of the video and the start and end time of the selected window
                vid_id, start_min, start_sec, end_min, end_sec = window_ids[d, :]
                vid_name = dataset_vid_ids[vid_id]
                vid_names.append(vid_name)

                segment = [(start_min, start_sec), (end_min, end_sec)]

                sub_dict = {"score": dist,
                            "segment": segment,
                            "dataset_signal_match": time_signals[d],
                            "template_signal": templ}
                closest_ids.append(sub_dict)

        # go through the list of selected windows and group them by video (for consistency with other methods)
        all_vid_matches_d = {}
        for v, v_name in enumerate(dataset_vid_ids):
            v_dict_ls = []
            for sv, s_vid in enumerate(vid_names):
                if s_vid == v_name:
                    v_dict_ls.append(closest_ids[sv])
            all_vid_matches_d[v_name] = v_dict_ls
        matches[movement_name] = all_vid_matches_d

    print("")
    matches_json = json.dumps(matches, cls=NpEncoder)

    annot_file_path = os.path.join(get_root_path("data"), "clip_timestamps", "template_matching_dtw.json")
    with open(annot_file_path, "w") as outfile:
        outfile.write(matches_json)
    return matches


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def main():
    # toy_fft()
    # toy_dbscan()
    n_top_freq = 15
    head_pose_parser = HeadPoseParser(vid_folder="sample_videos", pose_folder="sample_head_poses")
    template_parser = HeadPoseParser(vid_folder="template_head_pose_videos",
                                     pose_folder="template_head_pose_signals", pose_ext=".csv", templ=True)
    templ_signals = template_parser.signal_ls
    templ_vid_ids = template_parser.vid_ids

    inactive_clusterer = Clusterer(step=8, common_FPS=30, rand_seed=1907, cache=False)
    # time windows in seconds
    time_windows = [1, 2, 3, 4]

    # timing information of all samples
    all_window_ids = []
    # collecting fft version of same signals (videos) for different time windows
    t_signals = []
    scaled_t_signals = []
    fft_signals = []
    scaled_fft_signals = []

    norm_t_signals = []
    norm_scaled_t_signals = []
    norm_fft_signals = []
    norm_scaled_fft_signals = []
    for tw in time_windows:
        # each row in window_ids is structured like this: <start minute>, <start second>, <end minute>, <end second>
        window_id_ls, time_wx, scaled_time_wx, wx_fft, scaled_fft = create_chunks(head_pose_parser.signal_ls,
                                                                                  inactive_clusterer, tw,
                                                                                  num_top_freq=n_top_freq)

        #
        percent_keep_ls = [1, 0.5]
        dtw_dist_matches = compute_template_dtw(time_wx, window_id_ls, head_pose_parser.vid_ids, templ_signals,
                                                templ_vid_ids, percent_keep_ls)
        plot_matched_signals(dtw_dist_matches)
        clip_creator = ClipCreator(annot_file="template_matching_dtw.json")
        clip_creator.get_clips_from_videos()

        # all_window_ids.append(window_id_ls)
        #
        # t_signals.append(time_wx)
        # scaled_t_signals.append(scaled_time_wx)
        # fft_signals.append(wx_fft)
        # scaled_fft_signals.append(scaled_fft)
        #
        # normalize each frequency row so that its norm equals 1 (all three angles together) - proper for clustering
        # norm_time_wx = Normalizer().fit_transform(time_wx)
        # norm_t_signals.append(norm_time_wx)
        #
        # norm_scaled_time_wx = Normalizer().fit_transform(scaled_time_wx)
        # norm_scaled_t_signals.append(norm_scaled_time_wx)

        # norm_fft_wx = Normalizer().fit_transform(wx_fft)
        # norm_fft_signals.append(norm_fft_wx)
        #
        # norm_scaled_fft_wx = Normalizer().fit_transform(scaled_fft)
        # norm_scaled_fft_signals.append(norm_scaled_fft_wx)
    #
    window_id_arr = np.concatenate(all_window_ids, axis=0)
    # --------------------------------------------------------------------------------------------------------------
    all_time_signals = [t_signals, scaled_t_signals, norm_t_signals, norm_scaled_t_signals]

    # fft window arrays
    fft_arr = np.concatenate(fft_signals, axis=0)
    scaled_fft_arr = np.concatenate(scaled_fft_signals)
    norm_fft_arr = np.concatenate(norm_fft_signals)
    norm_scaled_fft_arr = np.concatenate(norm_scaled_fft_signals)

    wx_fft_ls = perform_fft(templ_signals, num_top_freq=n_top_freq)

    # tm = TemplateMatching()
    print("")

    # viz(norm_scaled_fft_arr)
    # best_K, best_score = kmeans_cluster(fft_arr)
    # kmeans = KMeans(n_clusters=best_K, random_state=1907, n_init="auto")
    # kmeans_get_ids(fft_arr, kmeans, all_window_ids, head_pose_parser.vid_ids)

    # sampling_freq = clusterer.common_fps
    # for i, vid in enumerate(head_pose_parser.signal_ls):
    #     nfft = None
    #     hash_matcher = HashMatcher(vid, head_pose_parser.vid_ids[i], sampling_freq, nfft)
    # print("")


if __name__ == '__main__':
    main()
