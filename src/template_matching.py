import json
import math
import os
from datetime import timedelta

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data

from clip_creator import ClipCreator
from head_pose_parser import HeadPoseParser
from skimage.feature import match_template
import seaborn as sns

from utilities import get_root_path


class TemplateMatching:
    def __init__(self):


        self.template_parser = HeadPoseParser(vid_folder="template_head_pose_videos",
                                              pose_folder="template_head_pose_signals", pose_ext=".csv", templ=True)
        # self.template_parser.plot_signals(self.template_parser.signal_ls, self.template_parser.vid_ids)
        self.dataset_parser = HeadPoseParser(vid_folder="sample_videos", pose_folder="sample_head_poses",
                                             pose_ext='.angles', templ=False)
        print("")


def get_time_match_results(dataset_signal_ls, dataset_vid_ids, templ_signal_ls, templ_vid_ids):
    matches = {}
    # iterate over template videos
    for i, templ_type in enumerate(templ_signal_ls):
        templ_vid_id = templ_vid_ids[i]
        movement_name = templ_vid_id[9:]
        # iterate over videos in dataset
        vid_dict = {}
        for j, video_sigs in enumerate(dataset_signal_ls):
            vid_name = dataset_vid_ids[j]
            vid_matches = []
            # iterate over transformations of template videos
            for k, templ_transf in enumerate(templ_type):
                # iterate over their transformations
                for l, vid_transf in enumerate(video_sigs):

                    # Match a template to a 2-D or 3-D image using normalized correlation. The output is an array with
                    # values between -1.0 and 1.0. The value at a given position corresponds to the correlation
                    # coefficient between the image and the template.
                    result = match_template(vid_transf, templ_transf)

                    threshold = 0.25
                    # thresh_loc = np.where(result >= threshold)
                    max_loc = np.argmax(result)
                    max_val = result[max_loc][0]

                    ss, se = find_seg_match(max_loc, templ_transf, vid_transf)
                    signal_match = vid_transf[ss: se, :]
                    ss_min, ss_sec = convert_to_timestamp(ss, fps=30)
                    se_min, se_sec = convert_to_timestamp(se, fps=30)

                    # TODO: double check whether we should keep original or padded segment
                    # increase time segment for context
                    # if ss_sec > 0:
                    #     dec_ss_sec = ss_sec - 1
                    # else:
                    #     dec_ss_sec = 0
                    # if se_sec < result.shape[0]:
                    #     inc_se_sec = se_sec + 1
                    # else:
                    #     inc_se_sec = result.shape[0]

                    sub_dict = {"score": max_val,
                                "segment": [(ss_min, ss_sec), (se_min, se_sec)],
                                "dataset_signal_match": signal_match.tolist(),
                                "template_signal": templ_transf.tolist()}
                    vid_matches.append(sub_dict)
                    # vid_matches.append([(ss_min, ss_sec), (se_min, se_sec)])
                    # vid_dict[vid_name] = sub_dict
                    vid_dict[vid_name] = vid_matches
        matches[movement_name] = vid_dict
    matches_json = json.dumps(matches)

    annot_file_path = os.path.join(get_root_path("data"), "clip_timestamps", "template_matching.json")
    with open(annot_file_path, "w") as outfile:
        outfile.write(matches_json)
    return matches


def find_seg_match(max_loc, templ, signal):
    len_seg = templ.shape[0]
    if max_loc < math.floor(len_seg/2):
        seg_start = 0
    else:
        seg_start = max_loc - math.floor(len_seg / 2)

    if max_loc + math.floor(len_seg/2) > signal.shape[0]:
        seg_end = signal.shape[0]
    else:
        seg_end = max_loc + len_seg - math.floor(len_seg/2)

    return seg_start, seg_end


def find_seg(max_loc, result):
    max_val = result[max_loc][0]
    seg_thresh = max_val * 0.5

    seg_start = max_loc
    seg_end = max_loc
    seg_start_val = result[seg_start][0]
    seg_end_val = result[seg_end][0]

    while (seg_start_val > seg_thresh) and (seg_start - 1 >= 0):
        seg_start = seg_start - 1
        seg_start_val = result[seg_start][0]

    while (seg_end_val > seg_thresh) and (seg_end + 2 <= result.shape[0]):
        seg_end = seg_end + 1
        seg_end_val = result[seg_end][0]

    final_ss = seg_start + 1
    final_se = seg_end - 1
    return final_ss, final_se


def convert_to_timestamp(frame_id, fps):
    td = timedelta(seconds=(frame_id / fps))
    minutes = int(td.seconds / 60)
    seconds = td.seconds % 60

    return minutes, seconds


def plot_signals(signal_ls, vid_ids):
    for i, signals_ind in enumerate(signal_ls):
        for k, transformed_sig in enumerate(signals_ind):
            # color theme
            sns.set_theme(style="whitegrid")
            color_palette = sns.color_palette('hls', 3)

            # define subplot grid
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 12))
            plt.subplots_adjust(hspace=0.5)
            fig.suptitle(f"Head Angle Poses: {vid_ids[i]}; Transform ID = {k}", fontsize=18, y=0.95)

            angle_labels = ["Yaw", "Pitch", "Roll"]
            x = np.arange(0, transformed_sig.shape[0], dtype=int)
            j = 0
            for column, ax in zip(transformed_sig.T, axs.ravel()):
                sns.lineplot(x=x, y=column, ax=ax, color=color_palette[j % 3])
                ax.set_ylabel(angle_labels[j])
                ax.set_ylim(-0.6, 0.6)
                j += 1
            fig.show()


def plot_matched_signals(matches):
    for movement_type, video_dict in matches.items():
        # extract signals to be plotted
        movement_type_raw_signals = []
        movement_type_templates = []
        for vid_name, match_inst in video_dict.items():
            # vid_info = seg_data[movement_type][vid_name]
            for clip_info in match_inst:
                dataset_signal = np.asarray(clip_info['dataset_signal_match'])
                movement_type_raw_signals.append(dataset_signal)
                templ_signal = np.asarray(clip_info['template_signal'])
                movement_type_templates.append(templ_signal)
        # some templates are repeated in the dictionary - keep only the unique ones for plotting
        L = {array.tostring(): array for array in movement_type_templates}
        unique_templates = list(L.values())
        print("")

        # create figure
        # color theme
        sns.set_theme(style="whitegrid")
        color_palette = sns.color_palette('hls', 3)

        # define subplot grid
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 12))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(f"Head Angle Poses: {movement_type}", fontsize=18, y=0.95)

        angle_names = ['Yaw', 'Pitch', 'Roll']
        xt = np.arange(0, unique_templates[0].shape[0], dtype=int)
        for signal in unique_templates:
            a = 0
            for column, ax in zip(signal.T, axs.ravel()):
                sns.lineplot(x=xt, y=column, ax=ax, color='black')
                ax.set_ylabel(angle_names[a])
                ax.set_ylim([-20, 20])
                a += 1
        xd = np.arange(0, movement_type_raw_signals[0].shape[0], dtype=int)
        for signal in movement_type_raw_signals:
            j = 0
            for column, ax in zip(signal.T, axs.ravel()):
                if column.shape[0] < xd.shape[0]:
                    diff = xd.shape[0] - column.shape[0]
                    column = np.pad(column, (0, diff), 'constant')
                sns.lineplot(x=xd, y=column, ax=ax, color=color_palette[j % 3])
                j += 1
        fig.show()
    return


def apply_morph_sig(signal_ls, original=1, eroded=0, dilated=0, filtered=0):
    transf_signal_arrays = []
    for i, head_pose in enumerate(signal_ls):
        single_signal = []
        kernel = np.ones((5, head_pose.shape[1]), np.uint8)
        if original:
            single_signal.append(head_pose)
        if eroded:
            eroded_sig = cv2.erode(head_pose, kernel, iterations=1)
            single_signal.append(eroded_sig)
        if dilated:
            dilated_sig = cv2.dilate(head_pose, kernel, iterations=1)
            single_signal.append(dilated_sig)
        if filtered:
            # for noise removal: erosion followed by dilation
            eroded_sig = cv2.erode(head_pose, kernel, iterations=1)
            dilated_eroded_sig = cv2.dilate(eroded_sig, kernel, iterations=1)
            single_signal.append(dilated_eroded_sig)
        transf_signal_arrays.append(single_signal)
    return transf_signal_arrays


def toy_tmp_match():
    image = data.coins()
    coin = image[170:220, 75:130]

    result = match_template(image, coin)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(coin, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    hcoin, wcoin = coin.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()


def main():
    template_matching = TemplateMatching()
    inital_templ_signals = template_matching.template_parser.signal_ls

    # correct order of head angles from ('pitch', 'yaw', 'roll') to ('yaw', 'pitch', 'roll') to match the dataset ones
    templ_signals = []
    for sig in inital_templ_signals:
        c_sig = sig[:, [1, 0, 2]]
        templ_signals.append(c_sig)
    templ_vids = template_matching.template_parser.vid_ids
    transformed_templ_signals = apply_morph_sig(templ_signals, original=1, eroded=0, dilated=0, filtered=0)

    plot_signals(transformed_templ_signals, templ_vids)

    dataset_signals = template_matching.dataset_parser.signal_ls
    dataset_vids = template_matching.dataset_parser.vid_ids
    transformed_dataset_signals = apply_morph_sig(dataset_signals, original=0, eroded=0, dilated=0, filtered=1)

    matches = get_time_match_results(transformed_dataset_signals, dataset_vids, transformed_templ_signals, templ_vids)
    plot_matched_signals(matches)

    clip_creator = ClipCreator(annot_file="template_matching.json")
    clip_creator.get_clips_from_videos()
    print("")
    # toy_tmp_match()


if __name__ == "__main__":
    main()
