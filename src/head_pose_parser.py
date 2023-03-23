import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns
from utilities import get_root_path
import numpy as np


# templ: bool showing if it is template
def get_file_data_cleaned(fp, templ, only_angles=True):
    MLid = int(os.path.basename(fp)[2:6])
    FPS = 30
    if MLid > 272:
        FPS = 60

    if templ:
        df = pd.read_csv(fp).iloc[:, 8:]
        x = df.to_numpy()
    if not templ and only_angles:
        x = np.loadtxt(fp)
        x = x[:, 0:3]

    cos_x = np.cos(x)

    # down-sample to 30 fps if necessary
    # if FPS == 60:
    #     downsample_ratio = 2
    #     x = temporal_rescale(x, downsample_ratio)

    return x


class HeadPoseParser:
    def __init__(self, vid_folder, pose_folder, templ=False, pose_ext=".angles"):
        self.pose_ext = pose_ext
        self.pose_dir = os.path.join(get_root_path("data"), pose_folder)
        for fname in os.listdir(self.pose_dir):
            if '.avi.' in fname:
                fpath_old = os.path.join(self.pose_dir, fname)
                fpath_new = os.path.join(self.pose_dir, fname.replace(".avi", ""))
                os.rename(fpath_old, fpath_new)

        self.video_dir = os.path.join(get_root_path("data"), vid_folder)
        self.head_pose_plot_dir = os.path.join(get_root_path("outputs"), "head_pose_plots")
        if not os.path.exists(self.head_pose_plot_dir):
            os.mkdir(self.head_pose_plot_dir)
        self.__signal_ls, self.__vid_ids = self.get_signals(templ)

    @property
    def signal_ls(self):
        return self.__signal_ls

    @property
    def vid_ids(self):
        return self.__vid_ids

    def get_signals(self, templ, only_angles=True):
        signal_ls = []
        vid_ids = []
        for hp_fname in os.listdir(self.pose_dir):
            hp_fpath = os.path.join(self.pose_dir, hp_fname)
            basename = os.path.basename(hp_fpath)
            vid_id = os.path.splitext(basename)[0]  # get basename without extension
            for vid_name in os.listdir(self.video_dir):
                if vid_id in vid_name:
                    v_fpath = os.path.join(self.video_dir, vid_name)
                    video_cap = cv2.VideoCapture(v_fpath)
                    # video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))
                    vid_length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    break
            assert os.path.isfile(hp_fpath) and hp_fpath.endswith(self.pose_ext), f"{hp_fpath} is not a file or not " \
                                                                                  f"the correct format {self.pose_ext}."
            print(f"Head Pose File: {hp_fpath}")

            # get the angle data and frame differential
            signals = get_file_data_cleaned(hp_fpath, templ, only_angles=only_angles)
            assert vid_length == signals.shape[0], "The video frame number does not match the head signal data length."
            signal_ls.append(signals)
            vid_ids.append(vid_id)
        return signal_ls, vid_ids

    def plot_signals(self, signal_ls, vid_ids):
        for i, signals_ind in enumerate(signal_ls):

            # color theme
            sns.set_theme(style="whitegrid")
            color_palette = sns.color_palette('hls', 3)

            # define subplot grid
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 12))
            plt.subplots_adjust(hspace=0.5)
            fig.suptitle("Head Angle Poses", fontsize=18, y=0.95)

            x = np.arange(0, signals_ind.shape[0], dtype=int)
            j = 0
            for column, ax in zip(signals_ind.T, axs.ravel()):
                sns.lineplot(x=x, y=column, ax=ax, color=color_palette[j % 3])
                j += 1
            fig.show()
            full_path = os.path.join(self.head_pose_plot_dir, f'{vid_ids[i]}.png')
            fig.savefig(full_path, format='png')
            print(f"Saved plot of most important features to '{full_path}'.")
        print("")

