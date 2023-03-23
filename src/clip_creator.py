import os.path

from utilities import get_root_path
from moviepy.editor import *
import json


# Press the green button in the gutter to run the script.
class ClipCreator:
    def __init__(self, annot_file):
        self.score_lim = 0.01
        self.annot_file_path = os.path.join(get_root_path("data"), "clip_timestamps", annot_file)
        self.annot_json = open(self.annot_file_path)
        # self.dataset_info = os.path.join(get_root_path("data"), dataset_info)

        # TODO: change to full video dataset (interested)
        # self.video_dir = os.path.join(get_root_path("data"), "interested")
        self.video_dir = os.path.join(get_root_path("data"), "sample_videos")

        self.clips_dir = os.path.join(get_root_path("outputs"), "clips")
        if not os.path.exists(self.clips_dir):
            os.mkdir(self.clips_dir)

    def get_clips_from_videos(self):
        seg_data = json.load(self.annot_json)

        for movement_type in seg_data:
            save_dir = os.path.join(self.clips_dir, movement_type)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            video_files = []
            for v_fname in os.listdir(self.video_dir):
                v_fpath = os.path.join(self.video_dir, v_fname)

                assert os.path.isfile(v_fpath) and v_fpath.endswith('.mp4'), f"{v_fpath} is not a file or not the " \
                                                                             f"correct video format (.mp4). "
                video_files.append(v_fpath)

                # TODO: change with real ID
                vid_id = v_fname[0:8]

                # create directory that will contain the clips from this particular video ID
                # save_dir = os.path.join(self.clips_dir, vid_id)
                # if not os.path.exists(save_dir):
                #     os.mkdir(save_dir)

                # getting segment and score information from annotation file for the particular video ID
                # vid_info = seg_data['results']['--1DO2V4K74']
                vid_info = seg_data[movement_type][vid_id]
                # loading video
                full_video = VideoFileClip(v_fpath)

                for clip_info in vid_info:
                    score = clip_info['score']
                    segment = clip_info['segment']

                    # TODO: remove
                    # segment = [10, 13]
                    print(f"Score: {score}; segment: {segment}")
                    if score > self.score_lim:
                        print(f"Including segment {segment} with score {score} as potentially relevant.")
                        # getting only segment
                        clip = full_video.subclip(tuple(segment[0]), tuple(segment[1]))

                        seg_name = f"[{segment[0][0]}_{segment[0][1]}]_[{segment[1][0]}_{segment[1][1]}]"
                        # clip_name = f'{vid_id}_{segment[0]}_{segment[1]}.mp4'
                        clip_name = f'{vid_id}_{seg_name}.mp4'
                        clip_fp = os.path.join(save_dir, clip_name)
                        clip.write_videofile(clip_fp)

                        # showing  clip - just for testing purposes
                        # resize clip (keeping aspect ratio) and display it in the center
                        # clip = clip.fx(vfx.resize, width=460).set_pos('center')
                        # clip.preview(fps=30, audio=False)

                full_video.close()
        # print(f"\nA total of {len(video_files)} videos are being considered.")


def main():
    annot_file_name = 'results.json'
    # dataset_info_f = 'activity_net.v1-3.min.json'
    # clip_creator = ClipCreator(annot_file_name, dataset_info_f)
    clip_creator = ClipCreator(annot_file_name)
    clip_creator.get_clips_from_videos()


if __name__ == '__main__':
    main()
