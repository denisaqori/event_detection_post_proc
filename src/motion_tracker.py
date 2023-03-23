import os
import datetime
import imutils
import cv2
import numpy as np

from utilities import get_root_path


class MotionTracker:
    def __init__(self, diff_threshold=10, min_area=25):
        # TODO: change to full video dataset (interested)
        # self.video_dir = os.path.join(get_root_path("data"), "interested")
        self.video_dir = os.path.join(get_root_path("data"), "sample_videos")
        self.diff_threshold = diff_threshold
        self.min_area = min_area

    def detect_motion(self):
        video_dict = {}
        for v_fname in os.listdir(self.video_dir):
            vid_name = v_fname[0:8]
            v_fpath = os.path.join(self.video_dir, v_fname)
            assert os.path.isfile(v_fpath) and v_fpath.endswith('.mp4'), f"{v_fpath} is not a file or not the " \
                                                                         f"correct video format (.mp4). "
            vid = cv2.VideoCapture(v_fpath)
            print(f"Video File: {v_fname}")

            # Time of movement
            motion_list = []

            previous_frame = None
            while True:
                motion = 0
                frame = vid.read()[1]
                text = 'still'

                # if the frame could not be grabbed, then we have reached the end of the video
                if frame is None:
                    break

                # resize the frame, convert it to grayscale, and blur it
                frame = imutils.resize(frame, width=500)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                # if the first frame is None, initialize it
                if previous_frame is None:
                    previous_frame = gray
                    continue

                # calculate difference and update previous frame
                diff_frame = cv2.absdiff(src1=previous_frame, src2=gray)
                previous_frame = gray

                # dilute the image a bit to make differences more visible; more suitable for contour detection
                kernel = np.ones((5, 5))
                diff_frame = cv2.dilate(diff_frame, kernel, iterations=2)

                # Only take different areas that are different enough (>20 / 255)
                thresh_frame = cv2.threshold(src=diff_frame, thresh=self.diff_threshold, maxval=255, type=cv2.THRESH_BINARY)[1]

                cnts = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                # loop over the contours
                for c in cnts:
                    # if the contour is too small, ignore it
                    if cv2.contourArea(c) < self.min_area:
                        continue
                    motion = 1
                    # compute the bounding box for the contour, draw it on the frame,
                    # and update the text
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = "motion"

                    # draw the text and timestamp on the frame
                    cv2.putText(frame, "Status: {}".format(text), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    # show the frame and record if the user presses a key
                    cv2.imshow("Original", frame)
                    cv2.imshow("Threshold", thresh_frame)
                    cv2.imshow("Frame Difference", diff_frame)

                # Appending status of motion
                motion_list.append(motion)

                key = cv2.waitKey(1)
                # if q entered whole process will stop
                if key == ord('q'):
                    break

            # cleanup the camera and close any open windows
            vid.release()
            cv2.destroyAllWindows()
            video_dict[vid_name] = motion_list
        print("")
        return video_dict

    def convert_frames_to_secs(self, fps, motion_ls):
        seg_ls = []

        i = 0
        j = 0
        while i < len(motion_ls):
            if motion_ls[i] == 1:
                # get duration of event (number of times the same value is repeated)
                j = i
                while j < len(motion_ls) and motion_ls[j] == 1:
                    j += 1

                seg_ls.append([i, j])
                i = j - 1
            i += 1
        print("Converted")
        return seg_ls

    def create_segment_annotations(self, video_dict, fps=30):
        new_vid_d = {}
        for vid_name, motion_ls in video_dict.items():
            new_vid_d[vid_name] = self.convert_frames_to_secs(fps, motion_ls)
        print("")


def main():
    motion_tracker = MotionTracker()
    video_dict = motion_tracker.detect_motion()
    motion_tracker.create_segment_annotations(video_dict)


if __name__ == '__main__':
    main()
