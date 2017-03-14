from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
# from Tracker import Tracker

def process_video(process_image, params_file, in_vid, out_vid):
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load( open(params_file, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    clip = VideoFileClip(in_vid)
    video_clip = clip.fl_image(process_image) # function expects color image
    video_clip.write_videofile(out_vid, audio=False)
