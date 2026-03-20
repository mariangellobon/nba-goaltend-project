# This script helps user overlay graph of acceleration with video data
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import os
import datetime


# --- FILE PATHS ---
# NOTE: update these values to correct video and imu data path
video_path = '/Users/mariaangellobon/Desktop/Syncing Video Data/layup_and_pin_hard_dataset1.mov'
data_path = '/Users/mariaangellobon/Desktop/Syncing Video Data/layup_and_pin_hard_datset1.csv'


# -- LOAD & Preprocess CSV DATA ---
root_df = pd.read_csv(data_path)
# keep only the time and Z-axis acceleration columns
root_df = root_df[["Data Set 1:Time(s)", "Data Set 1:Z-axis acceleration 1(m/s²)"]].copy()
# --- align knock moment between IMU data and video ---
#TODO UPDATE THESE VALUES
imu_knock_time_s = 19.0502       # Time(s) value in the CSV at the knock moment (row 220)
video_knock_time = 14.6737       # seconds into the video where the knock occurs (shifted 3 frames later at 29.78fps)
should_rotate = False           # set to true if video is rotated 90 degrees (for vertical phone videos)


# Compute offset so IMU time aligns with video time
offset = video_knock_time - imu_knock_time_s
root_df["video_time"] = root_df["Data Set 1:Time(s)"] + offset


max_acceleration = root_df["Data Set 1:Z-axis acceleration 1(m/s²)"].max() #used for setting y axis graph
min_acceleration = root_df["Data Set 1:Z-axis acceleration 1(m/s²)"].min() #used for setting y axis lower bound
max_time = root_df["video_time"].max() # get the end time of video


# -- --- Video Setup ---
cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
#set widith and height of video
# Get original frame size
orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# If you're rotating 90°, width and height will flip. Only need to do this if rotating
if should_rotate:
    rotated_width = orig_height
    rotated_height = orig_width
else:
    rotated_width = orig_width
    rotated_height = orig_height


# Define scale factor to make video smaller
scale = .75
scale_width = int(rotated_width * scale)
scale_height = int(rotated_height * scale)


# define codec and create VideoWriter object to save video
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('/Users/mariaangellobon/Desktop/Syncing Video Data/goaltend_layup_and_pin_hard_dataset1_overlay3.mov', fourcc, round(fps / 2), (scale_width, scale_height))
# -- --- Video Processing Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        #This means video is broken
        break
    # Lets get the current frame number
    current_frame_number = cap.get(cv.CAP_PROP_POS_FRAMES)
    current_time = current_frame_number / fps # this lets us know what exact time we are in the video. Since video is 30fps
    # lets roate the video if verticle
    if should_rotate:
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.resize(frame, (scale_width, scale_height))
    else:
        frame = cv.resize(frame, (scale_width, scale_height))


    # Now lets get the IMU data up to this frame
    graph_df = root_df[root_df["video_time"] <= current_time]


    if not graph_df.empty:
        # Step 1: Create figure and attach canvas
        fig = Figure(figsize=(24, 5), dpi=100)
        canvas = FigureCanvas(fig)  # attach canvas to figure
        fig.patch.set_alpha(0.0)  # make figure background transparent


        # Step 2: Draw the graph
        ax = fig.add_subplot(111)
        ax.plot(graph_df["video_time"], graph_df["Data Set 1:Z-axis acceleration 1(m/s²)"], color='blue')
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.4)
        ax.set_xlim([14, 18])
        ax.set_ylim([min_acceleration - 5, max_acceleration + 5])
        ax.set_xlabel("Time (s)", fontsize=16, fontweight='bold')
        ax.set_ylabel("Z-axis Acceleration (m/s²)", fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=15, width=2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        tick_step = 1  # seconds


        ax.set_xticks(np.arange(14, 19, tick_step))
        ax.set_facecolor((0.85, 0.85, 0.85, 0.6))  # darker, less opaque background
        # ax.axis('off')


        # Step 3: Render to buffer
        canvas.draw()


        # Step 4: Get image buffer from the CANVAS (not the figure)
        graph_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        graph_img = graph_img.reshape(canvas.get_width_height()[::-1] + (4,))  # ARGB
        # Split RGB and alpha
        graph_rgb = graph_img[:, :, :3]
        alpha_mask = graph_img[:, :, 3] / 255.0  # normalize alpha to [0, 1]


        # region of video to insert graph (currently bottom left corner)
        gh, gw, _ = graph_rgb.shape
        insert_x, insert_y = 10, scale_height - gh - 10


        # Extract region of interest from video
        roi = frame[insert_y:insert_y+gh, insert_x:insert_x+gw].astype(np.float32)


        # Blend using alpha to overlay graph on video
        blended = roi * (1 - alpha_mask[..., None]) + graph_rgb.astype(np.float32) * alpha_mask[..., None]
        frame[insert_y:insert_y+gh, insert_x:insert_x+gw] = blended.astype(np.uint8)
 


    cv.imshow('Video', frame)
    # save to output video
    out.write(frame)
    # Wait for 1 ms and check if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# after finish always release video
cap.release()
out.release()
cv.destroyAllWindows()
