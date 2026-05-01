# ===========================================================================
# Video Frame Navigator
# ===========================================================================
# PURPOSE
#   Step through a video frame-by-frame to find the exact release frame
#   (the moment the ball leaves the shooter's hand). Once you find it,
#   press T to log the frame number and video timestamp.
#
# HOW TO RUN
#   python video_frame_navigator.py <path_to_your_video>
#
#   Example:
#     python videoFrameNavigator.py './syncing_video_data/ball_pin_backboard_hard1_dataset3.mov'
#     python videoFrameNavigator.py './syncing_video_data/layup_and_pin_hard_dataset1.mov'
#
# CONTROLS (also shown in the bar at the top of the video window)
#   Space  - play / pause the video
#   A      - step back one frame  (auto-pauses)
#   D      - step forward one frame (auto-pauses)
#   F      - cycle playback speed  (1x -> 2x -> 4x -> 8x -> 16x -> 1x …)
#   P      - save the current frame as a .jpg image (no overlay)
#   T      - mark the current frame as the RELEASE FRAME and save to txt
#   Q / ESC - quit
#
# OUTPUT FILES  (saved in the same folder as the video)
#   <video_name>_release_frames.txt
#       Tab-separated file with columns: frame_number  video_time
#       Each press of T appends a new row, so the last row is your
#       most recent selection. You can open this file in Excel or
#       any text editor.
#
#   frame_<N>.jpg  (only created when you press P)
#       A clean screenshot of that frame with no HUD overlay.
#
# TIPS
#   1. Play the video at higher speed (F) to get near the release,
#      then pause (Space) and use A/D to scrub frame-by-frame.
#   2. You can press T more than once -- every press adds a line.
#      Only the last line matters if you are revising your pick.
#   3. The top bar always shows your currently marked release frame
#      so you can confirm before quitting.
# ===========================================================================

# pip install -r requirements.txt


import cv2
import sys
import os

SPEEDS = [1, 2, 4, 8, 16]
WINDOW_NAME = 'Video Frame Navigator'


def frame_to_timestamp(frame_number, fps):
    """Convert a 0-based frame number to MM:SS.mmm timestamp."""
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"


def draw_hud(frame, current_frame, total_frames, fps, speed_index, playing, release_frame):
    """Draw a top HUD bar with frame info and hotkey legend."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = max(0.7, w / 1920)
    thick = max(1, int(scale * 2.5))
    font_scale = scale * 0.75
    bar_h = int(60 * scale)

    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)

    timestamp = frame_to_timestamp(current_frame, fps)
    state = "PLAY" if playing else "PAUSE"
    rel_tag = f" | Release: {release_frame}" if release_frame is not None else ""

    hud_text = (
        f"Frame {current_frame + 1}/{total_frames}  {timestamp}  "
        f"{SPEEDS[speed_index]}x  [{state}]{rel_tag}  |  "
        f"Space:play/pause  A/D:frame  F:speed  P:save img  T:release  Q:quit"
    )
    cv2.putText(frame, hud_text, (10, bar_h - int(14 * scale)),
                font, font_scale, (255, 255, 255), thick)


def save_release(video_path, frame_number, fps):
    """Append the release-frame record to a txt file next to the video."""
    base = os.path.splitext(video_path)[0]
    out_path = base + "_release_frames.txt"
    timestamp = frame_to_timestamp(frame_number, fps)

    is_new = not os.path.exists(out_path)
    with open(out_path, "a") as f:
        if is_new:
            f.write("frame_number\tvideo_time\n")
        f.write(f"{frame_number}\t{timestamp}\n")

    print(f"Release frame saved -> frame {frame_number}, time {timestamp}  ({out_path})")
    return out_path


def main():
    if len(sys.argv) != 2:
        print("Usage: python video_frame_navigator.py <video_file_path>")
        print("Example: python video_frame_navigator.py video.mp4")
        return

    video_path = sys.argv[1]
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    
    # Open video file
    cap = cv2.VideoCapture(video_path)

    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video loaded: {video_path}")
    print(f"Total frames: {total_frames}  |  FPS: {fps}")
    print("\nControls (also shown in the top bar):")
    print("  Space  - play / pause")
    print("  A / D  - previous / next frame")
    print("  F      - cycle playback speed (1x -> 2x -> 4x -> 8x -> 16x)")
    print("  P      - save current frame as image")
    print("  T      - mark current frame as the release frame (saved to txt)")
    print("  Q / ESC - quit")

    current_frame = 0
    playing = False
    speed_index = 0
    release_frame = None

    save_img_dir = os.path.dirname(os.path.abspath(video_path))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1080, 720)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        draw_hud(frame, current_frame, total_frames, fps, speed_index, playing, release_frame)
        cv2.imshow(WINDOW_NAME, frame)

        wait_ms = max(1, int(1000 / fps / SPEEDS[speed_index])) if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('d'):
            playing = False
            if current_frame < total_frames - 1:
                current_frame += 1

        elif key == ord('a'):
            playing = False
            if current_frame > 0:
                current_frame -= 1

        elif key == ord(' '):
            playing = not playing
            print("Playing" if playing else "Paused")

        elif key == ord('f'):
            speed_index = (speed_index + 1) % len(SPEEDS)
            print(f"Speed set to {SPEEDS[speed_index]}x")

        elif key == ord('p'):
            img_name = f"frame_{current_frame + 1}.jpg"
            img_path = os.path.join(save_img_dir, img_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret_save, clean_frame = cap.read()
            if ret_save:
                cv2.imwrite(img_path, clean_frame)
                print(f"Frame image saved -> {img_path}")

        elif key == ord('t'):
            release_frame = current_frame + 1
            save_release(video_path, release_frame, fps)

        if playing and current_frame < total_frames - 1:
            current_frame += SPEEDS[speed_index]
            current_frame = min(current_frame, total_frames - 1)
        elif playing and current_frame >= total_frames - 1:
            playing = False
            print("End of video - playback stopped")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()