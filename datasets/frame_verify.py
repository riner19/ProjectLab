import cv2
import os
from collections import OrderedDict


def navigate_video_smooth(video_path, start_frame, cache_size=100):
    if not os.path.exists(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = max(0, min(start_frame, total_frames - 1))

    # 1. Initialize our memory cache (OrderedDict helps us easily delete the oldest frames)
    frame_cache = OrderedDict()

    # 2. Seek to the very first requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()

    # OpenCV's internal playhead is now sitting at current_frame + 1
    cap_internal_pos = current_frame + 1

    if ret:
        # Save a clean copy of the frame to memory
        frame_cache[current_frame] = frame.copy()

    print(f"Loaded successfully. Total Frames: {total_frames}")
    print(f"Caching up to {cache_size} frames for smooth reverse playback.")
    print("Controls: Arrows/A/D to move, Q to quit.")

    while True:
        # Grab the frame from cache (or fallback to the last known frame)
        if current_frame in frame_cache:
            display_frame = frame_cache[current_frame].copy()
        else:
            display_frame = frame.copy()

        # Draw the UI
        cv2.putText(display_frame, f"Frame: {current_frame} / {total_frames - 1}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Smooth Frame-by-Frame Player", display_frame)

        # Wait for key press
        key = cv2.waitKeyEx(0)

        # Quit
        if key in (27, ord('q'), ord('Q')):
            print("Exiting...")
            break

        target_frame = current_frame

        # Next Frame
        if key in (2555904, 65363, 63235, ord('d'), ord('D')):
            target_frame = min(current_frame + 1, total_frames - 1)
        # Previous Frame
        elif key in (2424832, 65361, 63234, ord('a'), ord('A')):
            target_frame = max(current_frame - 1, 0)

        # If the user didn't move, just restart the loop
        if target_frame == current_frame:
            continue

        # --- THE SMOOTH FETCHING LOGIC ---
        if target_frame in frame_cache:
            # FAST PATH: Frame is in RAM. Load instantly.
            current_frame = target_frame
        else:
            # SLOW PATH: We have to read from the video file.

            # If the video playhead isn't naturally at the target frame, we MUST seek.
            # (If we are just moving forward 1 frame, we skip this slow step!)
            if target_frame != cap_internal_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                cap_internal_pos = target_frame

            ret, new_frame = cap.read()

            if ret:
                current_frame = target_frame
                frame = new_frame

                # Save to cache
                frame_cache[current_frame] = frame.copy()
                cap_internal_pos = current_frame + 1

                # If cache gets too big, delete the oldest frame to save RAM
                if len(frame_cache) > cache_size:
                    frame_cache.popitem(last=False)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_FILE = "RGB_videos/V10.mp4"
    STARTING_FRAME = 18703



    # You can increase cache_size if you have lots of RAM and want to scroll back further instantly
    navigate_video_smooth(VIDEO_FILE, STARTING_FRAME, cache_size=100)