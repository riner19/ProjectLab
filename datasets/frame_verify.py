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

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    current_frame = max(0, min(start_frame, total_frames - 1))

    # 1. Initialize memory cache
    frame_cache = OrderedDict()

    # 2. Seek to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()

    cap_internal_pos = current_frame + 1

    if ret:
        frame_cache[current_frame] = frame.copy()

    print(f"Source Resolution: {width}x{height}")
    print(f"Loaded successfully. Total Frames: {total_frames}")
    print(f"Caching up to {cache_size} frames for smooth reverse playback.")
    print("Controls: Arrows/A/D to move, Q to quit.")

    # --- WINDOW CONFIGURATION ---
    # We use WINDOW_NORMAL so the window size is independent of the image resolution
    window_name = "Smooth Frame-by-Frame Player (Full HD Internal)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Resize the window to a "Small Window" (e.g., 960x540)
    # The internal data remains 1920x1080 (Full HD)
    cv2.resizeWindow(window_name, 960, 540)

    while True:
        # Grab the frame from cache or fallback
        if current_frame in frame_cache:
            display_frame = frame_cache[current_frame].copy()
        else:
            # Fallback if current_frame isn't cached (safety)
            display_frame = frame.copy()

        # Draw the UI on the Full HD frame
        # (Text will look crisp because it's drawn on the high-res buffer)
        cv2.putText(display_frame, f"Frame: {current_frame} / {total_frames - 1}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(display_frame, f"Res: {width}x{height}",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Show the frame in our resized window
        cv2.imshow(window_name, display_frame)

        # Wait for key press
        key = cv2.waitKeyEx(0)

        # Quit
        if key in (27, ord('q'), ord('Q')):
            print("Exiting...")
            break

        target_frame = current_frame

        # Next Frame (Right Arrow or D)
        if key in (2555904, 65363, 63235, ord('d'), ord('D')):
            target_frame = min(current_frame + 1, total_frames - 1)
        # Previous Frame (Left Arrow or A)
        elif key in (2424832, 65361, 63234, ord('a'), ord('A')):
            target_frame = max(current_frame - 1, 0)

        if target_frame == current_frame:
            continue

        # --- CACHING LOGIC ---
        if target_frame in frame_cache:
            current_frame = target_frame
        else:
            if target_frame != cap_internal_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                cap_internal_pos = target_frame

            ret, new_frame = cap.read()

            if ret:
                current_frame = target_frame
                frame = new_frame
                frame_cache[current_frame] = frame.copy()
                cap_internal_pos = current_frame + 1

                if len(frame_cache) > cache_size:
                    frame_cache.popitem(last=False)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ensure this path is correct for your local machine
    VIDEO_FILE = "RGB_videos_high_quality/V7.mp4"
    STARTING_FRAME = 1500


    navigate_video_smooth(VIDEO_FILE, STARTING_FRAME, cache_size=150)