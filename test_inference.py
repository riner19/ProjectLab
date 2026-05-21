import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from ultralytics import YOLO

# Import your trained architecture directly from your training script
from train_tcn import EdgeBoxingTCN

# ==========================================
# --- Configuration ---
# ==========================================
VIDEO_PATH = 'datasets/RGB_videos_720p/V1.mp4'
#VIDEO_PATH = 'guard/guard1.mp4'
MODEL_WEIGHTS = 'edge_boxing_tcn.pth'
YOLO_MODEL = 'yolo26n-pose.pt'

# Playback Modifications
START_TIME_SECONDS = 300  # Skip the first 60 seconds of the video
PLAYBACK_SPEED = 2  # 1.0 is normal speed, 0.5 is half-speed (slow motion)

T_MAX = 30
CONFIDENCE_THRESHOLD = 0.70

CLASS_MAP = {
    0: 'Idle/Guard',
    1: 'Jab',
    2: 'Cross',
    3: 'Lead Hook',
    4: 'Rear Hook',
    5: 'Lead Uppercut',
    6: 'Rear Uppercut'
}


def run_visual_inference():
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Booting inference engine on: {device.type.upper()}")

    # --- Load Models ---
    print("[*] Loading YOLO26 Spatial Extractor...")
    yolo = YOLO(YOLO_MODEL)

    print("[*] Loading Trained EdgeBoxingTCN...")
    tcn = EdgeBoxingTCN(input_channels=34, num_classes=7, hidden_dim=64).to(device)
    try:
        tcn.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        tcn.eval()  # Set to evaluation mode (disables dropout, fixes batchnorm)
    except Exception as e:
        print(f"[!] Failed to load TCN weights. Did you run train_tcn.py? Error: {e}")
        return

    # --- Initialize Video & Sliding Window ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[!] Could not open video file: {VIDEO_PATH}")
        return

    # Dynamically get the exact FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # Secure fallback if metadata reading fails

    # Calculate adjusted delay per frame in milliseconds
    frame_delay = max(1, int((1000 / fps) / PLAYBACK_SPEED))

    # Fast-forward logic
    if START_TIME_SECONDS > 0:
        start_frame_idx = int(START_TIME_SECONDS * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        print(f"[*] Fast-forwarding {START_TIME_SECONDS} seconds (Starting at frame {start_frame_idx})...")

    # The deque acts as our O(1) rolling temporal window
    temporal_window = deque(maxlen=T_MAX)

    # State tracking for smooth UI
    current_action = "Idle/Guard"
    current_confidence = 0.0

    print("[*] Starting Video Playback. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] End of video stream.")
            break

        # 1. Spatial Extraction (YOLO)
        results = yolo.predict(frame, verbose=False, device=device)[0]

        # Draw the YOLO skeleton directly on the frame for visualization
        annotated_frame = results.plot()

        # 2. Extract & Normalize Keypoints
        if hasattr(results, 'keypoints') and results.keypoints is not None and len(results.keypoints.xyn) > 0:
            kpts_xyn = results.keypoints.xyn[0].cpu().numpy()
            kpts_conf = results.keypoints.conf[0].cpu().numpy()

            left_hip_conf, right_hip_conf = kpts_conf[11], kpts_conf[12]

            # Mid-hip translation
            if left_hip_conf >= 0.5 and right_hip_conf >= 0.5:
                mid_hip_x = (kpts_xyn[11, 0] + kpts_xyn[12, 0]) / 2.0
                mid_hip_y = (kpts_xyn[11, 1] + kpts_xyn[12, 1]) / 2.0
            elif left_hip_conf >= 0.5:
                mid_hip_x, mid_hip_y = kpts_xyn[11, 0], kpts_xyn[11, 1]
            elif right_hip_conf >= 0.5:
                mid_hip_x, mid_hip_y = kpts_xyn[12, 0], kpts_xyn[12, 1]
            else:
                mid_hip_x, mid_hip_y = 0.5, 0.5

            translated_kpts = np.zeros((17, 2), dtype=np.float32)

            for i in range(17):
                if kpts_conf[i] >= 0.5:
                    translated_kpts[i, 0] = kpts_xyn[i, 0] - mid_hip_x
                    translated_kpts[i, 1] = kpts_xyn[i, 1] - mid_hip_y
                else:
                    translated_kpts[i, 0] = 0.0
                    translated_kpts[i, 1] = 0.0

            temporal_window.append(translated_kpts.flatten())
        else:
            # If YOLO loses the person, append a flattened zero-vector
            temporal_window.append(np.zeros(34, dtype=np.float32))

        # 3. Temporal Inference (TCN)
        # Only run inference if our window is fully populated with 30 frames
        if len(temporal_window) == T_MAX:
            # Format shape to (1, 34, 30) -> (Batch, Channels, Time)
            sequence_array = np.array(list(temporal_window), dtype=np.float32)
            sequence_tensor = torch.tensor(sequence_array).transpose(0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = tcn(sequence_tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            pred_class_idx = np.argmax(probs)
            pred_confidence = probs[pred_class_idx]

            # Update UI if confidence is high enough
            if pred_confidence > CONFIDENCE_THRESHOLD:
                current_action = CLASS_MAP[pred_class_idx]
                current_confidence = pred_confidence
            elif current_action != "Idle/Guard" and pred_confidence < 0.40:
                # Fall back to idle if confidence in the punch drops significantly
                current_action = "Idle/Guard"
                current_confidence = probs[0]

        # 4. Render the UI Overlay
        # Determine color: Green for punches, Gray for Idle
        text_color = (0, 255, 0) if current_action != "Idle/Guard" else (200, 200, 200)

        cv2.putText(annotated_frame, f"Action: {current_action}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
        cv2.putText(annotated_frame, f"Conf: {current_confidence:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # Show the video
        cv2.imshow("Real-Time Boxing TCN Inference", annotated_frame)

        # Break loop on 'q' key using the dynamic frame delay
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_visual_inference()