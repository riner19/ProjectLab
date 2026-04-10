import os
import glob
import shutil
import subprocess
from fractions import Fraction

VIDEO_DIR = r"RGB_videos"
OUTPUT_DIR = os.path.join(VIDEO_DIR, "resized")
TARGET_SIZE = "1920:1080"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def require_tool(tool_name: str) -> str:
    """Return full path to a tool or raise a clear error if it is missing."""
    tool_path = shutil.which(tool_name)
    if not tool_path:
        raise FileNotFoundError(
            f"'{tool_name}' was not found on PATH. "
            f"Please install it or add it to PATH before running this script."
        )
    return tool_path


def get_video_fps(ffprobe_path: str, video_path: str) -> str:
    """
    Read the native FPS from a video using ffprobe.

    Returns a string like '30' or '30000/1001' that ffmpeg can accept.
    """
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    fps_text = result.stdout.strip()
    if not fps_text:
        raise RuntimeError(f"Could not read FPS from video: {video_path}")

    # Validate that it is a legal ffmpeg FPS value
    try:
        Fraction(fps_text)
    except Exception as exc:
        raise RuntimeError(f"Invalid FPS value '{fps_text}' for {video_path}") from exc

    return fps_text


def resize_video(ffmpeg_path: str, video_path: str, output_path: str, fps: str) -> None:
    """Resize the video while preserving original FPS using GPU acceleration."""
    cmd = [
        ffmpeg_path,
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", video_path,
        "-vf", f"scale_cuda={TARGET_SIZE}",
        "-r", fps,
        "-c:v", "h264_nvenc",
        "-preset", "fast",
        "-crf", "23",
        "-y",
        output_path,
    ]

    subprocess.run(cmd, check=True)


def main() -> None:
    ffmpeg_path = require_tool("ffmpeg")
    ffprobe_path = require_tool("ffprobe")

    videos = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))

    if not videos:
        print(f"No .mp4 videos found in: {VIDEO_DIR}")
        return

    print(f"Found {len(videos)} video(s) in {VIDEO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    for video in videos:
        base = os.path.basename(video)
        output = os.path.join(OUTPUT_DIR, base)

        # Skip already processed files
        if os.path.exists(output):
            print(f"Skipping existing file: {output}")
            continue

        try:
            fps = get_video_fps(ffprobe_path, video)
            print(f"Processing {base} | FPS: {fps}")

            resize_video(ffmpeg_path, video, output, fps)

            print(f"Saved: {output}")
        except subprocess.CalledProcessError as exc:
            print(f"Failed to process {base}: {exc}")
        except Exception as exc:
            print(f"Error with {base}: {exc}")


if __name__ == "__main__":
    main()