import os

youtube_links = [
    "https://www.youtube.com/watch?v=kbgkeTTSau8&t=779s"
]

output_dir = "../datasets/RGB_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, link in enumerate(youtube_links):
    v_name = f"V{i + 1}.mp4"
    output_path = os.path.join(output_dir, v_name)

    print(f"\n--- Downloading {v_name} at Native Highest FPS ---")

    # yt-dlp automatically grabs the best resolution and best framerate available.
    # We save it directly to the final output path, bypassing the need for a temp file.
    download_cmd = f'yt-dlp -f "bestvideo+bestaudio/best" --merge-output-format mp4 -o "{output_path}" {link}'

    os.system(download_cmd)

print("\nAll videos have been downloaded at their maximum available resolution and native frame rate.")