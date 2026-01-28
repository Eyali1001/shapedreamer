#!/usr/bin/env python3
"""
Capture frames from the live diffusion stream and create a GIF.

Usage:
    # Capture from Modal endpoint
    uv run scripts/capture_gif.py --url https://eyali1001--live-diffusion-livediffusionserver-stream-dev.modal.run

    # Capture from local server
    uv run scripts/capture_gif.py --url http://localhost:5000/stream

    # Customize duration and output
    uv run scripts/capture_gif.py --url <stream-url> --duration 10 --output assets/demo.gif
"""

import argparse
import io
import os
import time
from pathlib import Path

import requests
from PIL import Image


def capture_gif(
    stream_url: str,
    output_path: str = "assets/live_diffusion.gif",
    duration: float = 5.0,
    fps: int = 10,
):
    """Capture frames from MJPEG stream and create a GIF."""

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frames = []
    target_frames = int(duration * fps)
    frame_interval = 1.0 / fps

    print(f"Capturing {target_frames} frames over {duration}s...")
    print(f"Stream URL: {stream_url}")

    try:
        response = requests.get(stream_url, stream=True, timeout=30)
        response.raise_for_status()

        buffer = b""
        start_time = time.time()
        last_frame_time = 0

        for chunk in response.iter_content(chunk_size=1024):
            buffer += chunk

            # Look for JPEG markers
            while True:
                # Find start of JPEG
                start = buffer.find(b'\xff\xd8')
                if start == -1:
                    break

                # Find end of JPEG
                end = buffer.find(b'\xff\xd9', start)
                if end == -1:
                    break

                # Extract JPEG
                jpeg_data = buffer[start:end + 2]
                buffer = buffer[end + 2:]

                # Rate limit frame capture
                current_time = time.time()
                if current_time - last_frame_time >= frame_interval:
                    try:
                        img = Image.open(io.BytesIO(jpeg_data))
                        # Resize to reasonable gif size
                        img = img.resize((256, 256), Image.Resampling.NEAREST)
                        frames.append(img.copy())
                        last_frame_time = current_time
                        print(f"\rCaptured {len(frames)}/{target_frames} frames", end="", flush=True)
                    except Exception as e:
                        print(f"\nError processing frame: {e}")

                # Check if we have enough frames
                if len(frames) >= target_frames:
                    raise StopIteration()

    except (StopIteration, KeyboardInterrupt):
        pass
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to stream: {e}")
        return

    if not frames:
        print("\nNo frames captured!")
        return

    print(f"\n\nSaving {len(frames)} frames to {output_path}...")

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),  # ms per frame
        loop=0,  # infinite loop
        optimize=True,
    )

    file_size = os.path.getsize(output_path) / 1024
    print(f"Saved! Size: {file_size:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Capture GIF from live diffusion stream")
    parser.add_argument(
        "--url",
        type=str,
        default="https://eyali1001--live-diffusion-livediffusionserver-stream-dev.modal.run",
        help="Stream URL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/live_diffusion.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration to capture (seconds)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second",
    )

    args = parser.parse_args()
    capture_gif(args.url, args.output, args.duration, args.fps)


if __name__ == "__main__":
    main()
