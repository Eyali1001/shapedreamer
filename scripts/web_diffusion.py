#!/usr/bin/env python3
"""
Web-based Live Diffusion Viewer

Streams continuously evolving diffusion images to a web browser.
Uses MJPEG streaming for smooth, low-latency video.

Usage:
    uv run scripts/web_diffusion.py --checkpoint checkpoints/best.pt

Then open http://localhost:8080 in your browser.
"""

import argparse
import io
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from flask import Flask, Response, render_template_string

from src.model.unet import UNet
from src.model.diffusion import GaussianDiffusion


# HTML template for the viewer
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Diffusion</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background: #0a0a0a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            font-size: 2em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        .video-container {
            border: 2px solid #333;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
        img {
            display: block;
            image-rendering: pixelated;
            image-rendering: crisp-edges;
        }
        .controls {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }
        label {
            color: #888;
            font-size: 0.9em;
        }
        input[type="range"] {
            width: 150px;
            accent-color: #667eea;
        }
        .value {
            color: #667eea;
            font-weight: bold;
        }
        .stats {
            color: #444;
            font-size: 0.8em;
            margin-top: 20px;
        }
        button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            transition: transform 0.2s;
        }
        button:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <h1>Live Diffusion</h1>
    <p class="subtitle">Continuously evolving geometric shapes</p>

    <div class="container">
        <div class="video-container">
            <img id="stream" src="/stream" width="512" height="512" />
        </div>

        <div class="controls">
            <div class="control-group">
                <label>Creativity</label>
                <input type="range" id="creativity" min="50" max="800" value="300"
                       onchange="updateCreativity(this.value)">
                <span class="value" id="creativity-value">300</span>
            </div>

            <div class="control-group">
                <label>Scale</label>
                <select id="scale" onchange="updateScale(this.value)">
                    <option value="256">256px</option>
                    <option value="512" selected>512px</option>
                    <option value="768">768px</option>
                </select>
            </div>
        </div>

        <p class="stats">
            Model: DDPM U-Net | Resolution: 32x32 | Device: {{ device }}
        </p>
    </div>

    <script>
        function updateCreativity(value) {
            document.getElementById('creativity-value').textContent = value;
            fetch('/set_creativity/' + value);
        }

        function updateScale(value) {
            const img = document.getElementById('stream');
            img.width = value;
            img.height = value;
        }

        // Reconnect stream if it disconnects
        document.getElementById('stream').onerror = function() {
            setTimeout(() => {
                this.src = '/stream?' + new Date().getTime();
            }, 1000);
        };
    </script>
</body>
</html>
"""


class DiffusionStreamer:
    """Handles the diffusion process and frame generation."""

    def __init__(self, model, diffusion, device, creativity=300):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.creativity = creativity
        self.current_image = None
        self.lock = threading.Lock()
        self.running = True

        # Initialize with first image
        self._initialize()

        # Start background evolution thread
        self.thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.thread.start()

    def _initialize(self):
        """Generate initial image."""
        with torch.no_grad():
            self.current_image = self.diffusion.sample(
                self.model,
                shape=(1, 1, 32, 32),
            )

    @torch.no_grad()
    def _evolve_step(self):
        """One evolution step."""
        if self.current_image is None:
            self._initialize()
            return

        # Add noise - jump back to timestep 'creativity'
        t = torch.tensor([self.creativity], device=self.device)
        noisy, _ = self.diffusion.q_sample(self.current_image, t)

        # Denoise ALL steps from creativity back to 0
        # This is critical - skipping steps produces noise
        x = noisy
        for timestep in range(self.creativity - 1, -1, -1):
            t_batch = torch.full((1,), timestep, device=self.device, dtype=torch.long)
            x = self.diffusion.p_sample(self.model, x, t_batch)

        with self.lock:
            self.current_image = x

    def _evolution_loop(self):
        """Background thread that continuously evolves the image."""
        while self.running:
            self._evolve_step()
            time.sleep(0.01)  # Small delay to prevent CPU overload

    def get_frame_jpeg(self) -> bytes:
        """Get current frame as JPEG bytes."""
        with self.lock:
            if self.current_image is None:
                # Return black frame
                img = Image.new('L', (32, 32), 0)
            else:
                # Convert tensor to PIL Image
                arr = self.current_image[0, 0].cpu().numpy()
                arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr, mode='L')

        # Scale up for better viewing (32 -> 512)
        img = img.resize((512, 512), Image.Resampling.NEAREST)

        # Convert to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        return buffer.getvalue()

    def set_creativity(self, value: int):
        """Update creativity parameter."""
        self.creativity = max(50, min(800, value))

    def stop(self):
        """Stop the evolution thread."""
        self.running = False


def create_app(checkpoint_path: str, device: str, creativity: int):
    """Create Flask app with diffusion streamer."""

    app = Flask(__name__)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")

    # Create diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    # Create streamer
    streamer = DiffusionStreamer(model, diffusion, device, creativity)

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE, device=device)

    @app.route('/stream')
    def stream():
        def generate():
            while True:
                frame = streamer.get_frame_jpeg()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)  # ~10 FPS

        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.route('/set_creativity/<int:value>')
    def set_creativity(value):
        streamer.set_creativity(value)
        return f'Creativity set to {streamer.creativity}'

    return app, streamer


def main():
    parser = argparse.ArgumentParser(description="Web-based live diffusion viewer")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detects MPS/CUDA/CPU)"
    )
    parser.add_argument(
        "--creativity", type=int, default=100,
        help="Initial creativity level (50-800). Lower = faster but subtler changes"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port to run server on"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to"
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    app, streamer = create_app(args.checkpoint, device, args.creativity)

    print(f"\n{'='*50}")
    print(f"  Live Diffusion Server")
    print(f"  Open http://{args.host}:{args.port} in your browser")
    print(f"{'='*50}\n")

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
