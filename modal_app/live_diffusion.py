"""
Modal-hosted Live Diffusion Web Server

Runs the diffusion model on Modal's A100 GPU and streams the evolving
images to your browser.

Usage:
    modal serve modal_app/live_diffusion.py

Then open the URL printed in the terminal.
"""

import modal

app = modal.App("live-diffusion")

# Use the checkpoint volume from training
volume = modal.Volume.from_name("diffusion-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "fastapi[standard]",
    )
)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Diffusion (Modal GPU)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a0a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            font-size: 2em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4aa 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #666; margin-bottom: 30px; }
        .video-container {
            border: 2px solid #333;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0, 212, 170, 0.2);
        }
        img {
            display: block;
            image-rendering: pixelated;
            image-rendering: crisp-edges;
        }
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        label { color: #888; }
        input[type="range"] { width: 150px; accent-color: #00d4aa; }
        .value { color: #00d4aa; font-weight: bold; margin-left: 10px; }
        .stats { color: #444; font-size: 0.8em; margin-top: 20px; }
        .gpu-badge {
            background: linear-gradient(90deg, #00d4aa, #7c3aed);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <h1>Live Diffusion <span class="gpu-badge">A100 GPU</span></h1>
    <p class="subtitle">Continuously evolving geometric shapes - powered by Modal</p>

    <div class="video-container">
        <img id="stream" width="512" height="512" />
    </div>

    <div class="controls">
        <button id="randomize" onclick="randomize()" style="
            background: linear-gradient(90deg, #7c3aed, #00d4aa);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            margin-right: 20px;
        ">Randomize</button>

        <label>Creativity:</label>
        <input type="range" id="creativity" min="50" max="300" value="150"
               onchange="updateCreativity(this.value)">
        <span class="value" id="creativity-value">150</span>

        <label style="margin-left: 30px;">Scale:</label>
        <select id="scale" onchange="updateScale(this.value)">
            <option value="256">256px</option>
            <option value="512" selected>512px</option>
            <option value="768">768px</option>
        </select>
    </div>

    <p class="stats">Model: DDPM U-Net | Resolution: 32x32 | Running on Modal A100</p>

    <script>
        // Construct URLs from current page URL
        const baseUrl = window.location.href.replace('index', '').split('?')[0];
        const streamUrl = baseUrl.replace(/\/$/, '') + 'stream'.replace(/^/, '-').replace('-', '');
        document.getElementById('stream').src = window.location.href.replace('index', 'stream').split('?')[0];

        function updateCreativity(value) {
            document.getElementById('creativity-value').textContent = value;
            fetch(window.location.href.replace('index', 'set-creativity').split('?')[0] + '?value=' + value);
        }
        function updateScale(value) {
            const img = document.getElementById('stream');
            img.width = value;
            img.height = value;
        }
        function randomize() {
            fetch(window.location.href.replace('index', 'randomize').split('?')[0])
                .then(() => {
                    // Refresh stream
                    const img = document.getElementById('stream');
                    img.src = window.location.href.replace('index', 'stream').split('?')[0] + '?' + Date.now();
                });
        }
        document.getElementById('stream').onerror = function() {
            setTimeout(() => { this.src = window.location.href.replace('index', 'stream').split('?')[0] + '?' + Date.now(); }, 1000);
        };
    </script>
</body>
</html>
"""


@app.cls(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    scaledown_window=300,
)
class LiveDiffusionServer:
    def __init__(self):
        self.creativity = 150  # Higher = more change per frame
        self.model = None
        self.diffusion = None
        self.current_image = None

    @modal.enter()
    def setup(self):
        """Load model on container start."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import math

        print("Loading model...")

        # Define model inline (EXACT same architecture as training)
        class SinusoidalPositionEmbeddings(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
            def forward(self, time):
                half_dim = self.dim // 2
                emb = math.log(10000) / (half_dim - 1)
                emb = torch.exp(torch.arange(half_dim, device=time.device, dtype=torch.float16) * -emb)
                emb = time[:, None].half() * emb[None, :]
                return torch.cat((emb.sin(), emb.cos()), dim=-1)

        class TimeEmbedding(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.sinusoidal = SinusoidalPositionEmbeddings(dim)
                self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
            def forward(self, time):
                return self.mlp(self.sinusoidal(time))

        class ResBlock(nn.Module):
            def __init__(self, in_ch, out_ch, time_emb_dim, num_groups=8):
                super().__init__()
                self.norm1 = nn.GroupNorm(num_groups, in_ch)
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
                self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
                self.norm2 = nn.GroupNorm(num_groups, out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
                # MUST be named residual_conv to match checkpoint
                if in_ch != out_ch:
                    self.residual_conv = nn.Conv2d(in_ch, out_ch, 1)
                else:
                    self.residual_conv = nn.Identity()
            def forward(self, x, t):
                h = self.conv1(F.silu(self.norm1(x)))
                h = h + self.time_mlp(t)[:, :, None, None]
                h = self.conv2(F.silu(self.norm2(h)))
                return h + self.residual_conv(x)

        class AttentionBlock(nn.Module):
            def __init__(self, ch, num_heads=4, num_groups=8):
                super().__init__()
                self.num_heads, self.head_dim = num_heads, ch // num_heads
                self.norm = nn.GroupNorm(num_groups, ch)
                self.qkv = nn.Conv2d(ch, ch*3, 1)
                # MUST be named proj_out to match checkpoint
                self.proj_out = nn.Conv2d(ch, ch, 1)
                self.scale = self.head_dim ** -0.5
            def forward(self, x):
                b, c, h, w = x.shape
                qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, self.head_dim, h*w)
                q, k, v = [qkv[:, i].permute(0,1,3,2) for i in range(3)]
                attn = F.softmax(torch.matmul(q, k.transpose(-2,-1)) * self.scale, dim=-1)
                out = torch.matmul(attn, v).permute(0,1,3,2).reshape(b, c, h, w)
                return self.proj_out(out) + x

        class Downsample(nn.Module):
            def __init__(self, in_ch, out_ch=None):
                super().__init__()
                self.conv = nn.Conv2d(in_ch, out_ch or in_ch, 3, stride=2, padding=1)
            def forward(self, x): return self.conv(x)

        class Upsample(nn.Module):
            def __init__(self, in_ch, out_ch=None):
                super().__init__()
                self.conv = nn.Conv2d(in_ch, out_ch or in_ch, 3, padding=1)
            def forward(self, x): return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

        class UNet(nn.Module):
            def __init__(self, base_channels=64, channel_mults=(1,2,4,8), time_emb_dim=128):
                super().__init__()
                self.channels = [base_channels * m for m in channel_mults]
                num_levels = len(channel_mults)

                # MUST use exact same names as training code
                self.time_embedding = TimeEmbedding(time_emb_dim)
                self.init_conv = nn.Conv2d(1, base_channels, 3, padding=1)

                # Encoder
                self.encoder_resblocks = nn.ModuleList()
                self.encoder_attns = nn.ModuleList()
                self.encoder_downsamples = nn.ModuleList()
                for i in range(num_levels - 1):
                    in_ch = self.channels[i] if i > 0 else base_channels
                    out_ch = self.channels[i]
                    self.encoder_resblocks.append(ResBlock(in_ch, out_ch, time_emb_dim))
                    self.encoder_attns.append(AttentionBlock(out_ch))
                    self.encoder_downsamples.append(Downsample(out_ch, self.channels[i+1]))

                # Bottleneck
                bc = self.channels[-1]
                self.bottleneck_res1 = ResBlock(bc, bc, time_emb_dim)
                self.bottleneck_attn = AttentionBlock(bc)
                self.bottleneck_res2 = ResBlock(bc, bc, time_emb_dim)

                # Decoder
                self.decoder_upsamples = nn.ModuleList()
                self.decoder_resblocks = nn.ModuleList()
                self.decoder_attns = nn.ModuleList()
                for i in range(num_levels - 2, -1, -1):
                    in_ch = self.channels[i + 1]
                    out_ch = self.channels[i]
                    self.decoder_upsamples.append(Upsample(in_ch, out_ch))
                    self.decoder_resblocks.append(ResBlock(out_ch * 2, out_ch, time_emb_dim))
                    self.decoder_attns.append(AttentionBlock(out_ch))

                # Output
                self.final_norm = nn.GroupNorm(8, base_channels)
                self.final_conv = nn.Conv2d(base_channels, 1, 3, padding=1)

            def forward(self, x, t):
                time_emb = self.time_embedding(t)
                h = self.init_conv(x)
                skips = []
                for resblock, attn, downsample in zip(self.encoder_resblocks, self.encoder_attns, self.encoder_downsamples):
                    h = resblock(h, time_emb)
                    h = attn(h)
                    skips.append(h)
                    h = downsample(h)
                h = self.bottleneck_res1(h, time_emb)
                h = self.bottleneck_attn(h)
                h = self.bottleneck_res2(h, time_emb)
                for upsample, resblock, attn in zip(self.decoder_upsamples, self.decoder_resblocks, self.decoder_attns):
                    h = upsample(h)
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = resblock(h, time_emb)
                    h = attn(h)
                h = self.final_norm(h)
                h = F.silu(h)
                return self.final_conv(h)

        # Load checkpoint
        self.device = "cuda"
        ckpt = torch.load("/checkpoints/best.pt", map_location=self.device)
        self.model = UNet().to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f"Loaded model from epoch {ckpt.get('epoch', '?')}")

        # Optimization: fp16 for ~2x speedup
        print("Converting to fp16...")
        self.model = self.model.half()
        print("Ready!")

        # Setup diffusion (keep everything on CPU first, then move to device)
        betas = torch.linspace(1e-4, 0.02, 1000)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Precompute and move to device (fp16 for speed)
        self.alphas_cumprod = alphas_cumprod.to(self.device).half()
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device).half()
        self.sqrt_one_minus = torch.sqrt(1 - alphas_cumprod).to(self.device).half()
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(self.device).half()
        self.betas = betas.to(self.device).half()
        post_var = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.post_var = post_var.to(self.device).half()
        self.post_log_var = torch.log(torch.clamp(post_var, min=1e-20)).to(self.device).half()

        # Start from pure noise - will evolve into shapes
        self.current_image = torch.randn(1, 1, 32, 32, device=self.device, dtype=torch.float16)
        print("Ready to serve!")

    def _ddim_sample(self, x, t, t_prev):
        """DDIM sampling step - deterministic and can skip steps."""
        import torch
        tb = torch.full((1,), t, device=self.device, dtype=torch.long)
        noise_pred = self.model(x, tb)

        # Predict x_0 from x_t and predicted noise
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device, dtype=torch.float16)

        # x_0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
        x0_pred = (x - self.sqrt_one_minus[t] * noise_pred) / self.sqrt_alphas_cumprod[t]
        x0_pred = torch.clamp(x0_pred, -1, 1)  # Clip for stability

        # x_{t-1} = sqrt(alpha_{t-1}) * x_0 + sqrt(1-alpha_{t-1}) * noise_pred
        if t_prev >= 0:
            x_prev = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
        else:
            x_prev = x0_pred
        return x_prev

    def evolve(self):
        import torch
        import time
        with torch.no_grad():
            # Add noise at timestep t
            t = self.creativity
            noise = torch.randn_like(self.current_image)
            noisy = self.sqrt_alphas_cumprod[t] * self.current_image + self.sqrt_one_minus[t] * noise

            # DDIM: use only 20 steps instead of 140+
            num_steps = 20
            step_size = t // num_steps
            timesteps = list(range(t, 10, -step_size))  # Stop at 10 to keep variation

            x = noisy
            for i, ts in enumerate(timesteps):
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 10
                x = self._ddim_sample(x, ts, t_prev)
            self.current_image = x
        time.sleep(0.02)  # Smaller delay since generation is faster

    def get_jpeg(self):
        import io
        import numpy as np
        from PIL import Image
        arr = self.current_image[0, 0].float().cpu().numpy()  # fp16 -> fp32 for numpy
        arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, 'L').resize((512, 512), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=90)
        return buf.getvalue()

    @modal.fastapi_endpoint(method="GET")
    def index(self):
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=HTML_TEMPLATE)

    @modal.fastapi_endpoint(method="GET")
    def stream(self):
        from fastapi.responses import StreamingResponse
        def gen():
            while True:
                self.evolve()
                frame = self.get_jpeg()
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

    @modal.fastapi_endpoint(method="GET")
    def set_creativity(self, value: int = 75):
        self.creativity = max(20, min(300, value))
        return {"creativity": self.creativity}

    @modal.fastapi_endpoint(method="GET")
    def randomize(self):
        import torch
        with torch.no_grad():
            self.current_image = torch.randn(1, 1, 32, 32, device=self.device, dtype=torch.float16)
        return {"status": "randomized"}
