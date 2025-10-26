import os
import io
import base64
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import librosa
import librosa.display
import json

# --- Default Parameters for Initial Generation ---
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_ALPHA = 0.55
DEFAULT_DYN_RANGE_DB = 80
MAX_FILE_MB = 60 # Set a reasonable limit for embedding
DEFAULT_N_MELS = 128

# --- Matplotlib Plotting Function (from original app.py) ---
def make_overlay_png_bytes(left: np.ndarray,
                           right: np.ndarray,
                           sr: int,
                           n_fft: int,
                           hop_length: int,
                           alpha: float,
                           dyn_range_db: float,
                           n_mels: int) -> bytes:
    """
    Computes and renders the stereo Mel spectrogram overlay, returning PNG bytes.
    """
    # Calculate Mel spectrograms
    Sl = librosa.feature.melspectrogram(y=left, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    Sr = librosa.feature.melspectrogram(y=right, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Use a global max for a consistent dB reference scale
    global_ref = max(np.max(Sl), np.max(Sr))
    if global_ref == 0: global_ref = 1 # Avoid division by zero for silence
    Sl_db = librosa.power_to_db(Sl, ref=global_ref)
    Sr_db = librosa.power_to_db(Sr, ref=global_ref)

    # Plotting
    fig = plt.figure(figsize=(12, 6), dpi=150)
    ax = plt.gca()

    vmax = 0
    vmin = -float(dyn_range_db)

    # Overlay spectrograms: Left=Red, Right=Blue
    librosa.display.specshow(
        Sl_db, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length,
        cmap="Reds", vmin=vmin, vmax=vmax, alpha=alpha, ax=ax,
    )
    librosa.display.specshow(
        Sr_db, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length,
        cmap="Blues", vmin=vmin, vmax=vmax, alpha=alpha, ax=ax,
    )

    ax.set_title(f"Stereo Mel Spectrogram (L:Red, R:Blue) | sr={sr}, n_fft={n_fft}, hop={hop_length}, n_mels={n_mels}")
    ax.set_ylabel("Frequency (Mel)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# --- HTML Template ---
# This contains the full application logic in HTML and JavaScript.
HTML_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>双声道梅尔频谱叠加可视化 - {filename}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{ --bg:#0b0d12; --card:#11161d; --muted:#a3b1c6; --text:#e8eef8; --accent:#4da3ff; --danger:#ff5d7a; --border:#263142; }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.4 system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }}
    header {{ padding:18px 22px; border-bottom:1px solid var(--border); background:linear-gradient(180deg, #0d1117, #0b0d12); position:sticky; top:0; z-index:10; }}
    header h1 {{ margin:0; font-size:18px; letter-spacing:.5px; }}
    .wrap {{ max-width:1100px; margin:24px auto; padding:0 16px; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px; }}
    .row {{ display:flex; flex-wrap:wrap; gap:12px; align-items:flex-end; }}
    .field {{ display:flex; flex-direction:column; gap:6px; min-width:160px; }}
    label {{ color:var(--muted); font-size:13px; }}
    select, input[type="text"] {{ background:#0c1219; color:var(--text); border:1px solid var(--border); border-radius:10px; padding:10px 12px; outline:none; }}
    input[type="range"] {{ width:220px; }}
    .btn {{ background:var(--accent); color:#09111a; border:none; padding:10px 16px; border-radius:12px; cursor:pointer; font-weight:600; transition:transform .04s ease, filter .15s ease; }}
    .btn:active {{ transform:translateY(1px); }}
    .sp {{ height:1px; background:var(--border); margin:16px 0; }}
    .preview {{ width:100%; background:#0a0f16; border:1px dashed #29405c; border-radius:16px; display:flex; align-items:center; justify-content:center; min-height:220px; overflow:auto; }}
    .preview img, .preview canvas {{ max-width:100%; height:auto; display:block; image-rendering: pixelated; }}
    .meta {{ font-size:13px; color:var(--muted); }}
    .footer {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    .pill {{ font-size:12px; padding:5px 8px; border:1px solid var(--border); border-radius:999px; background:#0c1219; color:#c7d7ee; }}
    .audio {{ margin-top:10px; }}
  </style>
</head>
<body>
  <header><h1>双声道梅尔频谱叠加可视化（L=红 / R=蓝）</h1></header>

  <div class="wrap">
    <div class="card">
      <div class="field" style="width:100%; margin-bottom:8px;">
        <label>当前文件</label>
        <span style="font-weight:600;">{filename}</span>
      </div>
      <div class="row">
        <div class="field">
          <label>FFT 大小 (n_fft)</label>
          <select id="nfft">
            <option>1024</option><option {nfft_2048_selected}>2048</option><option>4096</option><option>8192</option>
          </select>
        </div>
        <div class="field">
          <label>Hop 长度 (样本)</label>
          <select id="hop">
            <option>256</option><option {hop_512_selected}>512</option><option>1024</option><option>2048</option>
          </select>
        </div>
        <div class="field">
          <label>叠加透明度 (α)</label>
          <input id="alpha" type="range" min="0.2" max="0.9" step="0.01" value="{alpha}"/>
        </div>
        <div class="field">
          <label>动态范围 (dB)</label>
          <input id="dyn" type="range" min="40" max="120" step="1" value="{dyn_range}"/>
        </div>
        <div class="field" style="min-width:140px">
          <label>&nbsp;</label>
          <button id="run" class="btn">重新生成 (浏览器端)</button>
        </div>
      </div>

      <div class="sp"></div>
      <div class="preview" id="preview"><img src="{initial_image_url}" alt="Initial Mel Spectrogram"></div>
      <div class="footer" style="margin-top:12px">
        <span id="status" class="pill">状态：已加载</span>
        <span id="meta" class="meta"></span>
      </div>
      <audio id="player" class="audio" controls style="width:100%;" src="{audio_data_url}"></audio>
    </div>
  </div>

<script>const MEL_FILTERBANK = {mel_filterbank_json};</script>
<script>
// --- Embedded Audio Data ---
const AUDIO_DATA_URL = "{audio_data_url}";
let audioBuffer = null;
let audioContext = null;

// --- DOM Elements ---
const nfftEl = document.getElementById("nfft");
const hopEl = document.getElementById("hop");
const alphaEl = document.getElementById("alpha");
const dynEl = document.getElementById("dyn");
const runBtn = document.getElementById("run");
const previewEl = document.getElementById("preview");
const metaEl = document.getElementById("meta");
const statusEl = document.getElementById("status");
const player = document.getElementById("player");
let indicatorCanvas = null;
let animationFrameId = null;


// --- Playback Indicator ---
function drawIndicator() {{
    if (!indicatorCanvas || !player.duration) return;
    const indicatorCtx = indicatorCanvas.getContext('2d');
    const canvasWidth = indicatorCanvas.width;
    const canvasHeight = indicatorCanvas.height;
    
    indicatorCtx.clearRect(0, 0, canvasWidth, canvasHeight);
    const x = (player.currentTime / player.duration) * canvasWidth;

    // Draw a more visible line
    indicatorCtx.fillStyle = 'rgba(255, 0, 0, 0.8)'; // Red, more opaque
    indicatorCtx.fillRect(x - 1, 0, 3, canvasHeight); // 3px wide, centered

    animationFrameId = requestAnimationFrame(drawIndicator);
}}

player.addEventListener('play', () => {{
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    animationFrameId = requestAnimationFrame(drawIndicator);
}});
player.addEventListener('pause', () => {{
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
}});
player.addEventListener('ended', () => {{
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
    if (indicatorCanvas) {{
        const indicatorCtx = indicatorCanvas.getContext('2d');
        indicatorCtx.clearRect(0, 0, indicatorCanvas.width, indicatorCanvas.height);
    }}
}});

// --- Web Audio API Analysis ---
async function decodeAudio() {{
  if (audioBuffer) return true;
  try {{
    statusEl.textContent = "状态：解码音频中...";
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const response = await fetch(AUDIO_DATA_URL);
    const arrayBuffer = await response.arrayBuffer();
    audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    if (audioBuffer.numberOfChannels < 2) {{
        alert("错误：嵌入的音频为单声道，无法进行双声道分析。");
        return false;
    }}
    return true;
  }} catch (e) {{
    console.error("音频解码失败:", e);
    alert("音频解码失败，您的浏览器可能不支持该格式。");
    return false;
  }}
}}

function hannWindow(length) {{
    const win = new Float32Array(length);
    for (let i = 0; i < length; i++) {{
        win[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (length - 1)));
    }}
    return win;
}}

async function getSpectrogramData(channelData, n_fft, hop_length) {{
    const totalSamples = channelData.length;
    const frames = [];
    const window = hannWindow(n_fft);
    const n_mels = MEL_FILTERBANK.length;
    const fft_bins = MEL_FILTERBANK[0].length;

    const offlineCtx = new OfflineAudioContext(1, n_fft, audioBuffer.sampleRate);
    const analyser = offlineCtx.createAnalyser();
    analyser.fftSize = n_fft;
    analyser.smoothingTimeConstant = 0;

    const freqDataDb = new Float32Array(analyser.frequencyBinCount);

    for (let i = 0; i + n_fft <= totalSamples; i += hop_length) {{
        const buffer = offlineCtx.createBuffer(1, n_fft, audioBuffer.sampleRate);
        const frameData = buffer.getChannelData(0);

        for (let j = 0; j < n_fft; j++) {{
            frameData[j] = channelData[i + j] * window[j];
        }}
        
        const source = offlineCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(analyser);
        analyser.connect(offlineCtx.destination);
        source.start(0);

        await offlineCtx.startRendering();
        analyser.getFloatFrequencyData(freqDataDb);

        const powerSpec = new Float32Array(fft_bins);
        for (let k = 0; k < fft_bins; k++) {{
            powerSpec[k] = Math.pow(10, freqDataDb[k] / 10);
        }}

        const melSpec = new Float32Array(n_mels).fill(0);
        for (let m = 0; m < n_mels; m++) {{
            for (let k = 0; k < fft_bins; k++) {{
                melSpec[m] += MEL_FILTERBANK[m][k] * powerSpec[k];
            }}
        }}

        const melSpecDb = new Float32Array(n_mels);
        for (let m = 0; m < n_mels; m++) {{
            melSpecDb[m] = 10 * Math.log10(melSpec[m] + 1e-6);
        }}
        frames.push(melSpecDb);
    }}
    return frames;
}}

function drawSpectrogram(ctx, specData, sr, n_fft, alpha, dyn_range, colormap) {{
    const width = specData.length;
    if (width === 0) return;
    const height = specData[0].length;
    
    ctx.globalAlpha = alpha;

    const vmin = -dyn_range;
    const vmax = 0; // dBFS

    for (let x = 0; x < width; x++) {{
        for (let y = 0; y < height; y++) {{
            const db = specData[x][y];
            if (db < vmin) continue;

            const normalized = (db - vmin) / (vmax - vmin);
            const colorVal = Math.floor(255 * normalized);

            if (colormap === 'Reds') {{
                ctx.fillStyle = `rgb(${{colorVal}}, 0, 0)`;
            }} else {{ // Blues
                ctx.fillStyle = `rgb(0, 0, ${{colorVal}})`;
            }}
            // Draw bottom-up
            ctx.fillRect(x, height - 1 - y, 1, 1);
        }}
    }}
}}

async function runAnalysis() {{
  const ready = await decodeAudio();
  if (!ready) {{
    runBtn.disabled = false;
    statusEl.textContent = "状态：错误";
    return;
  }}

  runBtn.disabled = true;
  statusEl.textContent = "状态：分析中 (L)...";

  const n_fft = parseInt(nfftEl.value, 10);
  const hop_length = parseInt(hopEl.value, 10);
  const alpha = parseFloat(alphaEl.value);
  const dyn_range = parseFloat(dynEl.value);

  const leftChannel = audioBuffer.getChannelData(0);
  const rightChannel = audioBuffer.getChannelData(1);

  const specL = await getSpectrogramData(leftChannel, n_fft, hop_length);
  statusEl.textContent = "状态：分析中 (R)...";
  const specR = await getSpectrogramData(rightChannel, n_fft, hop_length);

  statusEl.textContent = "状态：渲染中...";

  // Prepare spectrogram canvas
  const canvas = document.createElement('canvas');
  const numFrames = specL.length > 0 ? specL.length : 800;
  const freqBins = MEL_FILTERBANK.length;
  canvas.width = numFrames;
  canvas.height = freqBins;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0a0f16'; // background
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw spectrograms
  drawSpectrogram(ctx, specL, audioBuffer.sampleRate, n_fft, alpha, dyn_range, 'Reds');
  drawSpectrogram(ctx, specR, audioBuffer.sampleRate, n_fft, alpha, dyn_range, 'Blues');

  // Set up container for canvases
  previewEl.innerHTML = '';
  const container = document.createElement('div');
  container.style.position = 'relative';
  container.style.lineHeight = '0';
  previewEl.appendChild(container);
  container.appendChild(canvas);

  // Create and add indicator canvas
  indicatorCanvas = document.createElement('canvas');
  indicatorCanvas.width = canvas.width;
  indicatorCanvas.height = canvas.height;
  indicatorCanvas.style.position = 'absolute';
  indicatorCanvas.style.top = '0';
  indicatorCanvas.style.left = '0';
  indicatorCanvas.style.pointerEvents = 'none';
  container.appendChild(indicatorCanvas);
  
  metaEl.textContent = `sr=${{audioBuffer.sampleRate}} Hz | 时长≈${{audioBuffer.duration.toFixed(2)}}s | n_fft=${{n_fft}}, hop=${{hop_length}}, n_mels=${{MEL_FILTERBANK.length}}, α=${{alpha}}, 动态范围=${{dyn_range}} dB (JS Render)`;
  statusEl.textContent = "状态：完成";
  runBtn.disabled = false;
}}

runBtn.addEventListener("click", runAnalysis);

// Initialize
metaEl.textContent = `sr={sr} Hz | 时长≈{duration}s | n_fft={n_fft}, hop={hop_length}, n_mels={n_mels}, α={alpha}, 动态范围={dyn_range} dB (Initial Render)`;

</script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(
        description="Generate a single-file HTML for stereo Mel spectrogram visualization.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the input audio file (.wav, .mp3).")
    parser.add_argument("-o", "--output", required=True, help="Path for the output HTML file.")
    args = parser.parse_args()

    # --- 1. Validate Input ---
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at '{args.input}'")
        return

    file_size_mb = os.path.getsize(args.input) / (1024 * 1024)
    if file_size_mb > MAX_FILE_MB:
        print(f"Error: File size ({file_size_mb:.1f} MB) exceeds the limit of {MAX_FILE_MB} MB for embedding.")
        return

    print(f"Processing '{args.input}'...")

    # --- 2. Load Audio and Check Channels ---
    try:
        y, sr = librosa.load(args.input, sr=None, mono=False)
        if y.ndim != 2 or y.shape[0] < 2:
            print("Error: The audio file is not stereo. This tool requires a stereo input.")
            return
        left, right = y[0, :], y[1, :]
        duration = round(len(left) / sr, 2)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # --- 3. Generate Initial Spectrogram Image ---
    print("Generating initial spectrogram with default parameters...")
    try:
        png_bytes = make_overlay_png_bytes(
            left, right, sr,
            n_fft=DEFAULT_N_FFT,
            hop_length=DEFAULT_HOP_LENGTH,
            alpha=DEFAULT_ALPHA,
            dyn_range_db=DEFAULT_DYN_RANGE_DB,
            n_mels=DEFAULT_N_MELS
        )
        initial_image_b64 = base64.b64encode(png_bytes).decode('ascii')
        initial_image_url = f"data:image/png;base64,{initial_image_b64}"
    except Exception as e:
        print(f"Error creating spectrogram image: {e}")
        return

    # --- 4. Embed Audio File ---
    print("Embedding audio data...")
    try:
        with open(args.input, 'rb') as f_audio:
            audio_bytes = f_audio.read()
        audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
        
        file_ext = os.path.splitext(args.input.lower())[1]
        mime_type = "audio/mpeg" if file_ext == ".mp3" else "audio/wav"
        audio_data_url = f"data:{mime_type};base64,{audio_b64}"
    except Exception as e:
        print(f"Error embedding audio file: {e}")
        return

    # --- 5. Generate Mel Filterbank for JS ---
    mel_filterbank = librosa.filters.mel(sr=sr, n_fft=DEFAULT_N_FFT, n_mels=DEFAULT_N_MELS)
    mel_filterbank_json = json.dumps(mel_filterbank.tolist())

    # --- 6. Populate and Write HTML File ---
    print(f"Writing to '{args.output}'...")
    final_html = HTML_TEMPLATE.format(
        filename=os.path.basename(args.input),
        initial_image_url=initial_image_url,
        audio_data_url=audio_data_url,
        sr=sr,
        duration=duration,
        n_fft=DEFAULT_N_FFT,
        hop_length=DEFAULT_HOP_LENGTH,
        alpha=DEFAULT_ALPHA,
        dyn_range=DEFAULT_DYN_RANGE_DB,
        n_mels=DEFAULT_N_MELS,
        mel_filterbank_json=mel_filterbank_json,
        nfft_2048_selected='selected' if DEFAULT_N_FFT == 2048 else '',
        hop_512_selected='selected' if DEFAULT_HOP_LENGTH == 512 else '',
    )

    with open(args.output, 'w', encoding='utf-8') as f_html:
        f_html.write(final_html)

    print("Done!")

if __name__ == "__main__":
    main()