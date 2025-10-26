# app.py
import os
import io
import base64
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_file
import matplotlib
matplotlib.use("Agg")  # 后端渲染
import matplotlib.pyplot as plt

import librosa
import librosa.display

ALLOWED_EXTS = {".wav", ".mp3"}
MAX_FILE_MB = 60

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

@app.route("/")
def index():
    # 读取 HTML 文件内容，并用 MAX_FILE_MB 替换占位符，使其在前端 JS 中可用
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        rendered_html = html_content.replace("{{MAX_PLACEHOLDER}}", str(MAX_FILE_MB))
        return rendered_html
    except FileNotFoundError:
        return "index.html not found", 404

def _is_allowed(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTS

def _load_audio_to_stereo(tmp_path: str, target_sr: int | None = None):
    """
    使用 librosa 读取音频；保持多声道，不做混缩。
    返回 (sr, left, right) 或抛出异常。
    """
    # mono=False 保持声道；sr=None 不重采样（保留原采样率）
    y, sr = librosa.load(tmp_path, sr=target_sr, mono=False)

    # 兼容不同形状：librosa 在 mono=False 时通常返回 (channels, samples)
    if y.ndim == 1:
        # 单声道
        return sr, None, None

    # 规范化为 (2, n_samples)
    if y.shape[0] == 2:
        left, right = y[0, :], y[1, :]
    elif y.shape[-1] == 2:
        left, right = y[:, 0], y[:, 1]
    else:
        # 多于 2 声道，取前两路
        left = y[0, :] if y.shape[0] >= 2 else y[:, 0]
        right = y[1, :] if y.shape[0] >= 2 else y[:, 1]

    return sr, left.astype(np.float32), right.astype(np.float32)

def _make_overlay_png(left: np.ndarray,
                      right: np.ndarray,
                      sr: int,
                      n_fft: int,
                      hop_length: int,
                      alpha: float,
                      dyn_range_db: float) -> bytes:
    """
    计算左右声道频谱，红/蓝叠加，返回 PNG 二进制。
    """
    # 计算 STFT 幅度谱
    Sl = np.abs(librosa.stft(left, n_fft=n_fft, hop_length=hop_length))
    Sr = np.abs(librosa.stft(right, n_fft=n_fft, hop_length=hop_length))

    # 以全局最大幅度为参考，统一 dB 标尺，避免单边压制
    global_ref = max(np.max(Sl), np.max(Sr))
    Sl_db = librosa.amplitude_to_db(Sl, ref=global_ref)
    Sr_db = librosa.amplitude_to_db(Sr, ref=global_ref)

    # 绘图
    fig = plt.figure(figsize=(12, 6), dpi=150)
    ax = plt.gca()

    # vmin/vmax 控 dB 动态范围（0 ~ -dyn）
    vmax = 0
    vmin = -float(dyn_range_db)

    # 频谱叠加：左=红, 右=蓝
    librosa.display.specshow(
        Sl_db,
        x_axis="time",
        y_axis="linear",
        sr=sr,
        hop_length=hop_length,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        ax=ax,
    )
    librosa.display.specshow(
        Sr_db,
        x_axis="time",
        y_axis="linear",
        sr=sr,
        hop_length=hop_length,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        ax=ax,
    )

    ax.set_title(f"Stereo Spectrogram Overlay  (L: Red, R: Blue)  |  sr={sr} Hz, n_fft={n_fft}, hop={hop_length}")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"ok": False, "message": "未选择文件"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "message": "未选择文件"}), 400

    if not _is_allowed(f.filename):
        return jsonify({"ok": False, "message": "仅支持 .wav / .mp3"}), 400

    try:
        n_fft = int(request.form.get("n_fft", "2048"))
        hop_length = int(request.form.get("hop_length", str(max(1, n_fft // 4))))
        alpha = float(request.form.get("alpha", "0.55"))
        dyn_range_db = float(request.form.get("dyn_range_db", "80"))
    except Exception:
        return jsonify({"ok": False, "message": "参数解析失败"}), 400

    # 将上传内容写入临时文件（保留原扩展名，利于解码器判断格式）
    suffix = os.path.splitext(f.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.stream.seek(0)
        tmp.write(f.read())
        tmp_path = tmp.name

    try:
        sr, left, right = _load_audio_to_stereo(tmp_path, target_sr=None)
        if left is None or right is None:
            return jsonify({"ok": False, "message": "检测到单声道音频：请上传双声道文件"}), 400

        png_bytes = _make_overlay_png(left, right, sr, n_fft, hop_length, alpha, dyn_range_db)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        duration = round(max(len(left), len(right)) / sr, 3)

        return jsonify({
            "ok": True,
            "image_data_url": f"data:image/png;base64,{b64}",
            "meta": {
                "sample_rate": sr,
                "duration_sec": duration,
                "n_fft": n_fft,
                "hop_length": hop_length,
                "alpha": alpha,
                "dyn_range_db": dyn_range_db
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "message": f"解析或绘制失败：{e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    # 生产环境请放到 WSGI（gunicorn 等）后面，这里方便直接跑
    app.run(host="127.0.0.1", port=8008, debug=True)

