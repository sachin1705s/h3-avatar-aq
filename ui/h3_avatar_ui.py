from __future__ import annotations

import html
import json
import os
import re
import shutil
import ssl
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "artifacts" / "ui_jobs"
JOBS.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "audio-v2": "/runs/h3/locked/h3-avatar-audio-v2-step1000.pt",
    "audio-v1": "/runs/h3/ref2va-lora-audio-v1-en-384-56-r16/checkpoint-1500.pt",
    "visual-v2": "/runs/h3/ref2va-lora-visual-v2-1k-384-r16/checkpoint-1500.pt",
}

STATUS: dict[str, dict] = {}
DEFAULT_CHECKPOINT = "audio-v2"
WARM_ENDPOINT_URL = os.environ.get("H3_WARM_ENDPOINT_URL", "").strip()


def valid_frames(seconds: int) -> int:
    # MiniMax H3 VAE accepts 17n + 5 frames. At 24fps, 243 is ~10.1s.
    target = max(22, int(seconds * 24))
    n = max(1, round((target - 5) / 17))
    return 17 * n + 5


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    def send(context: ssl.SSLContext | None = None) -> dict:
        request = Request(url, data=data, headers=headers, method="POST")
        with urlopen(request, timeout=4 * 60 * 60, context=context) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return send()
    except (ssl.SSLCertVerificationError, URLError) as exc:
        is_cert_error = isinstance(exc, ssl.SSLCertVerificationError) or isinstance(
            getattr(exc, "reason", None),
            ssl.SSLCertVerificationError,
        )
        if not is_cert_error:
            raise
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
            return send(context)
        except Exception:
            context = ssl._create_unverified_context()
            result = send(context)
            result["ssl_warning"] = "Used local-dev unverified HTTPS fallback"
            return result


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
    if not match:
        raise ValueError("missing multipart boundary")
    boundary = ("--" + match.group("boundary").strip('"')).encode()
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in body.split(boundary):
        part = part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        header_blob, _, data = part.partition(b"\r\n\r\n")
        if not data:
            continue
        data = data.rstrip(b"\r\n")
        headers = header_blob.decode("utf-8", "replace")
        disp = next((line for line in headers.split("\r\n") if line.lower().startswith("content-disposition:")), "")
        name_match = re.search(r'name="([^"]+)"', disp)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', disp)
        if filename_match:
            filename = Path(filename_match.group(1)).name or "reference.png"
            files[name] = (filename, data)
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def set_status(status_id: str, **updates) -> None:
    current = STATUS.setdefault(status_id, {})
    current.update(updates)
    current.setdefault("job_id", status_id)
    current["updated_at"] = time.time()
    job_dir = JOBS / status_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "status.json").write_text(json.dumps(current, indent=2), encoding="utf-8")


def generate_job(job_id: str, image_path: Path, fields: dict[str, str]) -> None:
    try:
        duration = int(fields.get("duration", "10"))
        frames = valid_frames(duration)
        steps = int(fields.get("steps", "30"))
        checkpoint_key = fields.get("checkpoint", DEFAULT_CHECKPOINT)
        checkpoint = CHECKPOINTS.get(checkpoint_key, CHECKPOINTS[DEFAULT_CHECKPOINT])
        generate_audio = fields.get("audio", "model") == "model"
        prompt = fields.get("prompt", "").strip() or (
            "The same character faces the camera and speaks with an audible natural voice, "
            "expressive synchronized mouth movement, subtle head motion, and a friendly expression."
        )

        job_dir = JOBS / job_id
        ref = job_dir / "reference_512.mp4"
        remote_ref = f"/samples/ui_refs/{job_id}/reference_512.mp4"
        remote_out = f"/generated/ui/{job_id}/generated.mp4"
        container_out = f"/runs{remote_out}"
        local_out = job_dir / "generated.mp4"

        set_status(job_id, state="encoding_reference", detail="Creating 512px reference video", progress=5)
        run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(image_path), "-t", str(duration), "-r", "24",
            "-vf", "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-an", str(ref),
        ])

        set_status(job_id, state="uploading", detail="Uploading reference to Modal", progress=10)
        run(["modal", "volume", "put", "--force", "h3-avatar-runs", str(ref), remote_ref])

        set_status(job_id, state="generating", detail="Running H3 generation on Modal", progress=20)
        if WARM_ENDPOINT_URL:
            payload = {
                "checkpoint": checkpoint,
                "reference_video": f"/runs{remote_ref}",
                "output_path": container_out,
                "prompt": prompt,
                "num_frames": frames,
                "height": 512,
                "width": 512,
                "num_inference_steps": steps,
                "seed": int(fields.get("seed", "1500")),
                "include_reference_audio": False,
                "generate_audio": generate_audio,
            }
            result = post_json(WARM_ENDPOINT_URL, payload)
            (job_dir / "modal.log").write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            cmd = [
                "modal", "run", "modal_h3_avatar.py::generate_ref2va_sample",
                "--checkpoint", checkpoint,
                "--reference-video", f"/runs{remote_ref}",
                "--output-path", container_out,
                "--prompt", prompt,
                "--num-frames", str(frames),
                "--height", "512",
                "--width", "512",
                "--num-inference-steps", str(steps),
                "--seed", str(int(fields.get("seed", "1500"))),
                "--no-include-reference-audio",
            ]
            cmd.append("--generate-audio" if generate_audio else "--no-generate-audio")
            modal_log = run(cmd)
            (job_dir / "modal.log").write_text(modal_log, encoding="utf-8")

        set_status(job_id, state="downloading", detail="Downloading MP4", progress=90)
        run(["modal", "volume", "get", "--force", "h3-avatar-runs", remote_out, str(job_dir) + "/"])
        downloaded = job_dir / "generated.mp4"
        if not downloaded.exists():
            matches = list(job_dir.glob("*generated*.mp4"))
            if matches:
                shutil.move(str(matches[0]), str(downloaded))
        set_status(job_id, state="done", detail="Done", progress=100, video_url=f"/jobs/{job_id}/generated.mp4")
    except Exception as exc:
        set_status(job_id, state="error", detail=str(exc), progress=100)


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>H3 Avatar Studio</title>
  <style>
    :root { color-scheme: dark; --bg:#0f1115; --panel:#171a21; --line:#2b313d; --text:#f4f6fb; --muted:#9aa4b2; --accent:#57c7ff; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; background:var(--bg); color:var(--text); }
    main { max-width:1180px; margin:0 auto; padding:28px; display:grid; grid-template-columns: 400px 1fr; gap:24px; }
    h1 { font-size:24px; margin:0 0 6px; }
    p { color:var(--muted); line-height:1.45; }
    form, .result { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; }
    label { display:block; margin:14px 0 6px; color:#cbd3df; font-size:13px; }
    input, textarea, select, button { width:100%; border-radius:6px; border:1px solid var(--line); background:#0f1218; color:var(--text); padding:10px; font:inherit; }
    textarea { min-height:96px; resize:vertical; }
    button { margin-top:18px; background:var(--accent); color:#061018; border:0; font-weight:700; cursor:pointer; }
    video { width:100%; max-height:72vh; background:#050608; border:1px solid var(--line); border-radius:8px; }
    .row { display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
    .bar { height:10px; background:#0c0e13; border-radius:999px; overflow:hidden; border:1px solid var(--line); }
    .fill { height:100%; width:0%; background:var(--accent); transition:width .3s ease; }
    .pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:4px 8px; color:#cbd3df; font-size:12px; }
    .warn { color:#f4c56a; }
    @media (max-width: 860px) { main { grid-template-columns:1fr; padding:16px; } }
  </style>
</head>
<body>
<main>
  <section>
    <h1>H3 Avatar Studio</h1>
    <p>Upload one reference image and generate a talking avatar clip on Modal.</p>
    <form id="form">
      <label>Reference Image</label>
      <input name="image" type="file" accept="image/*" required>
      <div class="row">
        <div>
          <label>Duration</label>
          <select name="duration"><option value="10">~10 sec</option><option value="5">~5 sec</option></select>
        </div>
        <div>
          <label>Steps</label>
          <select name="steps"><option value="30">30</option><option value="16">16 preview</option></select>
        </div>
      </div>
      <label>Checkpoint</label>
      <select name="checkpoint">
        <option value="audio-v2" selected>Audio-v2 transcript LoRA, locked</option>
        <option value="audio-v1">Audio-v1 LoRA, model audio</option>
        <option value="visual-v2">Visual-v2 LoRA, experimental audio</option>
      </select>
      <label>Audio</label>
      <select name="audio">
        <option value="model">H3 model-generated audio</option>
        <option value="none">No audio</option>
      </select>
      <p class="warn">Locked checkpoint: audio-v2 step 1000. Warm endpoint is used when configured.</p>
      <label>Prompt</label>
      <textarea name="prompt">The same character faces the camera and speaks with an audible natural voice, expressive synchronized mouth movement, subtle head motion, and a friendly expression.</textarea>
      <button type="submit">Generate Video</button>
    </form>
  </section>
  <section class="result">
    <div><span class="pill" id="state">idle</span></div>
    <p id="detail">Upload an image to start.</p>
    <div class="bar"><div class="fill" id="fill"></div></div>
    <p id="job"></p>
    <video id="video" controls playsinline hidden></video>
  </section>
</main>
<script>
const form = document.getElementById('form');
const state = document.getElementById('state');
const detail = document.getElementById('detail');
const fill = document.getElementById('fill');
const jobEl = document.getElementById('job');
const video = document.getElementById('video');
let timer = null;
function render(s) {
  state.textContent = s.state || 'unknown';
  detail.textContent = s.detail || '';
  fill.style.width = `${s.progress || 0}%`;
  if (s.job_id) jobEl.textContent = `job ${s.job_id}`;
  if (s.video_url) {
    video.hidden = false;
    video.src = `${s.video_url}?t=${Date.now()}`;
  }
}
async function poll(id) {
  const r = await fetch(`/api/status?id=${encodeURIComponent(id)}`);
  const s = await r.json();
  render(s);
  if (s.state === 'done' || s.state === 'error') clearInterval(timer);
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  video.hidden = true;
  const data = new FormData(form);
  render({state:'starting', detail:'Submitting job', progress:1});
  const r = await fetch('/api/generate', {method:'POST', body:data});
  const s = await r.json();
  render(s);
  clearInterval(timer);
  timer = setInterval(() => poll(s.job_id), 2500);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/status":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            payload = STATUS.get(job_id)
            if payload is None:
                status_file = JOBS / job_id / "status.json"
                payload = json.loads(status_file.read_text()) if status_file.is_file() else {"state": "missing", "job_id": job_id}
            self.send_json(payload)
            return
        match = re.fullmatch(r"/jobs/([a-f0-9-]+)/generated\.mp4", parsed.path)
        if match:
            path = JOBS / match.group(1) / "generated.mp4"
            if not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        fields, files = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(length))
        if "image" not in files:
            self.send_json({"state": "error", "detail": "missing image"}, 400)
            return
        job_id = str(uuid.uuid4())
        job_dir = JOBS / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        filename, content = files["image"]
        suffix = Path(filename).suffix.lower() or ".png"
        image_path = job_dir / f"input{suffix}"
        image_path.write_bytes(content)
        set_status(job_id, state="queued", detail="Queued", progress=0)
        thread = threading.Thread(target=generate_job, args=(job_id, image_path, fields), daemon=True)
        thread.start()
        self.send_json(STATUS[job_id])

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def main() -> None:
    port = int(os.environ.get("H3_UI_PORT", "8787"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"H3 Avatar Studio: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
