"""Modal launcher for the H3-Avatar AQ workflow.

AQ remains the local run/benchmark manager. Modal provides remote compute and
persistent volumes for the H3 trainer, model, dataset, checkpoints, and samples.

Common commands:

  modal run modal_h3_avatar.py::preflight
  modal run modal_h3_avatar.py::write_plan
  modal run modal_h3_avatar.py::volume_status

Real training is intentionally explicit:

  modal run modal_h3_avatar.py::train_smoke --trainer-repo https://github.com/IAmIronMan42/MiniMax-H3-FineTuning
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal


APP_NAME = os.environ.get("H3_AVATAR_MODAL_APP", "h3-avatar-aq")

DATA_VOL = modal.Volume.from_name("h3-avatar-data", create_if_missing=True)
RUNS_VOL = modal.Volume.from_name("h3-avatar-runs", create_if_missing=True)
CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("hf-token")

VOLS = {
    "/data": DATA_VOL,
    "/runs": RUNS_VOL,
    "/cache": CACHE_VOL,
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "accelerate",
        "av",
        "datasets",
        "deepspeed",
        "diffusers @ git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc",
        "einops",
        "fastapi[standard]",
        "huggingface-hub",
        "numpy",
        "opencv-python-headless",
        "openai-whisper",
        "peft",
        "pillow",
        "safetensors",
        "sentencepiece",
        "torch",
        "torchaudio",
        "torchcodec",
        "torchvision",
        "transformers",
        "tqdm",
    )
    .add_local_dir(
        ".",
        "/root/h3-avatar-aq",
        ignore=[
            ".git/**",
            ".venv/**",
            "__pycache__/**",
            "artifacts/**",
            "data/**",
            "tmp/**",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


def _run(cmd: list[str], cwd: str | Path = "/root") -> None:
    import subprocess

    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _strip_reference_audio(reference_video: str, output_dir: str) -> str:
    import subprocess

    ref = Path(reference_video)
    out = Path(output_dir) / f"{ref.stem}_noaudio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(ref), "-an", "-c:v", "copy", str(out)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return str(out)


def _project_plan() -> dict:
    path = Path("/root/h3-avatar-aq/configs/progress_checkpoints.json")
    return json.loads(path.read_text(encoding="utf-8"))


@app.function(image=image, volumes=VOLS, gpu="A10G", cpu=4.0, memory=16384, timeout=20 * 60)
def _preflight() -> dict:
    import shutil
    import torch

    roots = ["/data", "/runs", "/cache", "/root/h3-avatar-aq"]
    for root in roots:
        Path(root).mkdir(parents=True, exist_ok=True)
    result = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "roots": {root: shutil.disk_usage(root)._asdict() for root in roots},
        "project_files": sorted(str(p.relative_to("/root/h3-avatar-aq")) for p in Path("/root/h3-avatar-aq").glob("*")),
        "checkpoint_plan": _project_plan(),
    }
    (Path("/runs") / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    return result


@app.local_entrypoint()
def preflight():
    print(json.dumps(_preflight.remote(), indent=2))


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=4096, timeout=10 * 60)
def _write_plan() -> dict:
    project = Path("/root/h3-avatar-aq")
    out_dir = Path("/runs/plans")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "recipe.yaml",
        "configs/progress_checkpoints.json",
        "evals/avatar_benchmarks.json",
        "evals/progress_prompts.jsonl",
    ]
    copied = []
    for rel in files:
        src = project / rel
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        copied.append(str(dest))
    RUNS_VOL.commit()
    return {"copied": copied, "runs_plan_dir": str(out_dir)}


@app.local_entrypoint()
def write_plan():
    print(json.dumps(_write_plan.remote(), indent=2))


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=4096, timeout=10 * 60)
def _volume_status() -> dict:
    out = {}
    for label, root in {"data": "/data", "runs": "/runs", "cache": "/cache"}.items():
        base = Path(root)
        files = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                files.append({"path": str(p), "bytes": p.stat().st_size})
            if len(files) >= 50:
                break
        out[label] = {"exists": base.exists(), "sample_files": files}
    return out


@app.local_entrypoint()
def volume_status():
    print(json.dumps(_volume_status.remote(), indent=2))


@app.function(image=image, volumes=VOLS, secrets=[HF_SECRET], cpu=8.0, memory=32768, timeout=6 * 60 * 60)
def _download_model(model_id: str = "MiniMaxAI/MiniMax-H3", local_dir: str = "/cache/MiniMax-H3") -> dict:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=str(dest),
        token=token,
        resume_download=True,
        max_workers=8,
    )
    CACHE_VOL.commit()
    files = [p for p in dest.rglob("*") if p.is_file()]
    return {
        "model_id": model_id,
        "local_dir": str(dest),
        "snapshot_path": path,
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


@app.local_entrypoint()
def download_model(model_id: str = "MiniMaxAI/MiniMax-H3", local_dir: str = "/cache/MiniMax-H3"):
    print(json.dumps(_download_model.remote(model_id=model_id, local_dir=local_dir), indent=2))


@app.function(image=image, volumes=VOLS, secrets=[HF_SECRET], cpu=8.0, memory=32768, timeout=8 * 60 * 60)
def _download_dataset(
    dataset_id: str = "neosapience/TA2.0_animation_dataset",
    local_dir: str = "/data/TA2.0_animation_dataset",
) -> dict:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(dest),
        token=token,
        resume_download=True,
        max_workers=8,
    )
    DATA_VOL.commit()
    files = [p for p in dest.rglob("*") if p.is_file()]
    return {
        "dataset_id": dataset_id,
        "local_dir": str(dest),
        "snapshot_path": path,
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


@app.local_entrypoint()
def download_dataset(
    dataset_id: str = "neosapience/TA2.0_animation_dataset",
    local_dir: str = "/data/TA2.0_animation_dataset",
):
    print(json.dumps(_download_dataset.remote(dataset_id=dataset_id, local_dir=local_dir), indent=2))


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=4096, timeout=10 * 60)
def _asset_status() -> dict:
    targets = {
        "model": Path("/cache/MiniMax-H3"),
        "dataset": Path("/data/TA2.0_animation_dataset"),
        "trainer": Path("/runs/trainer/MiniMax-H3-FineTuning"),
    }
    out = {}
    for name, root in targets.items():
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
        out[name] = {
            "path": str(root),
            "exists": root.exists(),
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "sample": [str(p) for p in files[:20]],
        }
    return out


@app.local_entrypoint()
def asset_status():
    print(json.dumps(_asset_status.remote(), indent=2))


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=8192, timeout=15 * 60)
def _dataset_status(dataset_dir: str = "/data/TA2.0_animation_dataset") -> dict:
    root = Path(dataset_dir)
    videos = sorted((root / "videos").rglob("*.mp4")) if (root / "videos").exists() else []
    incomplete = sorted(root.rglob("*.incomplete")) if root.exists() else []
    locks = sorted(root.rglob("*.lock")) if root.exists() else []
    metadata = root / "metadata.parquet"
    csv_metadata = root / "metadata_export.csv"
    files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    return {
        "path": str(root),
        "exists": root.exists(),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "videos": len(videos),
        "expected_videos_from_metadata": 11255 if metadata.is_file() or csv_metadata.is_file() else None,
        "incomplete_downloads": len(incomplete),
        "locks": len(locks),
        "metadata": {
            "parquet": metadata.is_file(),
            "csv": csv_metadata.is_file(),
        },
        "sample_videos": [str(p) for p in videos[:5]],
        "sample_incomplete": [str(p) for p in incomplete[:5]],
    }


@app.local_entrypoint()
def dataset_status(dataset_dir: str = "/data/TA2.0_animation_dataset"):
    print(json.dumps(_dataset_status.remote(dataset_dir=dataset_dir), indent=2))


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=8192, timeout=15 * 60)
def _inspect_dataset_metadata() -> dict:
    import pandas as pd

    root = Path("/data/TA2.0_animation_dataset")
    candidates = [
        root / "metadata.parquet",
        root / "metadata_export.csv",
    ]
    found = [p for p in candidates if p.is_file()]
    if not found:
        return {"ok": False, "error": "metadata.parquet / metadata_export.csv not found", "root": str(root)}
    path = found[0]
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    sample = df.head(5).to_dict(orient="records")
    out = {
        "ok": True,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sample": sample,
    }
    dest = Path("/runs/ta20_metadata_inspect.json")
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    RUNS_VOL.commit()
    return out


@app.local_entrypoint()
def inspect_dataset_metadata():
    print(json.dumps(_inspect_dataset_metadata.remote(), indent=2, default=str))


@app.function(image=image, volumes=VOLS, cpu=4.0, memory=16384, timeout=30 * 60)
def _prepare_subset(
    limit: int = 300,
    seed: int = 7,
    require_files: bool = True,
    out_dir: str = "/runs/manifests/ta20_smoke",
) -> dict:
    import subprocess
    import sys

    root = Path("/data/TA2.0_animation_dataset")
    metadata = root / "metadata.parquet"
    if not metadata.is_file():
        metadata = root / "metadata_export.csv"
    if not metadata.is_file():
        raise RuntimeError("TA2.0 metadata missing; wait for dataset download")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "/root/h3-avatar-aq/scripts/prepare_ta20_subset.py",
        "--metadata",
        str(metadata),
        "--root",
        str(root),
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    if require_files:
        cmd.append("--require-files")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    files = {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in sorted(out.glob("*.jsonl"))}
    result = {"stdout": proc.stdout, "out": str(out), "files": files, "require_files": require_files}
    (Path("/runs") / "prepare_subset_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    return result


@app.local_entrypoint()
def prepare_subset(
    limit: int = 300,
    seed: int = 7,
    require_files: bool = True,
    out_dir: str = "/runs/manifests/ta20_smoke",
):
    print(
        json.dumps(
            _prepare_subset.remote(limit=limit, seed=seed, require_files=require_files, out_dir=out_dir),
            indent=2,
        )
    )


@app.function(image=image, volumes=VOLS, cpu=4.0, memory=16384, timeout=30 * 60)
def _prepare_ta20_visual_v2_manifest(
    limit: int = 1000,
    seed: int = 23,
    out_dir: str = "/runs/manifests/ta20_visual_v2_1k",
    max_per_ref: int = 2,
    prefer_language: str = "en",
    active_bonus: float = 0.25,
    min_frames: int = 80,
    require_files: bool = True,
) -> dict:
    import csv
    import random
    from collections import defaultdict

    root = Path("/data/TA2.0_animation_dataset")
    metadata = root / "metadata_export.csv"
    if not metadata.is_file():
        raise RuntimeError(f"TA2.0 metadata missing: {metadata}")

    def as_float(value, default: float = 0.0) -> float:
        try:
            if value in (None, "", "nan", "NaN"):
                return default
            return float(value)
        except Exception:
            return default

    def as_int(value, default: int = 0) -> int:
        try:
            if value in (None, "", "nan", "NaN"):
                return default
            return int(float(value))
        except Exception:
            return default

    def video_path(row: dict) -> Path:
        name = row.get("file_name") or row.get("video_path") or row.get("path") or row.get("mp4") or ""
        p = Path(str(name))
        return p if p.is_absolute() else root / p

    def score_row(row: dict) -> float:
        motion = str(row.get("motion_mode") or "").lower()
        language = str(row.get("language") or row.get("lang") or "").lower()
        frames = as_int(row.get("frame_num") or row.get("frames"))
        vqa = as_float(row.get("vqa_score") or row.get("quality_score"))
        c50 = as_float(row.get("c50"))
        snr = as_float(row.get("snr"))
        pad = (
            as_float(row.get("pad_top"))
            + as_float(row.get("pad_bottom"))
            + as_float(row.get("pad_left"))
            + as_float(row.get("pad_right"))
        )
        score = vqa
        score += min(max(c50, 0.0), 50.0) / 250.0
        score += min(max(snr, 0.0), 50.0) / 250.0
        score += active_bonus if motion == "active" else 0.0
        score += 0.10 if language == prefer_language.lower() else 0.0
        score += 0.05 if frames >= 124 else 0.0
        score -= min(pad / 512.0, 0.25)
        return score

    def normalize(row: dict) -> dict:
        path = video_path(row)
        motion = str(row.get("motion_mode") or "unknown")
        language = str(row.get("language") or row.get("lang") or "unknown")
        clip_id = str(row.get("clip_id") or path.stem)
        ref_id = str(row.get("ref_id") or row.get("character_id") or clip_id)
        caption = (
            "<Video 1> The same animated character talks directly to the camera "
            "with natural mouth and head motion while preserving the reference identity."
        )
        return {
            "clip_id": clip_id,
            "ref_id": ref_id,
            "video_path": str(path),
            "reference_path": "",
            "audio_path": "",
            "caption": caption,
            "motion_mode": motion,
            "language": language,
            "duration_sec": row.get("duration_sec") or row.get("duration") or "",
            "frame_num": row.get("frame_num") or row.get("frames") or "",
            "quality_score": as_float(row.get("vqa_score") or row.get("quality_score")),
            "vqa_score": as_float(row.get("vqa_score")),
            "c50": as_float(row.get("c50")),
            "snr": as_float(row.get("snr")),
            "curation_score": score_row(row),
        }

    with metadata.open(newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))

    rows = []
    reject_counts = {"missing_file": 0, "short": 0}
    for raw in raw_rows:
        frames = as_int(raw.get("frame_num") or raw.get("frames"))
        if frames and frames < min_frames:
            reject_counts["short"] += 1
            continue
        path = video_path(raw)
        if require_files and not path.is_file():
            reject_counts["missing_file"] += 1
            continue
        rows.append(normalize(raw))

    by_ref: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ref[row["ref_id"]].append(row)

    selected: list[dict] = []
    rng = random.Random(seed)
    refs = sorted(by_ref, key=lambda ref: max(r["curation_score"] for r in by_ref[ref]), reverse=True)
    # Shuffle within close-quality bands so the set is not only the first dataset style.
    banded_refs = []
    for i in range(0, len(refs), 100):
        band = refs[i : i + 100]
        rng.shuffle(band)
        banded_refs.extend(band)
    for ref_id in banded_refs:
        clips = sorted(by_ref[ref_id], key=lambda r: r["curation_score"], reverse=True)[:max_per_ref]
        for row in clips:
            if len(selected) >= limit:
                break
            selected.append(row)
        if len(selected) >= limit:
            break

    if not selected:
        raise RuntimeError("No usable TA2.0 rows after visual_v2 filtering")

    selected_refs = sorted({r["ref_id"] for r in selected})
    rng.shuffle(selected_refs)
    n_ref = len(selected_refs)
    n_test = max(1, int(n_ref * 0.10)) if n_ref >= 3 else 0
    n_val = max(1, int(n_ref * 0.10)) if n_ref >= 3 else 0
    test_refs = set(selected_refs[:n_test])
    val_refs = set(selected_refs[n_test : n_test + n_val])

    def h3_row(row: dict) -> dict:
        return {
            "id": row["clip_id"],
            "caption": row["caption"],
            "video": row["video_path"],
            "target_video": row["video_path"],
            "target_audio": None,
            "reference_images": [],
            "reference_videos": [row["video_path"]],
            "reference_audios": [],
            "metadata": {
                "ref_id": row["ref_id"],
                "motion_mode": row["motion_mode"],
                "language": row["language"],
                "quality_score": row["quality_score"],
                "curation_score": row["curation_score"],
            },
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=True) + "\n")

    train_rows = [r for r in selected if r["ref_id"] not in val_refs and r["ref_id"] not in test_refs]
    val_rows = [r for r in selected if r["ref_id"] in val_refs]
    test_rows = [r for r in selected if r["ref_id"] in test_refs]
    write_jsonl(out / "ta20_visual_v2_manifest.jsonl", selected)
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    write_jsonl(out / "test.jsonl", test_rows)
    write_jsonl(out / "h3_train.jsonl", [h3_row(r) for r in train_rows])
    write_jsonl(out / "h3_val.jsonl", [h3_row(r) for r in val_rows])
    write_jsonl(out / "h3_test.jsonl", [h3_row(r) for r in test_rows])

    summary = {
        "out": str(out),
        "source_rows": len(raw_rows),
        "usable_rows": len(rows),
        "selected": len(selected),
        "selected_refs": len(selected_refs),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "max_per_ref": max_per_ref,
        "prefer_language": prefer_language,
        "min_frames": min_frames,
        "reject_counts": reject_counts,
        "top_examples": selected[:5],
        "files": {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in sorted(out.glob("*.jsonl"))},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path("/runs") / "prepare_ta20_visual_v2_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    return summary


@app.function(image=image, volumes=VOLS, cpu=4.0, memory=16384, timeout=30 * 60)
def _prepare_ta20_audio_v1_manifest(
    limit: int = 500,
    seed: int = 31,
    out_dir: str = "/runs/manifests/ta20_audio_v1_en_500",
    max_per_ref: int = 2,
    min_frames: int = 124,
    min_vqa: float = 0.45,
    min_snr: float = 8.0,
    min_c50: float = 10.0,
    require_files: bool = True,
) -> dict:
    import csv
    import random
    from collections import defaultdict

    root = Path("/data/TA2.0_animation_dataset")
    metadata = root / "metadata_export.csv"
    if not metadata.is_file():
        raise RuntimeError(f"TA2.0 metadata missing: {metadata}")

    def as_float(value, default: float = 0.0) -> float:
        try:
            if value in (None, "", "nan", "NaN"):
                return default
            return float(value)
        except Exception:
            return default

    def as_int(value, default: int = 0) -> int:
        try:
            if value in (None, "", "nan", "NaN"):
                return default
            return int(float(value))
        except Exception:
            return default

    def video_path(row: dict) -> Path:
        name = row.get("file_name") or row.get("video_path") or row.get("path") or row.get("mp4") or ""
        p = Path(str(name))
        return p if p.is_absolute() else root / p

    def score_row(row: dict) -> float:
        vqa = as_float(row.get("vqa_score") or row.get("quality_score"))
        snr = as_float(row.get("snr"))
        c50 = as_float(row.get("c50"))
        frames = as_int(row.get("frame_num") or row.get("frames"))
        duration = as_float(row.get("duration_sec") or row.get("duration"))
        score = vqa
        score += min(max(snr, 0.0), 50.0) / 100.0
        score += min(max(c50, 0.0), 50.0) / 100.0
        score += 0.10 if frames >= 124 else 0.0
        score += 0.10 if 4.0 <= duration <= 10.5 else 0.0
        return score

    def normalize(row: dict) -> dict:
        path = video_path(row)
        clip_id = str(row.get("clip_id") or path.stem)
        ref_id = str(row.get("ref_id") or row.get("character_id") or clip_id)
        caption = (
            "<Video 1> The same animated character speaks clear English directly to the camera "
            "with natural voice, synchronized mouth movement, subtle head motion, and a friendly expression."
        )
        return {
            "clip_id": clip_id,
            "ref_id": ref_id,
            "video_path": str(path),
            "caption": caption,
            "motion_mode": str(row.get("motion_mode") or "unknown"),
            "language": str(row.get("language") or row.get("lang") or "unknown"),
            "duration_sec": row.get("duration_sec") or row.get("duration") or "",
            "frame_num": row.get("frame_num") or row.get("frames") or "",
            "vqa_score": as_float(row.get("vqa_score")),
            "c50": as_float(row.get("c50")),
            "snr": as_float(row.get("snr")),
            "curation_score": score_row(row),
        }

    reject_counts = {
        "missing_file": 0,
        "non_english": 0,
        "not_active": 0,
        "short": 0,
        "low_vqa": 0,
        "low_snr": 0,
        "low_c50": 0,
    }
    with metadata.open(newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))

    rows = []
    for raw in raw_rows:
        language = str(raw.get("language") or raw.get("lang") or "").lower()
        motion = str(raw.get("motion_mode") or "").lower()
        frames = as_int(raw.get("frame_num") or raw.get("frames"))
        vqa = as_float(raw.get("vqa_score") or raw.get("quality_score"))
        snr = as_float(raw.get("snr"))
        c50 = as_float(raw.get("c50"))
        path = video_path(raw)
        if language != "en":
            reject_counts["non_english"] += 1
            continue
        if motion != "active":
            reject_counts["not_active"] += 1
            continue
        if frames and frames < min_frames:
            reject_counts["short"] += 1
            continue
        if vqa < min_vqa:
            reject_counts["low_vqa"] += 1
            continue
        if snr < min_snr:
            reject_counts["low_snr"] += 1
            continue
        if c50 < min_c50:
            reject_counts["low_c50"] += 1
            continue
        if require_files and not path.is_file():
            reject_counts["missing_file"] += 1
            continue
        rows.append(normalize(raw))

    by_ref: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ref[row["ref_id"]].append(row)

    rng = random.Random(seed)
    refs = sorted(by_ref, key=lambda ref: max(r["curation_score"] for r in by_ref[ref]), reverse=True)
    banded_refs = []
    for i in range(0, len(refs), 100):
        band = refs[i : i + 100]
        rng.shuffle(band)
        banded_refs.extend(band)

    selected: list[dict] = []
    for ref_id in banded_refs:
        clips = sorted(by_ref[ref_id], key=lambda r: r["curation_score"], reverse=True)[:max_per_ref]
        for row in clips:
            if len(selected) >= limit:
                break
            selected.append(row)
        if len(selected) >= limit:
            break
    if not selected:
        raise RuntimeError("No usable TA2.0 rows after audio_v1 filtering")

    selected_refs = sorted({r["ref_id"] for r in selected})
    rng.shuffle(selected_refs)
    n_ref = len(selected_refs)
    n_test = max(1, int(n_ref * 0.10)) if n_ref >= 3 else 0
    n_val = max(1, int(n_ref * 0.10)) if n_ref >= 3 else 0
    test_refs = set(selected_refs[:n_test])
    val_refs = set(selected_refs[n_test : n_test + n_val])

    def h3_row(row: dict) -> dict:
        return {
            "id": row["clip_id"],
            "caption": row["caption"],
            "video": row["video_path"],
            "target_video": row["video_path"],
            "target_audio": row["video_path"],
            "reference_images": [],
            "reference_videos": [row["video_path"]],
            "reference_audios": [row["video_path"]],
            "metadata": {
                "ref_id": row["ref_id"],
                "motion_mode": row["motion_mode"],
                "language": row["language"],
                "vqa_score": row["vqa_score"],
                "c50": row["c50"],
                "snr": row["snr"],
                "curation_score": row["curation_score"],
                "audio_v1": True,
            },
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=True) + "\n")

    train_rows = [r for r in selected if r["ref_id"] not in val_refs and r["ref_id"] not in test_refs]
    val_rows = [r for r in selected if r["ref_id"] in val_refs]
    test_rows = [r for r in selected if r["ref_id"] in test_refs]
    write_jsonl(out / "ta20_audio_v1_manifest.jsonl", selected)
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    write_jsonl(out / "test.jsonl", test_rows)
    write_jsonl(out / "h3_train.jsonl", [h3_row(r) for r in train_rows])
    write_jsonl(out / "h3_val.jsonl", [h3_row(r) for r in val_rows])
    write_jsonl(out / "h3_test.jsonl", [h3_row(r) for r in test_rows])

    summary = {
        "out": str(out),
        "source_rows": len(raw_rows),
        "usable_rows": len(rows),
        "selected": len(selected),
        "selected_refs": len(selected_refs),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "filters": {
            "language": "en",
            "motion_mode": "active",
            "min_frames": min_frames,
            "min_vqa": min_vqa,
            "min_snr": min_snr,
            "min_c50": min_c50,
            "max_per_ref": max_per_ref,
        },
        "reject_counts": reject_counts,
        "top_examples": selected[:5],
        "files": {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in sorted(out.glob("*.jsonl"))},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path("/runs") / "prepare_ta20_audio_v1_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    return summary


@app.local_entrypoint()
def prepare_ta20_audio_v1_manifest(
    limit: int = 500,
    seed: int = 31,
    out_dir: str = "/runs/manifests/ta20_audio_v1_en_500",
    max_per_ref: int = 2,
    min_frames: int = 124,
    min_vqa: float = 0.45,
    min_snr: float = 8.0,
    min_c50: float = 10.0,
    require_files: bool = True,
):
    print(
        json.dumps(
            _prepare_ta20_audio_v1_manifest.remote(
                limit=limit,
                seed=seed,
                out_dir=out_dir,
                max_per_ref=max_per_ref,
                min_frames=min_frames,
                min_vqa=min_vqa,
                min_snr=min_snr,
                min_c50=min_c50,
                require_files=require_files,
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def prepare_ta20_visual_v2_manifest(
    limit: int = 1000,
    seed: int = 23,
    out_dir: str = "/runs/manifests/ta20_visual_v2_1k",
    max_per_ref: int = 2,
    prefer_language: str = "en",
    min_frames: int = 80,
    require_files: bool = True,
):
    print(
        json.dumps(
            _prepare_ta20_visual_v2_manifest.remote(
                limit=limit,
                seed=seed,
                out_dir=out_dir,
                max_per_ref=max_per_ref,
                prefer_language=prefer_language,
                min_frames=min_frames,
                require_files=require_files,
            ),
            indent=2,
        )
    )


@app.function(image=image, volumes=VOLS, gpu="L40S", cpu=8.0, memory=32768, timeout=8 * 60 * 60)
def _prepare_ta20_audio_v2_transcripts(
    manifest: str = "/runs/manifests/ta20_audio_v1_en_500/ta20_audio_v1_manifest.jsonl",
    output_dir: str = "/runs/manifests/ta20_audio_v2_transcripts_en_300",
    limit: int = 300,
    model_name: str = "small.en",
    min_words: int = 3,
    max_words: int = 24,
    max_no_speech_prob: float = 0.60,
    min_avg_logprob: float = -1.25,
) -> dict:
    import re
    import subprocess
    import tempfile

    import whisper

    src = Path(manifest)
    if not src.is_file():
        raise RuntimeError(f"manifest missing: {src}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(model_name, device="cuda", download_root="/cache/whisper")
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[:limit]

    accepted = []
    rejected = []

    def words(text: str) -> list[str]:
        return re.findall(r"[A-Za-z][A-Za-z']*", text)

    for idx, row in enumerate(rows, start=1):
        video = Path(row.get("video_path") or row.get("target_video") or row.get("video") or "")
        if not video.is_file():
            rejected.append({"row": row, "reason": "missing_video"})
            continue
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                check=True,
            )
            result = model.transcribe(
                str(wav),
                language="en",
                task="transcribe",
                fp16=True,
                temperature=0.0,
                condition_on_previous_text=False,
            )
        text = re.sub(r"\s+", " ", str(result.get("text") or "")).strip()
        segs = result.get("segments") or []
        no_speech = max((float(s.get("no_speech_prob", 0.0)) for s in segs), default=0.0)
        avg_logprob = sum(float(s.get("avg_logprob", -99.0)) for s in segs) / max(1, len(segs))
        w = words(text)
        item = {
            **row,
            "transcript": text,
            "word_count": len(w),
            "avg_logprob": avg_logprob,
            "max_no_speech_prob": no_speech,
            "asr_model": model_name,
        }
        reason = None
        if len(w) < min_words:
            reason = "too_few_words"
        elif len(w) > max_words:
            reason = "too_many_words"
        elif no_speech > max_no_speech_prob:
            reason = "high_no_speech"
        elif avg_logprob < min_avg_logprob:
            reason = "low_avg_logprob"
        elif not text:
            reason = "empty"
        if reason:
            item["reject_reason"] = reason
            rejected.append(item)
        else:
            accepted.append(item)
        print(f"[{idx}/{len(rows)}] accepted={len(accepted)} rejected={len(rejected)} text={text[:80]!r}", flush=True)

    def h3_row(row: dict) -> dict:
        video = row.get("video_path") or row.get("target_video") or row.get("video")
        transcript = row["transcript"].replace('"', "'")
        return {
            "id": row["clip_id"],
            "caption": (
                f'<Video 1> The same character says in English: "{transcript}" '
                "with natural voice, synchronized mouth movement, subtle head motion, and a friendly expression."
            ),
            "video": video,
            "target_video": video,
            "target_audio": video,
            "reference_images": [],
            "reference_videos": [video],
            "reference_audios": [video],
            "metadata": {
                "ref_id": row.get("ref_id"),
                "language": row.get("language", "en"),
                "transcript": row["transcript"],
                "word_count": row["word_count"],
                "avg_logprob": row["avg_logprob"],
                "max_no_speech_prob": row["max_no_speech_prob"],
                "audio_v2_transcript": True,
            },
        }

    # Split by ref_id so identities do not leak across train/val/test.
    refs = sorted({str(r.get("ref_id")) for r in accepted})
    import random

    rng = random.Random(42)
    rng.shuffle(refs)
    n_test = max(1, int(len(refs) * 0.10)) if len(refs) >= 3 else 0
    n_val = max(1, int(len(refs) * 0.10)) if len(refs) >= 3 else 0
    test_refs = set(refs[:n_test])
    val_refs = set(refs[n_test : n_test + n_val])
    train = [r for r in accepted if str(r.get("ref_id")) not in val_refs and str(r.get("ref_id")) not in test_refs]
    val = [r for r in accepted if str(r.get("ref_id")) in val_refs]
    test = [r for r in accepted if str(r.get("ref_id")) in test_refs]

    def write_jsonl(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=True) + "\n")

    write_jsonl(out / "accepted_transcripts.jsonl", accepted)
    write_jsonl(out / "rejected_transcripts.jsonl", rejected)
    write_jsonl(out / "h3_train.jsonl", [h3_row(r) for r in train])
    write_jsonl(out / "h3_val.jsonl", [h3_row(r) for r in val])
    write_jsonl(out / "h3_test.jsonl", [h3_row(r) for r in test])
    summary = {
        "out": str(out),
        "source_manifest": str(src),
        "model_name": model_name,
        "limit": limit,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "filters": {
            "min_words": min_words,
            "max_words": max_words,
            "max_no_speech_prob": max_no_speech_prob,
            "min_avg_logprob": min_avg_logprob,
        },
        "examples": accepted[:8],
        "reject_examples": rejected[:8],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path("/runs") / "prepare_ta20_audio_v2_transcripts_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    return summary


@app.local_entrypoint()
def prepare_ta20_audio_v2_transcripts(
    manifest: str = "/runs/manifests/ta20_audio_v1_en_500/ta20_audio_v1_manifest.jsonl",
    output_dir: str = "/runs/manifests/ta20_audio_v2_transcripts_en_300",
    limit: int = 300,
    model_name: str = "small.en",
    min_words: int = 3,
    max_words: int = 24,
    max_no_speech_prob: float = 0.60,
    min_avg_logprob: float = -1.25,
):
    print(
        json.dumps(
            _prepare_ta20_audio_v2_transcripts.remote(
                manifest=manifest,
                output_dir=output_dir,
                limit=limit,
                model_name=model_name,
                min_words=min_words,
                max_words=max_words,
                max_no_speech_prob=max_no_speech_prob,
                min_avg_logprob=min_avg_logprob,
            ),
            indent=2,
        )
    )


@app.function(image=image, volumes=VOLS, cpu=4.0, memory=8192, timeout=15 * 60)
def _validate_h3_manifest(manifest: str = "/runs/manifests/ta20_smoke/h3_train.jsonl", check_files: bool = True) -> dict:
    trainer = Path("/runs/trainer/MiniMax-H3-FineTuning")
    path = Path(manifest)
    if not path.is_file():
        raise RuntimeError(f"manifest missing: {path}")
    cmd = ["python", "validate_manifest.py", "--manifest", str(path)]
    if check_files:
        cmd.append("--check-files")
    import subprocess

    proc = subprocess.run(cmd, cwd=str(trainer), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = {"returncode": proc.returncode, "stdout": proc.stdout, "manifest": str(path)}
    (Path("/runs") / "validate_h3_manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return out


@app.local_entrypoint()
def validate_h3_manifest(manifest: str = "/runs/manifests/ta20_smoke/h3_train.jsonl", check_files: bool = True):
    print(json.dumps(_validate_h3_manifest.remote(manifest=manifest, check_files=check_files), indent=2))


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=4096, timeout=15 * 60)
def _export_manifest_clips(
    manifest: str = "/runs/manifests/ta20_smoke/h3_train.jsonl",
    output_dir: str = "/runs/samples/ta20_manifest_clips",
    limit: int = 12,
) -> dict:
    import shutil

    manifest_path = Path(manifest)
    out = Path(output_dir)
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest missing: {manifest_path}")
    out.mkdir(parents=True, exist_ok=True)

    copied = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if len(copied) >= limit:
            break
        if not line.strip():
            continue
        row = json.loads(line)
        src = Path(row.get("target_video") or row.get("video") or "")
        if not src.is_file():
            continue
        dest = out / f"{len(copied):03d}_{src.name}"
        shutil.copy2(src, dest)
        copied.append({"src": str(src), "dest": str(dest), "bytes": dest.stat().st_size})

    RUNS_VOL.commit()
    return {"manifest": str(manifest_path), "output_dir": str(out), "copied": copied}


@app.local_entrypoint()
def export_manifest_clips(
    manifest: str = "/runs/manifests/ta20_smoke/h3_train.jsonl",
    output_dir: str = "/runs/samples/ta20_manifest_clips",
    limit: int = 12,
):
    print(json.dumps(_export_manifest_clips.remote(manifest=manifest, output_dir=output_dir, limit=limit), indent=2))


@app.function(image=image, volumes=VOLS, gpu="H100", cpu=8.0, memory=32768, timeout=30 * 60)
def _inspect_inference_api() -> dict:
    import inspect
    import pkgutil

    import diffusers

    out: dict[str, object] = {
        "diffusers_version": getattr(diffusers, "__version__", "unknown"),
        "minimax_attrs": [name for name in dir(diffusers) if "MiniMax" in name or "H3" in name],
    }
    candidates = []
    for mod in pkgutil.walk_packages(diffusers.__path__, diffusers.__name__ + "."):
        name = mod.name
        if "minimax" in name.lower() or "h3" in name.lower():
            candidates.append(name)
    out["candidate_modules"] = candidates[:200]
    details = {}
    for attr in out["minimax_attrs"]:
        obj = getattr(diffusers, str(attr))
        try:
            details[str(attr)] = {
                "module": getattr(obj, "__module__", ""),
                "signature": str(inspect.signature(obj.__call__)) if hasattr(obj, "__call__") else "",
                "from_pretrained": str(inspect.signature(obj.from_pretrained)) if hasattr(obj, "from_pretrained") else "",
            }
        except Exception as exc:
            details[str(attr)] = {"error": repr(exc)}
    out["details"] = details
    dest = Path("/runs/inference_api_inspect.json")
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    RUNS_VOL.commit()
    return out


@app.local_entrypoint()
def inspect_inference_api():
    print(json.dumps(_inspect_inference_api.remote(), indent=2, default=str))


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=8192, timeout=15 * 60)
def _inspect_reference_classes() -> dict:
    import importlib
    import inspect
    import pkgutil

    import diffusers.modular_pipelines.minimax_h3 as h3

    found = {}
    for mod in pkgutil.walk_packages(h3.__path__, h3.__name__ + "."):
        module = importlib.import_module(mod.name)
        names = [name for name in dir(module) if "Reference" in name]
        if names:
            found[mod.name] = {
                name: {
                    "repr": repr(getattr(module, name)),
                    "signature": str(inspect.signature(getattr(module, name))) if callable(getattr(module, name)) else "",
                    "has_from_file": hasattr(getattr(module, name), "from_file"),
                }
                for name in names
            }
    return found


@app.local_entrypoint()
def inspect_reference_classes():
    print(json.dumps(_inspect_reference_classes.remote(), indent=2, default=str))


@app.function(image=image, volumes=VOLS, cpu=4.0, memory=16384, timeout=30 * 60)
def _inspect_ref2va_pipeline_doc() -> dict:
    from diffusers import ComponentsManager, MiniMaxH3Ref2VAModularPipeline

    manager = ComponentsManager()
    pipe = MiniMaxH3Ref2VAModularPipeline.from_pretrained("/cache/MiniMax-H3", components_manager=manager)
    out = {
        "class": type(pipe).__name__,
        "doc": str(getattr(pipe, "doc", "")),
        "blocks": repr(getattr(pipe, "blocks", None)),
        "config": getattr(pipe, "config", {}),
    }
    try:
        out["expected_components"] = list(pipe.components.keys())
    except Exception as exc:
        out["expected_components_error"] = repr(exc)
    try:
        out["workflow_names"] = list(pipe.blocks._workflows.keys())
    except Exception as exc:
        out["workflow_names_error"] = repr(exc)
    out = json.loads(json.dumps(out, default=str))
    dest = Path("/runs/ref2va_pipeline_doc.json")
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    RUNS_VOL.commit()
    return out


@app.local_entrypoint()
def inspect_ref2va_pipeline_doc():
    print(json.dumps(_inspect_ref2va_pipeline_doc.remote(), indent=2, default=str))


@app.function(image=image, volumes=VOLS, cpu=4.0, memory=16384, timeout=30 * 60)
def _inspect_minimax_h3_source() -> dict:
    import importlib
    import inspect

    modules = [
        "diffusers.modular_pipelines.minimax_h3",
        "diffusers.modular_pipelines.minimax_h3.before_encoder",
        "diffusers.modular_pipelines.minimax_h3.packing_ref2va",
        "diffusers.modular_pipelines.minimax_h3.modular_pipeline",
    ]
    out: dict[str, object] = {}
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            out[module_name] = {"import_error": repr(exc)}
            continue
        entries = {}
        for name in dir(module):
            if "Ref" in name or "Reference" in name or "MiniMaxH3" in name:
                obj = getattr(module, name)
                try:
                    source = inspect.getsource(obj)
                except Exception as exc:
                    source = f"<source unavailable: {exc!r}>"
                entries[name] = {
                    "module": getattr(obj, "__module__", ""),
                    "signature": str(inspect.signature(obj)) if callable(obj) else "",
                    "source": source[:20000],
                }
        out[module_name] = entries
    out = json.loads(json.dumps(out, default=str))
    dest = Path("/runs/minimax_h3_source_inspect.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    return out


@app.local_entrypoint()
def inspect_minimax_h3_source():
    print(json.dumps(_inspect_minimax_h3_source.remote(), indent=2, default=str))


@app.function(image=image, volumes=VOLS, gpu="H100", cpu=16.0, memory=262144, timeout=4 * 60 * 60)
def _generate_ref2va_sample(
    checkpoint: str = "/runs/h3/ref2va-lora-smoke-100/checkpoint-100.pt",
    reference_video: str = "/runs/samples/ta20_manifest_clips/000_clip_00603.mp4",
    output_path: str = "/runs/generated/ref2va-lora-smoke-100/sample_000.mp4",
    prompt: str = "The same animated character talks directly to the camera with natural lip movement and a friendly expression.",
    num_frames: int = 124,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 8,
    seed: int = 42,
    include_reference_audio: bool = True,
    generate_audio: bool = True,
) -> dict:
    import subprocess
    import torch
    from diffusers import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks, MiniMaxH3Reference
    from diffusers.utils.export_utils import encode_video
    from peft import LoraConfig

    use_lora = checkpoint.lower() not in {"", "base", "none", "null"}
    ckpt = Path(checkpoint) if use_lora else None
    ref = Path(reference_video)
    out = Path(output_path)
    if use_lora and not ckpt.is_file():
        raise RuntimeError(f"checkpoint missing: {ckpt}")
    if not ref.is_file():
        raise RuntimeError(f"reference video missing: {ref}")
    out.parent.mkdir(parents=True, exist_ok=True)

    manager = ComponentsManager()
    pipe = MiniMaxH3Ref2VABlocks().init_pipeline("/cache/MiniMax-H3", components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")

    missing = []
    transformer = getattr(pipe, "transformer_ref", None) or getattr(pipe, "transformer", None)
    if transformer is None:
        raise RuntimeError("No transformer_ref/transformer component found on pipeline")
    if use_lora:
        transformer.add_adapter(
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=["to_qkv", "to_out.0", "linear_1", "linear_2"],
                init_lora_weights="gaussian",
            )
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = transformer.load_state_dict(state["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys: {unexpected[:5]}")
    transformer.requires_grad_(False)
    try:
        transformer.set_attention_backend("_flash_3_hub")
    except Exception as exc:
        print(f"flash attention backend not enabled: {exc}", flush=True)

    reference_path = ref
    if not include_reference_audio:
        reference_path = out.parent / f"{ref.stem}_noaudio.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(ref), "-an", "-c:v", "copy", str(reference_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    reference = MiniMaxH3Reference(video=str(reference_path))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    outputs = ["videos", "audio", "sampling_rate"] if generate_audio else ["videos"]
    results = pipe(
        prompt=prompt,
        references=[reference],
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output=outputs,
    )
    if generate_audio:
        encode_video(
            results["videos"][0],
            fps=24,
            output_path=str(out),
            audio=results["audio"][0],
            audio_sample_rate=results["sampling_rate"],
        )
    else:
        encode_video(results["videos"][0], fps=24, output_path=str(out))
    RUNS_VOL.commit()
    return {
        "checkpoint": str(ckpt) if use_lora else "base",
        "reference_video": str(ref),
        "output_path": str(out),
        "bytes": out.stat().st_size,
        "prompt": prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "include_reference_audio": include_reference_audio,
        "generate_audio": generate_audio,
        "missing_key_count": len(missing),
    }


@app.local_entrypoint()
def generate_ref2va_sample(
    checkpoint: str = "/runs/h3/ref2va-lora-smoke-100/checkpoint-100.pt",
    reference_video: str = "/runs/samples/ta20_manifest_clips/000_clip_00603.mp4",
    output_path: str = "/runs/generated/ref2va-lora-smoke-100/sample_000.mp4",
    prompt: str = "The same animated character talks directly to the camera with natural lip movement and a friendly expression.",
    num_frames: int = 124,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 8,
    seed: int = 42,
    include_reference_audio: bool = True,
    generate_audio: bool = True,
):
    print(
        json.dumps(
            _generate_ref2va_sample.remote(
                checkpoint=checkpoint,
                reference_video=reference_video,
                output_path=output_path,
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                seed=seed,
                include_reference_audio=include_reference_audio,
                generate_audio=generate_audio,
            ),
            indent=2,
        )
    )


@app.cls(
    image=image,
    volumes=VOLS,
    gpu="H100",
    cpu=16.0,
    memory=262144,
    timeout=4 * 60 * 60,
    scaledown_window=20 * 60,
    max_containers=1,
)
class H3WarmRef2VAGenerator:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import ComponentsManager
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks
        from peft import LoraConfig

        self.torch = torch
        self.ComponentsManager = ComponentsManager
        self.MiniMaxH3Ref2VABlocks = MiniMaxH3Ref2VABlocks
        self.LoraConfig = LoraConfig
        self.manager = ComponentsManager()
        self.pipe = MiniMaxH3Ref2VABlocks().init_pipeline("/cache/MiniMax-H3", components_manager=self.manager)
        self.pipe.load_components(dtype=torch.bfloat16)
        self.manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
        self.transformer = getattr(self.pipe, "transformer_ref", None) or getattr(self.pipe, "transformer", None)
        if self.transformer is None:
            raise RuntimeError("No transformer_ref/transformer component found on pipeline")
        self.loaded_checkpoint = None
        self.missing_key_count = 0
        try:
            self.transformer.set_attention_backend("_flash_3_hub")
        except Exception as exc:
            print(f"flash attention backend not enabled: {exc}", flush=True)

    def _load_lora(self, checkpoint: str) -> None:
        ckpt = Path(checkpoint)
        if self.loaded_checkpoint == str(ckpt):
            return
        if not ckpt.is_file():
            raise RuntimeError(f"checkpoint missing: {ckpt}")
        if self.loaded_checkpoint is None:
            self.transformer.add_adapter(
                self.LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    target_modules=["to_qkv", "to_out.0", "linear_1", "linear_2"],
                    init_lora_weights="gaussian",
                )
            )
        state = self.torch.load(ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = self.transformer.load_state_dict(state["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys: {unexpected[:5]}")
        self.transformer.requires_grad_(False)
        self.loaded_checkpoint = str(ckpt)
        self.missing_key_count = len(missing)

    @modal.method()
    def generate(self, payload: dict) -> dict:
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
        from diffusers.utils.export_utils import encode_video

        checkpoint = payload.get("checkpoint") or "/runs/h3/locked/h3-avatar-audio-v2-step1000.pt"
        reference_video = payload.get("reference_video")
        output_path = payload.get("output_path")
        prompt = payload.get("prompt") or "The same character faces the camera and speaks with an audible natural voice."
        num_frames = int(payload.get("num_frames", 124))
        height = int(payload.get("height", 512))
        width = int(payload.get("width", 512))
        num_inference_steps = int(payload.get("num_inference_steps", 30))
        seed = int(payload.get("seed", 1500))
        include_reference_audio = bool(payload.get("include_reference_audio", False))
        generate_audio = bool(payload.get("generate_audio", True))

        if not reference_video:
            raise RuntimeError("reference_video is required")
        if not output_path:
            raise RuntimeError("output_path is required")
        ref = Path(reference_video)
        out = Path(output_path)
        if not ref.is_file():
            raise RuntimeError(f"reference video missing: {ref}")
        out.parent.mkdir(parents=True, exist_ok=True)

        if str(checkpoint).lower() not in {"", "base", "none", "null"}:
            self._load_lora(str(checkpoint))

        reference_path = str(ref)
        if not include_reference_audio:
            reference_path = _strip_reference_audio(str(ref), str(out.parent))
        reference = MiniMaxH3Reference(video=reference_path)
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        outputs = ["videos", "audio", "sampling_rate"] if generate_audio else ["videos"]
        results = self.pipe(
            prompt=prompt,
            references=[reference],
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output=outputs,
        )
        if generate_audio:
            encode_video(
                results["videos"][0],
                fps=24,
                output_path=str(out),
                audio=results["audio"][0],
                audio_sample_rate=results["sampling_rate"],
            )
        else:
            encode_video(results["videos"][0], fps=24, output_path=str(out))
        RUNS_VOL.commit()
        return {
            "checkpoint": str(checkpoint),
            "reference_video": str(ref),
            "output_path": str(out),
            "bytes": out.stat().st_size,
            "prompt": prompt,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "include_reference_audio": include_reference_audio,
            "generate_audio": generate_audio,
            "missing_key_count": self.missing_key_count,
            "warm": True,
        }


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=4096, timeout=4 * 60 * 60)
@modal.fastapi_endpoint(method="POST", label="h3-avatar-warm-generate", docs=True)
def generate_ref2va_warm_endpoint(payload: dict) -> dict:
    return H3WarmRef2VAGenerator().generate.remote(payload)


@app.function(image=image, volumes=VOLS, gpu="H100", cpu=16.0, memory=262144, timeout=8 * 60 * 60)
def _generate_ref2va_eval_batch(
    checkpoint: str = "/runs/h3/ref2va-lora-smoke-100/checkpoint-100.pt",
    reference_dir: str = "/runs/samples/ta20_val_eval_clips",
    output_dir: str = "/runs/generated/ref2va-lora-smoke-100/eval_batch_001",
    limit: int = 5,
    prompt: str = "The same animated character talks directly to the camera with natural lip movement and a friendly expression.",
    num_frames: int = 124,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 6,
    seed: int = 100,
    include_reference_audio: bool = True,
    generate_audio: bool = True,
) -> dict:
    import subprocess
    import torch
    from diffusers import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks, MiniMaxH3Reference
    from diffusers.utils.export_utils import encode_video
    from peft import LoraConfig

    ckpt = Path(checkpoint)
    refs = sorted(Path(reference_dir).glob("*.mp4"))[:limit]
    out_root = Path(output_dir)
    if not ckpt.is_file():
        raise RuntimeError(f"checkpoint missing: {ckpt}")
    if not refs:
        raise RuntimeError(f"no reference mp4s found in {reference_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    manager = ComponentsManager()
    pipe = MiniMaxH3Ref2VABlocks().init_pipeline("/cache/MiniMax-H3", components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")

    transformer = getattr(pipe, "transformer_ref", None) or getattr(pipe, "transformer", None)
    if transformer is None:
        raise RuntimeError("No transformer_ref/transformer component found on pipeline")
    transformer.add_adapter(
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["to_qkv", "to_out.0", "linear_1", "linear_2"],
            init_lora_weights="gaussian",
        )
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = transformer.load_state_dict(state["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected[:5]}")
    transformer.requires_grad_(False)
    try:
        transformer.set_attention_backend("_flash_3_hub")
    except Exception as exc:
        print(f"flash attention backend not enabled: {exc}", flush=True)

    items = []
    for index, ref in enumerate(refs):
        item_seed = seed + index
        out = out_root / f"{index:03d}_{ref.stem}_generated.mp4"
        print(f"generating eval sample {index + 1}/{len(refs)} from {ref.name}", flush=True)
        reference_path = ref
        if not include_reference_audio:
            reference_path = out_root / f"{ref.stem}_noaudio.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(ref), "-an", "-c:v", "copy", str(reference_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        reference = MiniMaxH3Reference(video=str(reference_path))
        generator = torch.Generator(device="cpu").manual_seed(item_seed)
        outputs = ["videos", "audio", "sampling_rate"] if generate_audio else ["videos"]
        results = pipe(
            prompt=prompt,
            references=[reference],
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output=outputs,
        )
        if generate_audio:
            encode_video(
                results["videos"][0],
                fps=24,
                output_path=str(out),
                audio=results["audio"][0],
                audio_sample_rate=results["sampling_rate"],
            )
        else:
            encode_video(results["videos"][0], fps=24, output_path=str(out))
        items.append(
            {
                "index": index,
                "checkpoint": str(ckpt),
                "reference_video": str(ref),
                "generated_video": str(out),
                "generated_bytes": out.stat().st_size,
                "prompt": prompt,
                "seed": item_seed,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "include_reference_audio": include_reference_audio,
                "generate_audio": generate_audio,
            }
        )

    result = {
        "output_dir": str(out_root),
        "items": items,
        "missing_key_count": len(missing),
    }
    (out_root / "eval_batch_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    return result


@app.local_entrypoint()
def generate_ref2va_eval_batch(
    checkpoint: str = "/runs/h3/ref2va-lora-smoke-100/checkpoint-100.pt",
    reference_dir: str = "/runs/samples/ta20_val_eval_clips",
    output_dir: str = "/runs/generated/ref2va-lora-smoke-100/eval_batch_001",
    limit: int = 5,
    prompt: str = "The same animated character talks directly to the camera with natural lip movement and a friendly expression.",
    num_frames: int = 124,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 6,
    seed: int = 100,
    include_reference_audio: bool = True,
    generate_audio: bool = True,
):
    print(
        json.dumps(
            _generate_ref2va_eval_batch.remote(
                checkpoint=checkpoint,
                reference_dir=reference_dir,
                output_dir=output_dir,
                limit=limit,
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                seed=seed,
                include_reference_audio=include_reference_audio,
                generate_audio=generate_audio,
            ),
            indent=2,
        )
    )


@app.function(image=image, volumes=VOLS, gpu="H100", cpu=16.0, memory=65536, timeout=8 * 60 * 60)
def _prepare_cache_smoke(
    manifest: str = "/runs/manifests/ta20_smoke/h3_train.jsonl",
    output: str = "/runs/cache/ta20_smoke_train_256_22",
    height: int = 256,
    width: int = 256,
    frames: int = 22,
    limit: int = 300,
) -> dict:
    trainer = Path("/runs/trainer/MiniMax-H3-FineTuning")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "prepare_cache.py",
        "--metadata",
        manifest,
        "--output",
        str(out),
        "--model",
        "/cache/MiniMax-H3",
        "--height",
        str(height),
        "--width",
        str(width),
        "--frames",
        str(frames),
        "--limit",
        str(limit),
        "--encode-text",
        "--encode-audio",
    ]
    _run(cmd, cwd=trainer)
    RUNS_VOL.commit()
    files = [p for p in out.rglob("*") if p.is_file()]
    return {"cache": str(out), "files": len(files), "bytes": sum(p.stat().st_size for p in files)}


@app.local_entrypoint()
def prepare_cache_smoke(
    manifest: str = "/runs/manifests/ta20_smoke/h3_train.jsonl",
    output: str = "/runs/cache/ta20_smoke_train_256_22",
    height: int = 256,
    width: int = 256,
    frames: int = 22,
    limit: int = 300,
):
    print(json.dumps(_prepare_cache_smoke.remote(manifest=manifest, output=output, height=height, width=width, frames=frames, limit=limit), indent=2))


@app.function(image=image, volumes=VOLS, gpu="H100", cpu=16.0, memory=65536, timeout=8 * 60 * 60)
def _train_lora_smoke(
    cache: str = "/runs/cache/ta20_smoke_train_256_22",
    output: str = "/runs/h3/ref2va-lora-smoke",
    max_steps: int = 100,
    lr: float = 1e-4,
    ckpt_every: int = 20,
    trainable: str = "lora",
    audio_loss_scale: float = 1.0,
) -> dict:
    trainer = Path("/runs/trainer/MiniMax-H3-FineTuning")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    train_py = trainer / "train.py"
    text = train_py.read_text(encoding="utf-8")
    old = "return loss_v + audio_weight * loss_a, loss_v.detach(), loss_a.detach(), audio_weight"
    new = (
        "audio_loss_scale = float(os.environ.get(\"H3_AUDIO_LOSS_SCALE\", \"1.0\"))\n"
        "    return loss_v + audio_loss_scale * audio_weight * loss_a, "
        "loss_v.detach(), loss_a.detach(), audio_loss_scale * audio_weight"
    )
    if old in text:
        train_py.write_text(text.replace(old, new), encoding="utf-8")
    cmd = [
        "python",
        "train.py",
        "--model",
        "/cache/MiniMax-H3",
        "--variant",
        "ref2va",
        "--cache",
        cache,
        "--output",
        str(out),
        "--max-steps",
        str(max_steps),
        "--trainable",
        trainable,
        "--strategy",
        "ddp",
        "--lr",
        str(lr),
        "--ckpt-every",
        str(ckpt_every),
        "--val-every",
        str(ckpt_every),
    ]
    os.environ["H3_AUDIO_LOSS_SCALE"] = str(audio_loss_scale)
    _run(cmd, cwd=trainer)
    RUNS_VOL.commit()
    files = [p for p in out.rglob("*") if p.is_file()]
    return {"output": str(out), "audio_loss_scale": audio_loss_scale, "files": len(files), "bytes": sum(p.stat().st_size for p in files)}


@app.local_entrypoint()
def train_lora_smoke(
    cache: str = "/runs/cache/ta20_smoke_train_256_22",
    output: str = "/runs/h3/ref2va-lora-smoke",
    max_steps: int = 100,
    lr: float = 1e-4,
    ckpt_every: int = 20,
    trainable: str = "lora",
    audio_loss_scale: float = 1.0,
):
    print(
        json.dumps(
            _train_lora_smoke.remote(
                cache=cache,
                output=output,
                max_steps=max_steps,
                lr=lr,
                ckpt_every=ckpt_every,
                trainable=trainable,
                audio_loss_scale=audio_loss_scale,
            ),
            indent=2,
        )
    )


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=8192, timeout=20 * 60)
def _trainer_help() -> dict:
    import subprocess

    trainer = Path("/runs/trainer/MiniMax-H3-FineTuning")
    if not trainer.is_dir():
        raise RuntimeError("trainer is missing; run train_smoke first")

    out: dict[str, str] = {}
    for rel in ("README.md", "examples/README.md", "requirements.txt"):
        path = trainer / rel
        out[rel] = path.read_text(encoding="utf-8", errors="replace")[:12000] if path.is_file() else ""
    for script in ("prepare_cache.py", "prepare_cache_pairs.py", "validate_manifest.py", "train.py"):
        path = trainer / script
        if not path.is_file():
            out[f"{script} --help"] = "missing"
            continue
        proc = subprocess.run(
            ["python", str(path), "--help"],
            cwd=str(trainer),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        out[f"{script} --help"] = proc.stdout[:12000]

    dest = Path("/runs/trainer_help.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    return out


@app.local_entrypoint()
def trainer_help():
    data = _trainer_help.remote()
    for key, value in data.items():
        print("\n" + "=" * 80)
        print(key)
        print("=" * 80)
        print(value)


@app.function(image=image, volumes=VOLS, cpu=4.0, memory=8192, timeout=30 * 60)
def _train_smoke(trainer_repo: str, trainer_ref: str = "main") -> dict:
    """Bootstrap the external H3 trainer and stop before heavy data/model work.

    This function creates the volume layout and clones the selected trainer.
    Actual training is deliberately guarded until dataset/model paths exist on
    Modal volumes, because H3 Ref2VA assets are large and expensive to move.
    """
    trainer_dir = Path("/runs/trainer/MiniMax-H3-FineTuning")
    if not trainer_dir.exists():
        trainer_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", trainer_repo, str(trainer_dir)])
    _run(["git", "fetch", "--all", "--tags"], cwd=trainer_dir)
    _run(["git", "checkout", trainer_ref], cwd=trainer_dir)
    status = {
        "trainer_dir": str(trainer_dir),
        "trainer_ref": trainer_ref,
        "model_dir_expected": "/cache/MiniMax-H3",
        "dataset_expected": "/data/TA2.0_animation_dataset",
        "next": [
            "download or mount MiniMax-H3 into /cache/MiniMax-H3",
            "download or mount TA2.0 into /data/TA2.0_animation_dataset",
            "run AQ subset preparation, then H3 cache/train",
        ],
    }
    (Path("/runs") / "train_smoke_bootstrap.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    RUNS_VOL.commit()
    return status


@app.local_entrypoint()
def train_smoke(
    trainer_repo: str = "https://github.com/IAmIronMan42/MiniMax-H3-FineTuning",
    trainer_ref: str = "main",
):
    print(json.dumps(_train_smoke.remote(trainer_repo=trainer_repo, trainer_ref=trainer_ref), indent=2))
