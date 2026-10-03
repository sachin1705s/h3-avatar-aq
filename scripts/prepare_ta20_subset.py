#!/usr/bin/env python3
"""Prepare an identity-split TA2.0 subset manifest for H3 Ref2VA LoRA.

Input can be a TA2.0 metadata Parquet/JSONL/CSV containing at least clip_id,
ref_id, and a video path column. The script keeps ref_id groups intact across
splits and emits both AQ metadata rows and H3 trainer manifest rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


VIDEO_KEYS = ("video_path", "file_name", "video", "path", "file", "mp4")
REF_KEYS = ("reference_path", "ref_path", "image_path", "ref_image", "reference_image")
AUDIO_KEYS = ("target_audio", "audio_path", "audio", "wav", "speech_audio")
CAPTION_KEYS = ("caption", "text", "prompt", "script", "transcript")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise SystemExit("Parquet input needs pandas/pyarrow installed.") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def first(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value) and str(value).lower() != "nan":
            return str(value)
    return ""


def resolve_path(root: Path, value: str) -> str:
    if not value:
        return ""
    p = Path(value)
    return str(p if p.is_absolute() else (root / p).resolve())


def normalize(row: dict[str, Any], root: Path) -> dict[str, Any]:
    video = first(row, VIDEO_KEYS)
    ref = first(row, REF_KEYS)
    audio = first(row, AUDIO_KEYS)
    motion = str(row.get("motion_mode") or "unknown")
    language = str(row.get("language") or row.get("lang") or "unknown")
    caption = first(row, CAPTION_KEYS) or (
        f"A single animated character talks to camera in {language}. "
        f"Motion mode: {motion}."
    )
    quality = row.get("vqa_score") or row.get("quality_score") or row.get("score") or 1.0
    clip_id = str(row.get("clip_id") or row.get("id") or Path(video).stem)
    ref_id = str(row.get("ref_id") or row.get("character_id") or "unknown")
    video_path = resolve_path(root, video)
    ref_path = resolve_path(root, ref)
    audio_path = resolve_path(root, audio)
    return {
        "clip_id": clip_id,
        "ref_id": ref_id,
        "video_path": video_path,
        "reference_path": ref_path,
        "audio_path": audio_path,
        "caption": caption,
        "motion_mode": motion,
        "language": language,
        "duration_sec": row.get("duration_sec") or row.get("duration") or "",
        "frame_num": row.get("frame_num") or row.get("frames") or "",
        "quality_score": float(quality),
    }


def to_h3_manifest(row: dict[str, Any]) -> dict[str, Any]:
    refs = [row["reference_path"]] if row.get("reference_path") else []
    audios = [row["audio_path"]] if row.get("audio_path") else []
    caption = str(row.get("caption") or "")
    if refs and "<Picture 1>" not in caption:
        caption = "<Picture 1> " + caption
    return {
        "id": str(row["clip_id"]),
        "caption": caption,
        "video": row["video_path"],
        "target_video": row["video_path"],
        "target_audio": row.get("audio_path") or None,
        "reference_images": refs,
        "reference_videos": [],
        "reference_audios": audios,
        "metadata": {
            "ref_id": row.get("ref_id"),
            "motion_mode": row.get("motion_mode"),
            "language": row.get("language"),
            "quality_score": row.get("quality_score"),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default="data/metadata", type=Path)
    ap.add_argument("--limit", default=300, type=int)
    ap.add_argument("--seed", default=7, type=int)
    ap.add_argument("--val-frac", default=0.10, type=float)
    ap.add_argument("--test-frac", default=0.10, type=float)
    ap.add_argument("--require-files", action="store_true", help="Keep only rows whose target video exists on disk.")
    args = ap.parse_args()

    rows = [normalize(r, args.root) for r in read_rows(args.metadata)]
    rows = [r for r in rows if r["video_path"] and r["ref_id"]]
    source_rows = len(rows)
    if args.require_files:
        rows = [r for r in rows if Path(r["video_path"]).is_file()]
    if not rows:
        raise SystemExit("No usable rows after filtering metadata.")
    rng = random.Random(args.seed)
    by_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ref[row["ref_id"]].append(row)

    refs = list(by_ref)
    rng.shuffle(refs)
    selected: list[dict[str, Any]] = []
    selected_refs: list[str] = []
    for ref_id in refs:
        if len(selected) >= args.limit:
            break
        selected_refs.append(ref_id)
        selected.extend(by_ref[ref_id])
    selected = selected[: args.limit]

    n_ref = len(selected_refs)
    n_test = max(1, int(n_ref * args.test_frac)) if n_ref >= 3 else 0
    n_val = max(1, int(n_ref * args.val_frac)) if n_ref >= 3 else 0
    test_refs = set(selected_refs[:n_test])
    val_refs = set(selected_refs[n_test : n_test + n_val])

    train_rows = [r for r in selected if r["ref_id"] not in test_refs and r["ref_id"] not in val_refs]
    val_rows = [r for r in selected if r["ref_id"] in val_refs]
    test_rows = [r for r in selected if r["ref_id"] in test_refs]

    write_jsonl(args.out / "ta20_smoke_manifest.jsonl", selected)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "val.jsonl", val_rows)
    write_jsonl(args.out / "test.jsonl", test_rows)
    write_jsonl(args.out / "h3_train.jsonl", [to_h3_manifest(r) for r in train_rows])
    write_jsonl(args.out / "h3_val.jsonl", [to_h3_manifest(r) for r in val_rows])
    write_jsonl(args.out / "h3_test.jsonl", [to_h3_manifest(r) for r in test_rows])
    print(json.dumps({
        "total": len(selected),
        "source_rows": source_rows,
        "usable_rows": len(rows),
        "require_files": args.require_files,
        "refs": len(selected_refs),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "h3_train": str(args.out / "h3_train.jsonl"),
        "h3_val": str(args.out / "h3_val.jsonl"),
        "h3_test": str(args.out / "h3_test.jsonl"),
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
