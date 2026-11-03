#!/usr/bin/env python3
"""Render all scenes, join them, and add subtitles.

Steps (run from anywhere):
    1. narration audio:  <python3.12 venv>/bin/python video/make_audio.py
    2. this script:      python video/build_video.py [--quality high|low] [--jobs 3]

It runs data_prep.py, renders every scene with Manim, concatenates them into
video/black_hole_disks.mp4 (H.264 + AAC, loudness-normalised) and writes
video/black_hole_disks.srt from the narration text and the beat timings
recorded during rendering.  The subtitles are also embedded as a soft track.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from narration import SCENES, SCENE_ORDER  # noqa: E402

BUILD = HERE / "build"
OUT_MP4 = HERE / "black_hole_disks.mp4"
OUT_SRT = HERE / "black_hole_disks.srt"


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def render(scene, quality):
    res, fps, sub = (("1920,1080", 30, "1080p30") if quality == "high"
                     else ("854,480", 15, "480p15"))
    media = BUILD / f"media_{quality}" / scene      # per-scene dir: no LaTeX-cache races
    out = media / "videos" / "scenes" / sub / f"{scene}.mp4"
    inputs = [HERE / "scenes.py", HERE / "audio" / "durations.json", HERE / "data" / "video_data.npz"]
    if out.exists() and (BUILD / f"beats_{scene}.json").exists() and \
            out.stat().st_mtime > max(p.stat().st_mtime for p in inputs):
        print(f"  {scene}: up to date", flush=True)
        return out
    log = BUILD / f"render_{scene}_{quality}.log"
    with open(log, "w") as fh:
        subprocess.run(["manim", "render", "--resolution", res, "--frame_rate", str(fps),
                        "--disable_caching", "--media_dir", str(media), str(HERE / "scenes.py"),
                        scene], check=True, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    return out


def srt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def write_srt(offsets):
    entries = []
    for scene, offset in zip(SCENE_ORDER, offsets):
        beats = json.loads((BUILD / f"beats_{scene}.json").read_text())
        for beat, start, dur in zip(SCENES[scene], beats["starts"], beats["durations"]):
            # split the beat into caption chunks of at most two 42-character lines
            lines = textwrap.wrap(beat["text"], 42)
            chunks = [" \n".join(lines[i:i + 2]) for i in range(0, len(lines), 2)]
            weights = [len(c) for c in chunks]
            t = offset + start
            for c, w in zip(chunks, weights):
                dt = dur * w / sum(weights)
                entries.append((t, t + dt, c.replace(" \n", "\n")))
                t += dt
    with open(OUT_SRT, "w") as fh:
        for i, (a, b, text) in enumerate(entries, 1):
            fh.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")
    print(f"wrote {OUT_SRT.relative_to(ROOT)} ({len(entries)} captions)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", choices=["high", "low"], default="high")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    BUILD.mkdir(exist_ok=True)

    if not (HERE / "data" / "video_data.npz").exists():
        run([sys.executable, str(HERE / "data_prep.py")])
    with ThreadPoolExecutor(args.jobs) as pool:
        files = list(pool.map(lambda s: render(s, args.quality), SCENE_ORDER))
    for f in files:
        print(f"  {f.relative_to(ROOT)}  {duration(f):.1f} s")

    concat = BUILD / "concat.txt"
    concat.write_text("".join(f"file '{f}'\n" for f in files))
    offsets, t = [], 0.0
    for f in files:
        offsets.append(t)
        t += duration(f)
    write_srt(offsets)

    joined = BUILD / "joined.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "160k",
         "-ar", "48000", str(joined)])
    run(["ffmpeg", "-v", "error", "-y", "-i", str(joined), "-i", str(OUT_SRT),
         "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text",
         "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", str(OUT_MP4)])
    print(f"wrote {OUT_MP4.relative_to(ROOT)}: {duration(OUT_MP4):.1f} s, "
          f"{OUT_MP4.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
