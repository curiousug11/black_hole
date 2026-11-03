#!/usr/bin/env python3
"""Synthesise the narration with Kokoro TTS (hexgrad/Kokoro-82M).

Writes one WAV per beat to video/audio/<scene>_<beat>.wav and the durations to
video/audio/durations.json, which the Manim scenes use for timing.

    python video/make_audio.py [--voice af_heart] [--speed 1.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from narration import SCENES

HERE = Path(__file__).resolve().parent
AUDIO = HERE / "audio"
SAMPLE_RATE = 24_000          # Kokoro output rate
GAP = 0.12                    # silence (s) between sentence chunks inside one beat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")   # American English

    AUDIO.mkdir(exist_ok=True)
    cache_file = AUDIO / "cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    durations = {}
    for scene, beats in SCENES.items():
        durations[scene] = []
        for i, beat in enumerate(beats):
            say = beat.get("say", beat["text"])
            key = hashlib.sha1(f"{args.voice}|{args.speed}|{say}".encode()).hexdigest()
            path = AUDIO / f"{scene}_{i}.wav"
            if not (path.exists() and cache.get(path.name) == key):
                chunks = []
                for _, _, audio in pipe(say, voice=args.voice, speed=args.speed,
                                        split_pattern=r"\n+"):
                    chunks.append(np.asarray(audio, dtype=np.float32))
                    chunks.append(np.zeros(int(GAP * SAMPLE_RATE), dtype=np.float32))
                wav = np.concatenate(chunks)
                sf.write(path, wav, SAMPLE_RATE)
                cache[path.name] = key
                print(f"  synthesised {path.name}")
            dur = sf.info(path).duration
            durations[scene].append(round(dur, 3))
        print(f"{scene}: {len(beats)} beats, {sum(durations[scene]):.1f} s")
    (AUDIO / "durations.json").write_text(json.dumps(durations, indent=1))
    cache_file.write_text(json.dumps(cache, indent=1))
    total = sum(sum(v) for v in durations.values())
    print(f"total narration: {total:.1f} s ({total / 60:.1f} min), voice {args.voice}")


if __name__ == "__main__":
    main()
