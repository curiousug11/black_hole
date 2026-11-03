# Explainer video

`black_hole_disks.mp4` is a ~6-minute narrated explainer of this project, in
the style of 3Blue1Brown. It is 1080p at 30 fps, with English subtitles
embedded as a soft track and also provided as `black_hole_disks.srt`.

* **Animation:** [Manim Community](https://www.manim.community/) v0.22.
* **Narration:** [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) v0.9.4,
  voice `af_heart`.

Every curve in the video is computed from the project itself
(`accretion_disk/` and the data in `data/observations/`); nothing is drawn by hand.

| # | Scene | Content |
|---|---|---|
| 1 | `S1_Intro` | Black holes are dark, yet their surroundings glow: X-rays vs ultraviolet |
| 2 | `S2_Temperature` | Thin disk, energy balance, T(r), T_max at 49/36 r_in |
| 3 | `S3_Spectrum` | Each ring is a blackbody; their sum gives ν², ν^(1/3) and a Wien cutoff |
| 4 | `S4_Scaling` | T_max ∝ (Ṁ/M²)^(1/4); a mass dial; the factor of 56 between X-ray and UV; accretion rate |
| 5 | `S5_Observer` | Inclination and distance; L_iso = 2 cos i L |
| 6 | `S6_Numerics` | agnpy + astropy + ring sum; N⁻² convergence |
| 7 | `S7_Observations` | SDSS composite, slopes of 10⁵ quasars, RXTE fits to LMC X-3 (f_col ≈ 1.8) |
| 8 | `S8_Conclusion` | Recap and what the model leaves out |

## Files

| File | Purpose |
|---|---|
| `narration.py` | The script: text of every beat (and phonetic spellings for the TTS) |
| `make_audio.py` | Kokoro TTS → `audio/<scene>_<beat>.wav` and `audio/durations.json` |
| `data_prep.py` | Computes all curves → `data/video_data.npz`, `data/facts.json` |
| `scenes.py` | Manim scenes; each narration beat drives its animations' timing |
| `build_video.py` | Renders all scenes, joins them, normalises loudness, writes the subtitles |

## Rebuilding

Kokoro 0.9.4 needs Python ≤ 3.12, so the narration is generated in a separate
virtual environment. The WAV files are not committed; regenerate them first.

```bash
python3.12 -m venv ttsenv
ttsenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
ttsenv/bin/pip install "kokoro>=0.9.4" soundfile        # also needs espeak-ng (apt)
ttsenv/bin/python video/make_audio.py                    # ~2 min on 4 CPU cores

pip install manim     # system deps: ffmpeg, libcairo2-dev, libpango1.0-dev, LaTeX (+ dvisvgm)
python video/build_video.py --quality high --jobs 4      # full HD render
python video/build_video.py --quality low                # fast 480p preview
```

To change the narration, edit `narration.py` and rerun `make_audio.py`; only
changed beats are re-synthesised. The scenes adapt to the new clip lengths
automatically. A different voice can be chosen with
`make_audio.py --voice <name>` (see the Kokoro voice list).
