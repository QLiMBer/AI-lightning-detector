Lightning Detector — Implementation Plan (MiniCPM‑V 4.5)

Goals

- Detect lightning flashes in `.mp4` videos under a given directory, running locally on RTX 3090.
- Output both a human‑readable report and structured JSON for downstream use.
- Favor recall with confidences so results can be filtered/sorted.

Architecture Outline

- Input layer: enumerate `videos/` (or a configurable input directory); filter `.mp4` files.
- Video sampler: decode frames via `decord`; configurable FPS cap and max frames per clip; optional 3D‑Resampler temporal packing for throughput.
- Prompting: send frames plus an instruction to MiniCPM‑V 4.5 to identify lightning and return JSON with time ranges and confidences.
- Heuristic prefilter: deferred. We will start without it and evaluate pure‑model performance first.
- Aggregation: merge overlapping detections; compute per‑video summary stats.
- Outputs: per‑video `.json` with detailed detections and `.txt` summary; top‑level index summary.

Data Flow

1) Load model (`openbmb/MiniCPM-V-4_5`) in BF16 on GPU with `transformers==4.44.2`.
2) For each `.mp4`:
   - Compute basic metadata (duration, fps, resolution) via `decord`.
   - Choose sampling strategy:
     - Baseline: uniform frames up to a cap (simple approach for short videos).
     - Long/high‑FPS: temporal packing with 3D‑Resampler (provide `temporal_ids`).
   - (Skip for now) Brightness‑spike prefilter; can be added later if needed.
   - Prompt MiniCPM‑V 4.5:
     - Instruct strict JSON output schema (example below) and define “lightning flash” vs non‑lightning (camera flash, fireworks, etc.).
     - Provide either: whole‑video sampled frames, or per‑window frames if using heuristic prefilter.
   - Parse JSON; attach model sampling parameters and context.
3) Save `reports/<video_basename>.json` and `reports/<video_basename>.txt`.

Proposed JSON Schema (v1)

```json
{
  "video": "path/to/file.mp4",
  "fps_sampled": 5,
  "method": "temporal_packing|uniform",
  "model": "openbmb/MiniCPM-V-4_5",
  "detections": [
    {"start_sec": 12.4, "end_sec": 12.9, "confidence": 0.82, "notes": "bright flash in storm cloud"}
  ],
  "non_lightning_events": [
    {"start_sec": 45.0, "end_sec": 45.4, "confidence": 0.51, "label": "camera_flash"}
  ]
}
```

Prompt Template (initial)

- System style: “You analyze video frames and identify lightning flashes. A lightning flash is a natural atmospheric electrical discharge; exclude camera flashes, fireworks, headlights, reflections, sudden exposure changes.”
- User task: “Return only JSON with a list of lightning events with `start_sec`, `end_sec`, and `confidence` in [0,1]. If uncertain, still include low‑confidence events. Optionally include `non_lightning_events` for likely false positives.”
- Provide: frames + a parallel list of per‑frame timestamps so the model can reason about time windows; or let the program post‑map frame indices to seconds.

Performance & Throughput

- Default sampling: 5 FPS; cap packed frames to avoid OOM; adjust `max_slice_nums` for hi‑res frames.
- Use temporal packing for long videos (e.g., pack 3–6 consecutive frames) to boost coverage.
- Batch per video serially first; later we can parallelize across videos if VRAM allows.

Video Characteristics (from your samples)

- Typical resolution 1080p (1920×1080), ~30 FPS; some recordings in slow‑motion modes (higher FPS). Our sampler will downsample to a fixed `choose_fps` (e.g., 5) so slow‑motion clips are handled consistently. Duration and timestamps are computed from `frame_index / native_fps` to maintain accurate time mapping.

Validation Plan

- Manual verification on a handful of known‑lightning clips; compare detected windows to ground truth eyeballing.
- Iterate on prompts to balance recall vs noise.
- Tune brightness‑spike heuristic threshold to expand/contract candidate windows.

Deliverables (Docs‑first stage)

- Setup and usage docs (added).
- This plan and open questions.
- No code implementation yet (pending your review).

Folders (initial defaults)

- Input videos: `videos/`
- Outputs: `reports/` (JSON + text summaries)

Next Step (if approved)

- Scaffold CLI tool:
  - `lightning-detector scan --input videos/ --fps 5 --packing 3 --json reports/ --txt reports/`
  - Implement baseline sampler + JSON prompt + parser.
  - Add optional heuristic prefilter flag.
