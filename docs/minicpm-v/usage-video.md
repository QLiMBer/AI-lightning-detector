Video Inference — MiniCPM‑V 4.5

Why frames (via decord)?

- The MiniCPM‑V Transformers chat API accepts sequences of images plus text. For videos, upstream examples construct a list of frames (PIL Images) and optionally provide `temporal_ids` for the 3D‑Resampler. Therefore, a lightweight video reader (e.g., `decord`) is used to decode `.mp4` into frames before calling `model.chat`.
- References: main repo “Chat with Video” example (3D‑Resampler with `temporal_ids`) and Cookbook “Video Understanding” recipe (simple frame sampling).

Baseline video QA (simple frame sampling)

```python
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
from decord import VideoReader, cpu

model_id = 'openbmb/MiniCPM-V-4_5'
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation='sdpa',
    dtype=torch.bfloat16,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

MAX_NUM_FRAMES = 64  # reduce if you hit CUDA OOM

def uniform_sample(seq, n):
    gap = len(seq) / n
    idx = [int(i * gap + gap / 2) for i in range(n)]
    return [seq[i] for i in idx]

def encode_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = max(1, round(vr.get_avg_fps()))
    frame_idx = list(range(0, len(vr), sample_fps))
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    return frames

video_path = 'video.mp4'
frames = encode_video(video_path)
question = 'Describe the video briefly.'
msgs = [{ 'role': 'user', 'content': frames + [question] }]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    use_image_id=False,
    max_slice_nums=1,
)
print(res)
```

High‑FPS/long video with 3D‑Resampler (temporal packing)

```python
import math
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from scipy.spatial import cKDTree

TIME_SCALE = 0.1  # seconds
MAX_NUM_FRAMES = 180  # frames after packing
MAX_NUM_PACKING = 3    # up to 6 supported; adjust per VRAM/speed

def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]

def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]

def encode_video_with_temporal_ids(video_path, choose_fps=5, force_packing=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    duration = len(vr) / fps

    if choose_fps * int(duration) <= MAX_NUM_FRAMES:
        packing = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, duration))
    else:
        packing = math.ceil(duration * choose_fps / MAX_NUM_FRAMES)
        if packing <= MAX_NUM_PACKING:
            choose_frames = round(duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing = MAX_NUM_PACKING

    idx = np.array(list(range(len(vr))))
    # uniform sample to choose_frames
    gap = len(idx) / max(1, choose_frames)
    idx = np.array([int(i * gap + gap/2) for i in range(max(1, choose_frames))])

    if force_packing is not None:
        packing = min(force_packing, MAX_NUM_PACKING)

    frames_np = vr.get_batch(idx).asnumpy()
    frame_ts = idx / fps
    scale = np.arange(0, duration, TIME_SCALE)
    ts_ids = map_to_nearest_scale(frame_ts, scale) / TIME_SCALE
    ts_ids = ts_ids.astype(np.int32)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames_np]
    temporal_ids = group_array(ts_ids, packing)
    return frames, temporal_ids

frames, temporal_ids = encode_video_with_temporal_ids('video.mp4', choose_fps=5)
msgs = [{ 'role': 'user', 'content': frames + ['Describe the video briefly.'] }]

res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    use_image_id=False,
    max_slice_nums=1,
    temporal_ids=temporal_ids,
)
print(res)
```

Notes

- Use `use_image_id=False` for repeated frames or multi‑frame inputs.
- Increase `max_slice_nums` for high‑resolution frames if you hit VRAM limits.
- Temporal packing improves throughput for long/high‑FPS videos; tune `choose_fps`, packing cap, and MAX_NUM_FRAMES per VRAM/time.

Parameters seen in upstream examples

- `enable_thinking`: toggles longer “deep thinking” mode when set True.
- `use_image_id=False`: recommended when many frames are used.
- `max_slice_nums`: split high‑res frames to avoid OOM (e.g., 1–2).
- `temporal_ids`: list of integer ID groups enabling the 3D‑Resampler to pack consecutive frames.

Slow‑motion videos

- Phone slo‑mo clips often have high native FPS. Our encoder functions downsample to a fixed target FPS (e.g., 5) for consistent coverage and stable memory use while keeping timestamps accurate via `frame_index / native_fps`.
