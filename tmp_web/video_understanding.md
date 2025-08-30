# Video Understanding

### Initialize model

```python
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch

model_path = 'openbmb/MiniCPM-V-4_5'
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation='sdpa',  # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True)
```

### Video encode function

```python
from decord import VideoReader, cpu

MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number


def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames
```

### Chat with video

```python
video_path = "assets/badminton.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msg = [
    {'role': 'user', 'content': frames + [question]},
]

# Set decode params for video
params = {}
params["use_image_id"] = False
# use 1 if cuda OOM and video resolution > 448*448
params["max_slice_nums"] = 2

res = model.chat(
    image=None,
    msgs=msg,
    tokenizer=tokenizer,
    **params
)

print("Response:")
print(res)
```

### Example Output

```
num frames: 21
Response:
The video captures a dynamic badminton match between two skilled players, one in yellow and the other in red. The game takes place on an indoor court with green flooring and white lines marking its boundaries. Throughout the footage, we see both athletes displaying impressive footwork as they move swiftly across the playing area to return shots.

At various points, the player wearing yellow is seen executing powerful smashes from different locations around the court, including near the net where he successfully hits his opponent out of bounds twice. In response, the competitor dressed in red demonstrates agility by diving repeatedly onto the floor after failing to retrieve these low shots made close to the ground or driven straight towards him at high speed.

As their rally progresses, each participant showcases quick reflexes and strategic positioning during exchanges that span diagonally across multiple corners while maintaining control over ball placement along sidelines marked by distinct pink borders surrounding them. Their movements are fluid yet intense – reflecting competitive spirit typical for professional sports events like this one which appears staged indoors under artificial lighting conditions visible through overhead fixtures illuminating entire venue space filled with spectators seated behind protective glass panels separating audience members safely away from actual play zone itself (which could be either clay or hard surface based on visual cues).
```
