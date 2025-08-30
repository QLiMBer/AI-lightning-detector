# Multi Images

### Initialize model

```python
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

model_path = 'openbmb/MiniCPM-V-4_5'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                  attn_implementation='sdpa', torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True)
```

### Chat with multiple images

```python
image1 = Image.open('assets/multi1.png').convert('RGB')
image2 = Image.open('assets/multi2.png').convert('RGB')
question = 'Compare the two images, tell me about the differences between them.'

msgs = [{'role': 'user', 'content': [image1, image2, question]}]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
```

### Sample Images

![alt text](./assets/multi1.png)

![alt text](./assets/multi2.png)

### Example Output

```
The vases have different shapes, with the first being rounder and more bulbous. The patterns on the vases are also distinct: the first vase has red designs against a white background, while the second features green and blue floral motifs. Additionally, the neck of the first vase is narrower than that of the second one.
```
