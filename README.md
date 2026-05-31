# Adapting Keras MobiletNetV2 Tensors to PyTorch Implementation

Meant to be a pure implementation of MobileNetV2 with 0.35, 0.5 and 1.0 alpha variants and ImageNet pre-trained weights from the original Keras project.

Point-based detection is becoming highly relevant for the industry's shift toward Embodied AI and agentic workflows. In robotics, GUI interaction, and document understanding, predicting a central point or click-target is often more practical and contextually accurate than predicting a full bounding box.

This repository contains a LibreFOMO training example (MobileNetV2 backbone) for a people detection task on [LOAF](https://loafisheye.github.io/) for overhead people detection.

```
# pip install torch huggingface_hub

from huggingface_hub import hf_hub_download
import importlib.util
import torch

repo_id = "bdanko/LibreFOMO"

# Download model.py and checkpoint from HF
model_py = hf_hub_download(repo_id=repo_id, filename="model.py")
ckpt_path = hf_hub_download(repo_id=repo_id, filename="LibreFOMOm.pt")

# Import the downloaded model.py
spec = importlib.util.spec_from_file_location("librefomo_model", model_py)
librefomo_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(librefomo_model)

# Load checkpoint
model, ckpt = librefomo_model.load_librefomo_checkpoint(ckpt_path, map_location="cuda:0")

# Smoke-test inference
x = torch.zeros(1, 3, ckpt["imgsz"], ckpt["imgsz"])
with torch.no_grad():
    y = model(x)

print(y.shape)
print(ckpt["size"], ckpt["imgsz"], ckpt["names"])
```

Temporarily hosted all `.pt` PyTorch tensor files at [https://huggingface.co/bdanko/LibreFOMO](https://huggingface.co/bdanko/LibreFOMO)
