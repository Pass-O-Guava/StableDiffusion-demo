# Stable Diffusion Python Demo

## 1. Install

### I. Create Project and downlod weights.
```shell
mkdir diffusers-demo
cd diffusers-demo

sudo apt-get install git-lfs
git init
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
## download: 66G

conda activate stablediffusion
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.19.2 diffusers==0.12.1 invisible-watermark

# -------- option -----------
pip install omegaconf==2.1.1
pip install einops==0.3.0
pip install pytorch-lightning==1.4.2
pip install torchmetrics==0.6
pip install open-clip-torch==2.7.0
pip install safetensors
# ---------------------------
```
### II. Files tree of "stable-diffusion-v1-5":
```shell
(stablediffusion) ubuntu@wilson:~/wy/AIGC/diffusers$ tree -h stable-diffusion-v1-5/
stable-diffusion-v1-5/
|-- [ 14K]  README.md
|-- [4.0K]  feature_extractor
|   `-- [ 342]  preprocessor_config.json
|-- [ 543]  model_index.json
|-- [4.0K]  safety_checker
|   |-- [4.6K]  config.j
|   |-- [1.1G]  model.safetensors
|   `-- [1.1G]  pytorch_model.bin
|-- [4.0K]  scheduler
|   `-- [ 308]  scheduler_config.json
|-- [4.0K]  text_encoder
|   |-- [ 617]  config.json
|   |-- [469M]  model.safetensors
|   `-- [469M]  pytorch_model.bin
|-- [4.0K]  tokenizer
|   |-- [512K]  merges.txt
|   |-- [ 472]  special_tokens_map.json
|   |-- [ 806]  tokenizer_config.json
|   `-- [1.0M]  vocab.json
|-- [4.0K]  unet
|   |-- [ 743]  config.json
|   |-- [3.2G]  diffusion_pytorch_model.bin
|   `-- [3.2G]  diffusion_pytorch_model.safetensors
|-- [4.0G]  v1-5-pruned-emaonly.ckpt
|-- [4.0G]  v1-5-pruned-emaonly.safetensors
|-- [7.2G]  v1-5-pruned.ckpt
|-- [7.2G]  v1-5-pruned.safetensors
|-- [1.8K]  v1-inference.yaml
`-- [4.0K]  vae
    |-- [ 547]  config.json
    |-- [319M]  diffusion_pytorch_model.bin
    `-- [319M]  diffusion_pytorch_model.safetensors
```

### III. Models:
|  Model  |  Name  |  File Size  |
|  ----   | ----   | ----        |
| text_encoder  | pytorch_model.bin | **469M** |
| unet  | diffusion_pytorch_model.bin | **3.2G** |
| vae  | diffusion_pytorch_model.bin | **319M** |
| safety_checker  | pytorch_model.bin | **1.1G** |
| use_train？？  | v1-5-pruned-emaonly.ckpt | 4.0G |
| use_train？？ | v1-5-pruned.ckpt | 7.2G |


## 2. Test: 
### I. txt2img

python txt2img.py

```shell
(stablediffusion) ubuntu@wilson:~/wy/AIGC/diffusers$ python txt2img.py 
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
    PyTorch 1.12.1+cu113 with CUDA 1105 (you have 1.12.1+cu113)
    Python  3.9.16 (you have 3.9.16)
  Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
  Memory-efficient attention, SwiGLU, sparse and more won't be available.
  Set XFORMERS_MORE_DETAILS=1 for more details
A matching Triton is not available, some optimizations will not be enabled.
Error caught was: No module named 'triton'
100%|███████████████████████████████████████████████████████████████████████████████████| 50/50 [00:17<00:00,  2.92it/s]
==> nsfw_content_detected: False
==> image saved.
```
![t2i](runs/astronaut_rides_horse_00.png)
![t2i](runs/astronaut_rides_horse_01.png)
![t2i](runs/astronaut_rides_horse_02.png)

### II. img2img

python img2img.py

## Plan

- [x] txt2img
- [ ] img2img
- [ ] inpaint
- [ ] others scheduler(default PLMS/DDIM/K-LMS/CycleDiffusion)
- [ ] others SD model


## Acknowledge
> https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion