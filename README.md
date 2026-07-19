# ViTEraser (AAAI 2024)

The official implementation of [ViTEraser: Harnessing the Power of Vision Transformers for Scene Text Removal with SegMIM Pretraining](https://arxiv.org/abs/2306.12106) (AAAI 2024). 
The ViTEraser revisits the conventional single-step one-stage framework and improves it with ViTs for feature modeling and the proposed SegMIM pretraining. 
Below are the frameworks of ViTEraser and SegMIM.

![ViTEraser](figures/viteraser.svg)
![SegMIM](figures/segmim.svg)

## Todo List
- [x] Inference code and model weights 
- [x] ViTEraser training code 
- [x] SegMIM pre-training code

## Environment
We recommend using [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies. Install uv, then run:

```bash
git clone https://github.com/shannanyinxiang/ViTEraser.git
cd ViTEraser
uv sync
```

`uv sync` installs the Python version pinned in `.python-version`, creates a local `.venv`, and installs the CUDA 12.8 build of PyTorch together with the other dependencies. This build supports NVIDIA Blackwell GPUs such as the GeForce RTX 50 series. Prefix Python commands with `uv run` to use this environment; activating the virtual environment is not required.

## Datasets 

### 1. Text Removal Dataset
- SCUT-EnsText [[paper]](https://ieeexplore.ieee.org/document/9180003): 

  1. Download the training and testing sets of SCUT-EnsText at [link](https://github.com/HCIILAB/SCUT-EnsText).
  2. Rename `all_images` and `all_labels` folders to `image` and `label`, respectively.
  3. Generate text masks: 
  ```
    # Generating masks for the training set of SCUT-EnsText
    uv run python tools/generate_mask.py \
      --data_root data/TextErase/SCUT-EnsText/train    

    # Generating masks for the testing set of SCUT-EnsText
    # Masks are not used for inference. Just keep the same data structure as the training stage.
    uv run python tools/generate_mask.py \
      --data_root data/TextErase/SCUT-EnsText/test
  ```

### 2. SegMIM Pretraining Datasets 
(optional, only required by SegMIM pretraining)
- ICDAR2013 [[paper](https://ieeexplore.ieee.org/document/6628859)][[download link](https://pan.baidu.com/s/1QcR1yNNIqrWgvigk3UwEJg?pwd=52td)]
- ICDAR2015 [[paper](https://ieeexplore.ieee.org/document/7333942)][[download link](https://pan.baidu.com/s/11mxtWL-bnO0qwEi5aE0OyA?pwd=rqrg)]
- MLT2017 [[paper](https://ieeexplore.ieee.org/document/8270168)][[download link](https://pan.baidu.com/s/1DRazAnO-bRR46ybWzDLKOA?pwd=42hv)]
- ArT [[paper](https://arxiv.org/abs/1909.07741)][[download link](https://pan.baidu.com/s/13cn5zw4vI57ET83ouHPh4Q?pwd=rzwr)]
- LSVT [[paper](https://arxiv.org/abs/1909.07741)][[download link](https://pan.baidu.com/s/1NO1q7KVgFn9CrKsUH5gzzA?pwd=6jvr)]
- ReCTS [[paper](https://arxiv.org/abs/1909.07741)][[download link](https://pan.baidu.com/s/11c_VyEXD1YILA6FxODLElQ?pwd=ssmb)]
- TextOCR [[paper](https://arxiv.org/abs/1909.07741)][[download link](https://pan.baidu.com/s/1-IX6H8wppsMDhI74JhvU5Q?pwd=q5f2)]

Please prepare the above datasets into the `data` folder following the file structure below.

```
data
├─TextErase
│  └─SCUT-EnsText
│     ├─train
│     │  ├─image
│     │  ├─label
│     │  └─mask
│     └─test
│        ├─image
│        ├─label
│        └─mask
└─SegMIMDatasets
   ├─ArT
   ├─ICDAR2013
   ├─ICDAR2015
   ├─LSVT
   ├─MLT2017
   ├─ReCTS
   └─TextOCR
```

## Models

The download links of pre-trained ViTEraser weights are provided in the following table.

| Name | BaiduNetDisk | GoogleDrive|
| -    |  -   |   -   |
| ViTEraser-Tiny | [link](https://pan.baidu.com/s/1EOFRUXh87vm7MpxBlRqgeg?pwd=3evn) | [link](https://drive.google.com/file/d/1f6Awu37YD7A4VC8gIvZHmLtk5wqJlSQi/view?usp=drive_link) | 
| ViTEraser-Small | [link](https://pan.baidu.com/s/1ze-B8rYDYOhZ9zHAp77N3A?pwd=47mr) | [link](https://drive.google.com/file/d/1JDaallum-Z1iZ8GULimz4OaQjfVRKP5i/view?usp=drive_link) | 
| ViTEraser-Base | [link](https://pan.baidu.com/s/1G26NsjI_pcUWKdOqjMon0w?pwd=qurn) | [link](https://drive.google.com/file/d/1nvIN_HAR1LqIbmkSlWmtHIJjj3B9YEj4/view?usp=sharing) |

## Inference

Download a pretrained checkpoint from the [Models](#models) section, then remove text from a single image with:

```bash
uv run python inference.py \
    --input path/to/input.jpg \
    --checkpoint path/to/viteraser_tiny.pth \
    --output output/result.png \
    --model tiny
```

To process every supported image in a directory recursively, pass directories instead:

```bash
uv run python inference.py \
    --input path/to/input_images/ \
    --checkpoint path/to/viteraser_tiny.pth \
    --output output/results/ \
    --model tiny
```

The output is resized back to each image's original resolution. Use `--model small` or `--model base` with the corresponding checkpoint.

### Gradio WebUI

Launch the local WebUI with:

```bash
uv run python webui.py
```

Then open <http://127.0.0.1:7860>. High-quality tiled inference is enabled by default: it processes overlapping 512-pixel tiles without resizing the whole image, blends tile boundaries, and preserves pixels outside changed regions. The first inference loads the checkpoint into GPU memory; subsequent runs reuse the loaded model. To access it from another machine on the network, run `uv run python webui.py --host 0.0.0.0`.

The command-line interface uses the same high-quality tiled mode by default. Add `--fast` to use the previous whole-image resize mode when speed matters more than preserving fine details.

Argument changes for different scales of ViTEraser are as below:

| Argument | Tiny | Small | Base |
| - | - | - | - |
| swin_enc_embed_dim | 96 | 96 | 128 |
| swin_enc_depths | 2 2 6 2 | 2 2 18 2 | 2 2 18 2 |
| swin_enc_num_heads | 3 6 12 24 | 3 6 12 24 | 4 8 16 32 |
| swin_enc_window_size | 16 | 16 | 8 |
| swin_dec_depths | 2 6 2 2 2 | 2 18 2 2 2 | 2 18 2 2 2 |
| swin_dec_num_heads | 24 12 6 3 2 | 24 12 6 3 2 | 32 16 8 4 2 |
| swin_dec_window_size | 16 | 8 | 8 |

## Evaluation

The command for calculating metrics is:
```
uv run python eval/evaluation.py \
    --gt_path data/TextErase/SCUT-EnsText/test/label/ \
    --target_path path/to/model/output/

uv run python -m pytorch_fid \
    data/TextErase/SCUT-EnsText/test/label/ \
    path/to/model/output/ \
    --device cuda:0
```

## ViTEraser Training

### 1. Training without SegMIM pretraining

- Download the ImageNet-pretrained weights of Swin Transformer V2 (Tiny: [download link](https://pan.baidu.com/s/19v-qKJO4c0iK52y7Lx1Qgg?pwd=j8yj), Small: [download link](https://pan.baidu.com/s/1kLAA27KqPlTEZnLkxTjC2w?pwd=8rm6), Base: [download link](https://pan.baidu.com/s/1_UO_MGN-O4pXsekBP_YPxg?pwd=75bf), originally released at [repo](https://github.com/microsoft/Swin-Transformer)).
- Download the ImageNet-pretrained weights of VGG-16 ([download link](https://pan.baidu.com/s/13dS0Q55ydoF6zdGKxS1lkg?pwd=5scx), originally released by PyTorch).
- Put the pretrained weights into the `pretrained` folder.
- Run the example scripts in the `scripts/viteraser-training-wosegmim` folder.
For instance, run the following command to train ViTEraser-Tiny without SegMIM pretraining.
```
uv run bash scripts/viteraser-training-wosegmim/viteraser-tiny-train.sh
```

### 2. Training with SegMIM pretraining

- Download the SegMIM pretraining weights for ViTEraser-Tiny ([download link](https://pan.baidu.com/s/1lqhWgpmrnxHbk1USRpSGtw?pwd=xr6a)), ViTEraser-Small ([download link](https://pan.baidu.com/s/16TcTOdwPAZnmUgk_SUR7Ag?pwd=i6zr)), or ViTEraser-Base ([download link](https://pan.baidu.com/s/1HGlb1xAfKykS8wp3FPwSIQ?pwd=frdq)).
- Download the ImageNet-pretrained weights of VGG-16 ([download link](https://pan.baidu.com/s/13dS0Q55ydoF6zdGKxS1lkg?pwd=5scx), originally released by PyTorch).
- Put the pretrained weights into the `pretrained` folder.
- Run the example scripts in the `scripts/viteraser-training-withsegmim` folder.
For instance, run the following command to train ViTEraser-Tiny with SegMIM pretraining.
```
uv run bash scripts/viteraser-training-withsegmim/viteraser-tiny-train-withsegmim.sh
```

## SegMIM Pretraining
- Download the ImageNet-pretrained weights of Swin Transformer V2 (Tiny: [download link](https://pan.baidu.com/s/19v-qKJO4c0iK52y7Lx1Qgg?pwd=j8yj), Small: [download link](https://pan.baidu.com/s/1kLAA27KqPlTEZnLkxTjC2w?pwd=8rm6), Base: [download link](https://pan.baidu.com/s/1_UO_MGN-O4pXsekBP_YPxg?pwd=75bf), originally released at [repo](https://github.com/microsoft/Swin-Transformer)) into the `pretrained` folder.
- Run the example scripts in the `scripts/segmim` folder.
For instance, run the following command to perform SegMIM pretraining of ViTEraser-Tiny.
```
# end-to-end encoder-decoder pretraining
uv run bash scripts/segmim/viteraser-tiny-segmim.sh

# standalone encoder finetuning
uv run bash scripts/segmim/viteraser-tiny-encoder-finetune.sh
```

## Citation
```
@inproceedings{peng2024viteraser,
  title={ViTEraser: Harnessing the power of vision transformers for scene text removal with SegMIM pretraining},
  author={Peng, Dezhi and Liu, Chongyu and Liu, Yuliang and Jin, Lianwen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4468--4477},
  year={2024}
}
```

## Copyright
This repository can only be used for non-commercial research purpose.

For commercial use, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).

Copyright 2024, [Deep Learning and Vision Computing Lab](http://www.dlvc-lab.net), South China University of Technology. 
