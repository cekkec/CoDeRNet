# CoDeRNet

## Installation

```bash
conda create -n codernet python=3.12 -y
conda activate codernet
pip install -r requirements.txt
# The METEOR scorer in eval_func/ relies on Java; install a JDK if not present.
```


## Datasets

Download links (used in our experiments):

- **LEVIR-MCI**: https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main
- **WHU-CDC** : https://www.kaggle.com/datasets/yuehaozhang1109/whu-cdc/data

Both datasets are passed to the scripts via `--data_folder` / `--list_path` / `--caption_json`. The expected layouts:

### LEVIR-MCI (3-class change segmentation + captioning)

```
<LEVIR_MCI_ROOT>/
├── images/
│   ├── train/{A, B, label}/*.png
│   ├── val/{A, B, label}/*.png
│   └── test/{A, B, label}/*.png
└── <list_path>/
    ├── train.txt, val.txt, test.txt
    ├── vocab.json
    └── tokens/                  # token caches per split
```

### WHU-CDC (binary change segmentation + captioning)

```
<WHU_CDC_ROOT>/
├── images/
│   ├── train/{A, B, label}/*.png
│   ├── val/{A, B, label}/*.png
│   └── test/{A, B, label}/*.png
└── whuCCcaptions.json           # caption JSON (one entry per image)
```


## Training

The training entry handles both datasets through a single `--dataset` switch. Paper hyperparameters are baked in; only data paths, the optimizer schedule, and runtime knobs are exposed on the CLI.

```bash
# LEVIR-MCI
python train.py \
    --dataset levir \
    --data_folder  <LEVIR_MCI_ROOT>/images \
    --list_path    <LEVIR_MCI_ROOT>/<list_path> \
    --token_folder <LEVIR_MCI_ROOT>/<list_path>/tokens \
    --savepath ./Results/levir_run/

# WHU-CDC
python train.py \
    --dataset whu \
    --data_folder  <WHU_CDC_ROOT>/images \
    --caption_json <WHU_CDC_ROOT>/whuCCcaptions.json \
    --savepath ./Results/whu_run/
```


## Evaluation

```bash
# Print aggregate metrics only
python test.py \
    --dataset levir \
    --data_folder  <LEVIR_MCI_ROOT>/images \
    --list_path    <LEVIR_MCI_ROOT>/<list_path> \
    --token_folder <LEVIR_MCI_ROOT>/<list_path>/tokens \
    --checkpoint ./Results/levir_run/.../best_SUM_mIoU_B4_CIDEr.pth

# Also dump per-image predictions: colored mask PNGs + caption TXTs
python test.py \
    --dataset levir \
    --checkpoint ./Results/levir_run/.../best_SUM_mIoU_B4_CIDEr.pth \
    --save_pred --result_path ./predict_result/
```

Reported metrics: `mIoU` (or `cIoU` for binary WHU-CDC), `BLEU-1..4`, `METEOR`, `ROUGE-L`, `CIDEr-D`.

When `--save_pred` is set:

```
<result_path>/<ckpt_name>/
├── masks/<key>.png             # colored mask
└── captions/<key>_cap.txt      # "pred: ...\nref: ..."
```

Mask coloring:
- **LEVIR-MCI** — black (background) / yellow (road) / red (building)
- **WHU-CDC**  — black (background) / white (change)
