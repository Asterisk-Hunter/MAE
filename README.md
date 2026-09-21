# Masked Autoencoders (MAE) — CIFAR-10

A from-scratch reimplementation of **Masked Autoencoders Are Scalable Vision Learners**
(He, Chen, Xie, Li, Dollár, Girshick — CVPR 2022, [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)),
sized to pretrain locally on a 6 GB laptop GPU, plus an interactive demo for
probing how masking ratio affects reconstruction.

## The idea, as implemented

- Images are cut into regular non-overlapping patches; a **high mask ratio (75% by
  default)** leaves only a sparse set of visible patches.
- **Asymmetric architecture.** A deep ViT *encoder* runs on the visible patches only,
  so attention cost falls by the square of the keep rate — at 75% masking that is a
  16x reduction. A *lightweight* decoder receives the encoded visible tokens plus
  learnable `[mask]` tokens and predicts raw pixels for every patch.
- **Loss is MSE on the masked patches only** — the model never gets credit for
  copying visible pixels.

## Layout

| File | Purpose |
| --- | --- |
| `mae/model.py` | `MaskedAutoencoderViT`: patch embedding, random masking, asymmetric encoder/decoder, masked MSE, `unpatchify`, `encode()` |
| `mae/data.py` | CIFAR-10 loaders returning tensors in `[0, 1]` |
| `train.py` | Self-supervised pretraining (AdamW, warmup + cosine, AMP, checkpointing) |
| `demo.py` | Gradio mask-ratio explorer + `--sweep` grid renderer |

## Setup

```bash
pip install -r requirements.txt
```

PyTorch with CUDA is best installed from the official index for your GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Pretrain

```bash
python train.py                            # 100 epochs, ~13 min on an RTX 4050
python train.py --epochs 30                # ~4 min
python train.py --epochs 1 --limit-batches 5   # smoke test
```

Checkpoints land in `checkpoints/mae_cifar10.pt` (a dict of `model` weights, the
`config` needed to rebuild the architecture, and the epoch/loss).

## Interactive demo

```bash
python demo.py                             # http://127.0.0.1:7860
```

Upload any image, or hit **🎲 Random CIFAR-10 sample**. Then:

- Scrub the **mask ratio from 10% to 90%** to inspect the semantic completion
  threshold — where reconstructions stop being plausible and collapse toward blur.
- The **mask seed** holds the random permutation fixed, so raising the ratio
  *removes patches progressively* rather than reshuffling the whole mask. Visible
  sets are strictly nested as the ratio grows.
- Reported PSNR covers the whole image and the masked region separately.

Render a labelled sweep grid instead of launching a server:

```bash
python demo.py --sweep outputs/mask_ratio_sweep.png
```

That writes `original | masked input | reconstruction` rows for 10%…90% masking.

## Measured performance

RTX 4050 Laptop (6 GB), batch 256, AMP fp16, `--workers 4`:

| | |
| --- | --- |
| Parameters | 6.17 M total (5.35 M encoder) |
| Tokens | 64 (32 px / 4 px), 16 visible at 75% |
| Steady-state epoch | **~7.4 s** (195 steps) |
| 30 epochs | ~4 min |
| 100 epochs | ~13 min |
| Peak VRAM | **0.79 GB** |

> Data loading, not the GPU, is the bottleneck. Keep the default `--workers 4`
> (Windows `spawn` works because `train.py` guards its entry point). With
> `num_workers=0` the same step takes ~4x longer as the GPU waits on collation.

## Scaling up

Peak VRAM is 0.79 GB against a 6 GB budget, so there is large headroom — better
reconstructions are cheap here:

```bash
python train.py --embed-dim 384 --depth 12 --num-heads 6 --decoder-embed-dim 256
```

Other knobs worth trying: `--patch-size 8` (16 coarser tokens, faster), or
`--patch-size 2` (256 fine tokens, sharper output, ~16x encoder attention cost),
and `--norm-pix-loss` to normalize each target patch before the MSE, as in the paper.

## Design notes

- **Why patch size 4, not 16.** The paper's 16x16 patches assume 224x224 inputs. On
  32x32 CIFAR-10 that yields only 4 tokens, so a 75% mask leaves a *single* visible
  patch — degenerate. Patch size 4 gives a 8x8 = 64 token grid, and patch size 8
  gives 16. Both are configurable.
- **Pixel space.** Images stay in `[0, 1]` instead of being mean/std normalized, so
  the training loss and the demo visualizations share one consistent space.
- **Positional embeddings** are fixed 2D sine-cosine buffers (not learned), matching
  the original paper.
- **Masking is per-sample** in training: every image gets its own random permutation.
  The demo instead seeds the generator so the permutation is reproducible.
- **`model.encode(imgs)`** returns mean-pooled encoder features for a linear probe.
  Note that the encoder has no `[CLS]` token, as in the paper.
- The model's input resolution matches the checkpoint, so the demo resizes uploaded
  images to 32x32. Output is upscaled 8x with nearest-neighbour for display; blockiness
  is inherent to 32x32 data, not a bug.

## Environment note

`requirements.txt` pins `gradio<6`. Gradio 6 requires `anyio>=4` (this machine has
anyio 3.7.1, which raises `AsyncLibraryNotFoundError` when `Blocks` is constructed in a
sync script) and pulls in Pillow 12, which breaks streamlit's `pillow<12` pin. Gradio
5.50 also restores Pillow 11.

## Reference

He, Chen, Xie, Li, Dollár, Girshick. *Masked Autoencoders Are Scalable Vision Learners.*
CVPR 2022 (Oral). [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) ·
[PDF](https://arxiv.org/pdf/2111.06377.pdf)
