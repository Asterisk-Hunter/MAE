"""Interactive MAE reconstruction demo (Gradio).

Upload an image (or pull a random CIFAR-10 test sample), then scrub the mask
ratio from 10% to 90% and watch which patches the model can complete.

    python demo.py                                   # uses checkpoints/mae_cifar10.pt
    python demo.py --ckpt checkpoints/mae_cifar10.pt --port 7860
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import gradio as gr
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from mae.data import build_dataset
from mae.model import MaskedAutoencoderViT

DISPLAY_SCALE = 8  # nearest-neighbour upscale so 32x32 output is visible

_state: dict = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive MAE mask-ratio demo")
    p.add_argument("--ckpt", default="checkpoints/mae_cifar10.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true")
    p.add_argument("--inbrowser", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--sweep", metavar="OUT.png", default=None,
                   help="render a mask-ratio sweep grid to this PNG and exit")
    return p.parse_args()


# ------------------------------------------------------------------ helpers
def to_tensor(pil: Image.Image, img_size: int) -> torch.Tensor:
    img = pil.convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BICUBIC)
    return TF.to_tensor(img)


def to_pil(tensor: torch.Tensor, scale: int = 1) -> Image.Image:
    arr = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(arr)
    if scale != 1:
        img = img.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    return img


def psnr(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor | None = None) -> float:
    diff = (a - b) ** 2
    diff = ((diff * weight).sum() / (weight.sum() + 1e-8)) if weight is not None else diff.mean()
    return 10.0 * math.log10(1.0 / (diff.item() + 1e-8))


def random_sample() -> Image.Image:
    ds = _state["test_ds"]
    idx = int(torch.randint(len(ds), (1,)).item())
    img, _ = ds[idx]
    return TF.to_pil_image(img)


# ------------------------------------------------------------------ callback
@torch.no_grad()
def run_entry(pil: Image.Image | None, mask_ratio: float, seed: float):
    model, cfg, device = _state["model"], _state["cfg"], _state["device"]
    if pil is None:
        pil = random_sample()

    x = to_tensor(pil, cfg["img_size"]).unsqueeze(0).to(device)
    # A seeded generator makes the permutation reproducible, so visible patches
    # stay nested as the ratio grows -> scrubbing reads as progressive removal.
    gen = torch.Generator(device=device).manual_seed(int(seed))
    rec, masked, mask = model.reconstruct(x, mask_ratio=float(mask_ratio), generator=gen)

    patch_dim = model.patch_embed.patch_dim
    mask_px = model.unpatchify(mask.unsqueeze(-1).expand(-1, -1, patch_dim))
    n_visible = int((1 - mask).sum().item())

    txt = (
        f"**{mask_ratio:.0%} masked** — encoder sees {n_visible}/"
        f"{model.patch_embed.num_patches} patches "
        f"({model.patch_embed.grid_size}x{model.patch_embed.grid_size} grid)\n\n"
        f"PSNR whole image: `{psnr(rec, x):.2f} dB` · "
        f"PSNR masked region: `{psnr(rec, x, mask_px):.2f} dB`"
    )
    if not _state["trained"]:
        txt += ("\n\n> ⚠️ **Random weights** — no checkpoint at "
                f"`{_state['ckpt_path']}`. Run `python train.py` first.")

    strip = torch.cat([x, masked, rec], dim=3)
    return (to_pil(masked[0], DISPLAY_SCALE), to_pil(rec[0], DISPLAY_SCALE),
            to_pil(strip[0], DISPLAY_SCALE), txt)


def render_sweep(path: str, ratios: tuple[float, ...] = (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9),
                 pil: Image.Image | None = None) -> None:
    """Write a labelled `original | masked | reconstruction` grid over mask ratios."""
    model, cfg, device = _state["model"], _state["cfg"], _state["device"]
    if pil is None:
        pil = random_sample()
    x = to_tensor(pil, cfg["img_size"]).unsqueeze(0).to(device)

    rows = []
    with torch.no_grad():
        for r in ratios:
            gen = torch.Generator(device=device).manual_seed(0)
            rec, masked, _ = model.reconstruct(x, mask_ratio=float(r), generator=gen)
            rows.append(torch.cat([x[0], masked[0], rec[0]], dim=2))
    grid = to_pil(torch.cat(rows, dim=1), DISPLAY_SCALE)

    cell = cfg["img_size"] * DISPLAY_SCALE
    header = 18
    canvas = Image.new("RGB", (cell * 3, header + cell * len(ratios)), (0, 0, 0))
    canvas.paste(grid, (0, header))
    draw = ImageDraw.Draw(canvas)
    for j, name in enumerate(["original", "masked input", "reconstruction"]):
        draw.text((j * cell + 4, 4), name, fill=(255, 255, 255))
    for i, r in enumerate(ratios):
        draw.text((4, header + i * cell + 4), f"{r:.0%}", fill=(0, 255, 0))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MAE — Masked Autoencoder explorer") as demo:
        gr.Markdown(
            "# Masked Autoencoder explorer\n"
            "A ViT encoder sees only the **visible** patches; a light decoder predicts the "
            "pixels of the **masked** ones. Scrub the mask ratio to find where semantic "
            "completion starts to break down."
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="Input image (or pick a CIFAR-10 sample)",
                                  height=220)
                pick = gr.Button("🎲 Random CIFAR-10 sample")
                ratio = gr.Slider(0.1, 0.9, value=0.75, step=0.05, label="Mask ratio")
                seed = gr.Number(value=0, precision=0, label="Mask seed (keeps the mask fixed)")
            with gr.Column(scale=1):
                out_masked = gr.Image(label="Masked input (what the encoder sees)", height=200)
                out_rec = gr.Image(label="Reconstruction", height=200)
                out_strip = gr.Image(label="Original | Masked | Reconstruction", height=140)
                out_txt = gr.Markdown()

        outputs = [out_masked, out_rec, out_strip, out_txt]
        inputs = [img_in, ratio, seed]
        img_in.change(run_entry, inputs, outputs)
        ratio.release(run_entry, inputs, outputs)
        seed.change(run_entry, inputs, outputs)
        pick.click(random_sample, None, img_in).then(run_entry, inputs, outputs)
        demo.load(random_sample, None, img_in).then(run_entry, inputs, outputs)
    return demo


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt_path = Path(args.ckpt)
    trained = ckpt_path.exists()
    if trained:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", MaskedAutoencoderViT.default_config())
        print(f"loaded checkpoint {ckpt_path} (epoch {ckpt.get('epoch', '?')}, "
              f"loss {ckpt.get('loss', float('nan')):.4f})")
    else:
        cfg = MaskedAutoencoderViT.default_config()
        print(f"[warn] no checkpoint at {ckpt_path}; running with random weights")

    model = MaskedAutoencoderViT(**cfg).to(device).eval()
    if trained:
        model.load_state_dict(ckpt["model"])

    _state.update(
        model=model, cfg=cfg, device=device, trained=trained, ckpt_path=ckpt_path,
        test_ds=build_dataset(args.data_root, train=False, img_size=cfg["img_size"]),
    )

    if args.sweep:
        render_sweep(args.sweep)
        print(f"wrote {args.sweep}")
        return

    build_ui().launch(server_name=args.host, server_port=args.port,
                      share=args.share, inbrowser=args.inbrowser)


if __name__ == "__main__":
    main()
