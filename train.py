"""Pretrain a Masked Autoencoder on CIFAR-10.

    python train.py                      # default ViT-Tiny config, 100 epochs
    python train.py --epochs 30          # quicker run
    python train.py --limit-batches 5 --epochs 1   # smoke test
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from mae.data import build_loader
from mae.model import MaskedAutoencoderViT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAE pretraining on CIFAR-10")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="checkpoints/mae_cifar10.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-4, help="base LR at batch size 256")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=5.0)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    # architecture
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--decoder-embed-dim", type=int, default=128)
    p.add_argument("--decoder-depth", type=int, default=4)
    p.add_argument("--decoder-num-heads", type=int, default=4)
    p.add_argument("--norm-pix-loss", action="store_true",
                   help="normalize each target patch before MSE (as in the paper)")
    # runtime
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--limit-batches", type=int, default=0, help="cap steps/epoch (smoke tests)")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def lr_at(step: int, total: int, base_lr: float, warmup: int, min_frac: float = 0.05) -> float:
    """Linear warmup then cosine decay to ``min_frac`` of the base LR."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    config = dict(
        img_size=args.img_size, patch_size=args.patch_size, in_chans=3,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads, mlp_ratio=4.0,
        norm_pix_loss=args.norm_pix_loss,
    )
    model = MaskedAutoencoderViT(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.patch_embed.parameters()) + \
        sum(p.numel() for p in model.enc_blocks.parameters()) + \
        sum(p.numel() for p in model.enc_norm.parameters())

    loader = build_loader(args.data_root, img_size=args.img_size,
                          batch_size=args.batch_size, num_workers=args.workers)
    steps_per_epoch = args.limit_batches or len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    lr = args.lr * args.batch_size / 256.0
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim == 1 else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device      : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"amp         : {use_amp}")
    print(f"patches     : {model.patch_embed.num_patches} tokens "
          f"({args.img_size}px / {args.patch_size}px), "
          f"visible at mask {args.mask_ratio:.0%}: "
          f"{max(1, int(model.patch_embed.num_patches * (1 - args.mask_ratio)))}")
    print(f"encoder     : {n_enc/1e6:.2f}M  total: {n_params/1e6:.2f}M params")
    print(f"steps       : {steps_per_epoch}/epoch x {args.epochs} = {total_steps} "
          f"(warmup {warmup_steps})\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for i, (imgs, _) in enumerate(loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, lr, warmup_steps)

            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            running += loss.item()
            seen += 1
            if i == 0 or (i + 1) % args.log_every == 0:
                print(f"epoch {epoch+1:3d}/{args.epochs}  step {i+1:4d}/{steps_per_epoch}  "
                      f"loss {running/seen:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        avg = running / max(1, seen)
        torch.save({"model": model.state_dict(), "config": config,
                    "args": vars(args), "epoch": epoch + 1, "loss": avg,
                    "mask_ratio": args.mask_ratio}, out_path)
        flag = ""
        if avg < best:
            best = avg
            flag = "  (best)"
        print(f"[epoch {epoch+1}] avg loss {avg:.4f}  {time.time()-t0:.1f}s  -> {out_path}{flag}",
              flush=True)


if __name__ == "__main__":
    main()
