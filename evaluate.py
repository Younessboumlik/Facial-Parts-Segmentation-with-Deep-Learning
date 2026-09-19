"""Evaluate the trained models on the LaPa validation set with correct metrics.

The notebook's own metrics were mislabelled variants of pixel accuracy (see
metrics.py). This script recomputes real per-class IoU / Dice / precision /
recall from a confusion matrix accumulated over the whole validation set.

No retraining is involved: the weights are used exactly as they were saved.

Usage
-----
    python evaluate.py                       # all three models, 200 val images
    python evaluate.py --limit 0             # the full validation set
    python evaluate.py --models unet segnet  # a subset
    python evaluate.py --data /path/to/LaPa  # skip the kagglehub download
"""

import argparse
import json
import os
import time
from glob import glob

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

from metrics import NUM_CLASSES, SegmentationMetrics

IMAGE_H, IMAGE_W = 256, 256

MODEL_FILES = {
    "unet": "unet_model.keras",
    "pspnet": "pspnet_model.keras",
    "segnet": "segnet_model.keras",
}


def find_dataset(explicit: str | None) -> str:
    """Return the LaPa root (the directory containing val/images)."""
    if explicit:
        root = explicit
    else:
        import kagglehub
        print("Downloading LaPa via kagglehub (cached after the first run)...")
        root = kagglehub.dataset_download("kiranraghavendrauci/lapa-face-parsing-dataset")

    # The archive nests the data one level down in some versions.
    for candidate in (root, os.path.join(root, "LaPa")):
        if os.path.isdir(os.path.join(candidate, "val", "images")):
            return candidate
    raise SystemExit(f"Could not find val/images under {root}")


def load_pairs(root: str, limit: int):
    images = sorted(glob(os.path.join(root, "val", "images", "*.jpg")))
    labels = sorted(glob(os.path.join(root, "val", "labels", "*.png")))
    if len(images) != len(labels):
        raise SystemExit(f"{len(images)} images vs {len(labels)} labels - mismatch")
    if not images:
        raise SystemExit("No validation images found")
    if limit and limit > 0:
        images, labels = images[:limit], labels[:limit]
    return images, labels


def read_pair(image_path: str, label_path: str):
    """Identical preprocessing to training: BGR, resized, scaled to [0, 1]."""
    x = cv2.imread(image_path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_W, IMAGE_H))
    x = (x / 255.0).astype(np.float32)

    y = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # INTER_NEAREST keeps label values intact; interpolation would invent classes.
    y = cv2.resize(y, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_NEAREST)
    return x, y.astype(np.int32)


def resolve_model_path(name: str, repo_id: str | None) -> str:
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", MODEL_FILES[name])
    if os.path.exists(local):
        return local
    if not repo_id:
        raise SystemExit(
            f"{local} not found and no --repo given. Either keep models/ locally "
            f"or pass --repo <user>/<model-repo>."
        )
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=MODEL_FILES[name])


def evaluate_model(name: str, images, labels, batch_size: int, repo_id: str | None):
    import tensorflow as tf

    path = resolve_model_path(name, repo_id)
    print(f"\n{'='*66}\n{name.upper()}  <- {os.path.basename(path)}")
    model = tf.keras.models.load_model(path, compile=False)

    acc = SegmentationMetrics(NUM_CLASSES)
    started = time.time()

    for start in range(0, len(images), batch_size):
        chunk_x, chunk_y = [], []
        for ip, lp in zip(images[start:start + batch_size], labels[start:start + batch_size]):
            x, y = read_pair(ip, lp)
            chunk_x.append(x)
            chunk_y.append(y)

        preds = model.predict(np.stack(chunk_x), verbose=0)
        acc.update(np.stack(chunk_y), np.argmax(preds, axis=-1))

        done = min(start + batch_size, len(images))
        print(f"\r  {done}/{len(images)} images", end="", flush=True)

    elapsed = time.time() - started
    print(f"\r  {len(images)} images in {elapsed:.0f}s "
          f"({elapsed/len(images)*1000:.0f} ms/image)")
    print()
    print(acc.to_markdown())

    del model
    import gc
    gc.collect()
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODEL_FILES),
                    choices=list(MODEL_FILES))
    ap.add_argument("--limit", type=int, default=200,
                    help="validation images to use; 0 means all (default: 200)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--data", default=None, help="path to the LaPa root")
    ap.add_argument("--repo", default=None, help="HF repo to pull weights from")
    ap.add_argument("--out", default="evaluation.md", help="markdown report path")
    args = ap.parse_args()

    root = find_dataset(args.data)
    images, labels = load_pairs(root, args.limit)
    print(f"Dataset: {root}")
    print(f"Evaluating on {len(images)} validation images")

    reports, summary = {}, []
    for name in args.models:
        acc = evaluate_model(name, images, labels, args.batch_size, args.repo)
        reports[name] = acc
        rep = acc.report()
        summary.append((name, rep["mean_iou_no_bg"], rep["mean_dice_no_bg"],
                        rep["frequency_weighted_iou"], rep["pixel_accuracy"]))

    print(f"\n{'='*66}\nSUMMARY  ({len(images)} validation images)\n")
    header = f"{'model':<9}{'mIoU(no bg)':>13}{'mDice(no bg)':>14}{'fwIoU':>9}{'pixel acc':>11}"
    print(header)
    print("-" * len(header))
    for name, miou, mdice, fwiou, pa in summary:
        print(f"{name:<9}{miou:>13.4f}{mdice:>14.4f}{fwiou:>9.4f}{pa:>11.4f}")

    best = max(summary, key=lambda r: r[1])
    print(f"\nBest mean IoU (excluding background): {best[0]} at {best[1]:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Evaluation - LaPa validation set\n\n")
        f.write(f"Computed with `metrics.py` on {len(images)} validation images "
                f"at {IMAGE_H}x{IMAGE_W}.\n\n")
        f.write("| Model | mIoU (no bg) | mDice (no bg) | fwIoU | Pixel acc |\n")
        f.write("|---|---|---|---|---|\n")
        for name, miou, mdice, fwiou, pa in summary:
            f.write(f"| {name} | {miou:.4f} | {mdice:.4f} | {fwiou:.4f} | {pa:.4f} |\n")
        for name in args.models:
            f.write(f"\n## {name}\n\n{reports[name].to_markdown()}\n")
    print(f"Report written to {args.out}")

    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({n: reports[n].report() for n in args.models}, f, indent=2)


if __name__ == "__main__":
    main()
