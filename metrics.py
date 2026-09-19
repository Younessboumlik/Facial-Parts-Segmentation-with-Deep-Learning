"""Correct segmentation metrics for the LaPa 11-class task.

Why this file exists
--------------------
The metrics defined in the training notebook (`iou`, `dice_coefficient`,
`precision`, `recall`) all collapse to variants of global pixel accuracy:

    intersection = count(y_true == y_pred)      # every correct pixel, background included
    union        = count(y_true != y_pred)      # every wrong pixel

so `iou` returns correct / (correct + wrong), which is pixel accuracy, and
`dice_coefficient` returns exactly twice that (hence the logged values > 1).
`precision` and `recall` divide that same correct-pixel count by the number of
*non-background* pixels, which is why they logged around 3.0.

Real segmentation metrics are computed per class from a confusion matrix, then
averaged. That is what this module does.

Definitions (per class c)
-------------------------
    TP = pixels of class c predicted as c
    FP = pixels of another class predicted as c
    FN = pixels of class c predicted as something else

    IoU (Jaccard) = TP / (TP + FP + FN)
    Dice (F1)     = 2*TP / (2*TP + FP + FN)
    Precision     = TP / (TP + FP)      "when I said c, how often was I right"
    Recall        = TP / (TP + FN)      "of all real c, how much did I find"

Mean IoU is the unweighted mean of per-class IoU. Reporting it *without* the
background class matters here: background is roughly two thirds of every LaPa
image, so including it inflates the average and hides failures on small parts
like the eyes and lips.
"""

import numpy as np

NUM_CLASSES = 11

CLASS_NAMES = [
    "background", "skin", "left eyebrow", "right eyebrow", "left eye",
    "right eye", "nose", "upper lip", "inner mouth", "lower lip", "hair",
]


class SegmentationMetrics:
    """Accumulates a confusion matrix across batches, then reports per-class scores."""

    def __init__(self, num_classes: int = NUM_CLASSES, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES[:num_classes]
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Add one batch. Both arrays hold class indices, any shape."""
        t = np.asarray(y_true).ravel().astype(np.int64)
        p = np.asarray(y_pred).ravel().astype(np.int64)
        if t.shape != p.shape:
            raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")

        valid = (t >= 0) & (t < self.num_classes) & (p >= 0) & (p < self.num_classes)
        t, p = t[valid], p[valid]

        # Row = ground truth, column = prediction.
        flat = np.bincount(
            t * self.num_classes + p, minlength=self.num_classes ** 2
        )
        self.confusion += flat.reshape(self.num_classes, self.num_classes)

    # -- derived quantities -------------------------------------------------

    @property
    def tp(self) -> np.ndarray:
        return np.diag(self.confusion).astype(np.float64)

    @property
    def fp(self) -> np.ndarray:
        return self.confusion.sum(axis=0).astype(np.float64) - self.tp

    @property
    def fn(self) -> np.ndarray:
        return self.confusion.sum(axis=1).astype(np.float64) - self.tp

    @staticmethod
    def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        """Classes absent from both truth and prediction give NaN, not 0.

        NaN keeps them out of the mean instead of dragging it down with a score
        that was never actually measured.
        """
        out = np.full_like(num, np.nan, dtype=np.float64)
        np.divide(num, den, out=out, where=den > 0)
        return out

    def per_class_iou(self) -> np.ndarray:
        return self._safe_divide(self.tp, self.tp + self.fp + self.fn)

    def per_class_dice(self) -> np.ndarray:
        return self._safe_divide(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    def per_class_precision(self) -> np.ndarray:
        return self._safe_divide(self.tp, self.tp + self.fp)

    def per_class_recall(self) -> np.ndarray:
        return self._safe_divide(self.tp, self.tp + self.fn)

    def pixel_accuracy(self) -> float:
        total = self.confusion.sum()
        return float(self.tp.sum() / total) if total else float("nan")

    def mean_iou(self, include_background: bool = False) -> float:
        iou = self.per_class_iou()
        if not include_background:
            iou = iou[1:]
        return float(np.nanmean(iou))

    def mean_dice(self, include_background: bool = False) -> float:
        dice = self.per_class_dice()
        if not include_background:
            dice = dice[1:]
        return float(np.nanmean(dice))

    def frequency_weighted_iou(self) -> float:
        """Mean IoU weighted by how often each class actually appears."""
        freq = self.confusion.sum(axis=1) / max(self.confusion.sum(), 1)
        iou = self.per_class_iou()
        mask = ~np.isnan(iou)
        return float((freq[mask] * iou[mask]).sum())

    # -- reporting ----------------------------------------------------------

    def report(self) -> dict:
        return {
            "per_class": [
                {
                    "class": name,
                    "iou": float(i),
                    "dice": float(d),
                    "precision": float(p),
                    "recall": float(r),
                    "pixels": int(self.confusion[idx].sum()),
                }
                for idx, (name, i, d, p, r) in enumerate(zip(
                    self.class_names,
                    self.per_class_iou(),
                    self.per_class_dice(),
                    self.per_class_precision(),
                    self.per_class_recall(),
                ))
            ],
            "mean_iou_no_bg": self.mean_iou(False),
            "mean_iou_with_bg": self.mean_iou(True),
            "mean_dice_no_bg": self.mean_dice(False),
            "frequency_weighted_iou": self.frequency_weighted_iou(),
            "pixel_accuracy": self.pixel_accuracy(),
        }

    def to_markdown(self) -> str:
        rep = self.report()
        lines = [
            "| Class | IoU | Dice | Precision | Recall | Pixels |",
            "|---|---|---|---|---|---|",
        ]
        for row in rep["per_class"]:
            def f(v):
                return "n/a" if np.isnan(v) else f"{v:.4f}"
            lines.append(
                f"| {row['class']} | {f(row['iou'])} | {f(row['dice'])} | "
                f"{f(row['precision'])} | {f(row['recall'])} | {row['pixels']:,} |"
            )
        lines += [
            "",
            f"- **Mean IoU (excl. background):** {rep['mean_iou_no_bg']:.4f}",
            f"- Mean IoU (incl. background): {rep['mean_iou_with_bg']:.4f}",
            f"- Mean Dice (excl. background): {rep['mean_dice_no_bg']:.4f}",
            f"- Frequency-weighted IoU: {rep['frequency_weighted_iou']:.4f}",
            f"- Pixel accuracy: {rep['pixel_accuracy']:.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# For future training runs
# ---------------------------------------------------------------------------
# Do not reuse the notebook's metric functions. Keras ships a correct streaming
# mean-IoU that works directly with one-hot targets and softmax outputs:
#
#     model.compile(
#         loss="categorical_crossentropy",
#         optimizer=tf.keras.optimizers.Adam(1e-4),
#         metrics=[
#             tf.keras.metrics.OneHotMeanIoU(num_classes=11, name="mean_iou"),
#             tf.keras.metrics.CategoricalAccuracy(name="pixel_acc"),
#         ],
#     )
#
# OneHotMeanIoU accumulates a confusion matrix over the whole epoch, so unlike a
# per-batch average it gives the true dataset-level score.
