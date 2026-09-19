# Evaluation - LaPa validation set

Computed with `metrics.py` on 500 validation images at 256x256.

| Model | mIoU (no bg) | mDice (no bg) | fwIoU | Pixel acc |
|---|---|---|---|---|
| unet | 0.7423 | 0.8479 | 0.9485 | 0.9728 |
| pspnet | 0.6190 | 0.7528 | 0.9267 | 0.9603 |
| segnet | 0.7488 | 0.8523 | 0.9490 | 0.9734 |

## unet

| Class | IoU | Dice | Precision | Recall | Pixels |
|---|---|---|---|---|---|
| background | 0.9754 | 0.9876 | 0.9893 | 0.9858 | 21,971,271 |
| skin | 0.9185 | 0.9575 | 0.9559 | 0.9591 | 5,171,322 |
| left eyebrow | 0.6782 | 0.8083 | 0.8148 | 0.8019 | 76,525 |
| right eyebrow | 0.6755 | 0.8063 | 0.8084 | 0.8042 | 73,824 |
| left eye | 0.6997 | 0.8233 | 0.7989 | 0.8493 | 39,676 |
| right eye | 0.6971 | 0.8215 | 0.8077 | 0.8358 | 38,862 |
| nose | 0.8861 | 0.9396 | 0.9592 | 0.9208 | 377,211 |
| upper lip | 0.5907 | 0.7427 | 0.7183 | 0.7688 | 64,149 |
| inner mouth | 0.7149 | 0.8338 | 0.8448 | 0.8230 | 81,385 |
| lower lip | 0.6722 | 0.8040 | 0.7981 | 0.8100 | 103,726 |
| hair | 0.8897 | 0.9416 | 0.9350 | 0.9484 | 4,770,049 |

- **Mean IoU (excl. background):** 0.7423
- Mean IoU (incl. background): 0.7634
- Mean Dice (excl. background): 0.8479
- Frequency-weighted IoU: 0.9485
- Pixel accuracy: 0.9728

## pspnet

| Class | IoU | Dice | Precision | Recall | Pixels |
|---|---|---|---|---|---|
| background | 0.9665 | 0.9830 | 0.9849 | 0.9810 | 21,971,271 |
| skin | 0.8721 | 0.9317 | 0.9301 | 0.9333 | 5,171,322 |
| left eyebrow | 0.4846 | 0.6529 | 0.7077 | 0.6059 | 76,525 |
| right eyebrow | 0.4523 | 0.6229 | 0.6586 | 0.5908 | 73,824 |
| left eye | 0.5361 | 0.6980 | 0.7108 | 0.6856 | 39,676 |
| right eye | 0.5297 | 0.6926 | 0.6772 | 0.7087 | 38,862 |
| nose | 0.8344 | 0.9097 | 0.8903 | 0.9301 | 377,211 |
| upper lip | 0.4263 | 0.5978 | 0.6064 | 0.5895 | 64,149 |
| inner mouth | 0.6168 | 0.7630 | 0.7111 | 0.8231 | 81,385 |
| lower lip | 0.5876 | 0.7403 | 0.7382 | 0.7423 | 103,726 |
| hair | 0.8500 | 0.9189 | 0.9133 | 0.9246 | 4,770,049 |

- **Mean IoU (excl. background):** 0.6190
- Mean IoU (incl. background): 0.6506
- Mean Dice (excl. background): 0.7528
- Frequency-weighted IoU: 0.9267
- Pixel accuracy: 0.9603

## segnet

| Class | IoU | Dice | Precision | Recall | Pixels |
|---|---|---|---|---|---|
| background | 0.9742 | 0.9869 | 0.9853 | 0.9886 | 21,971,271 |
| skin | 0.9257 | 0.9614 | 0.9622 | 0.9606 | 5,171,322 |
| left eyebrow | 0.6918 | 0.8178 | 0.8837 | 0.7611 | 76,525 |
| right eyebrow | 0.6899 | 0.8165 | 0.8496 | 0.7859 | 73,824 |
| left eye | 0.7126 | 0.8322 | 0.8643 | 0.8024 | 39,676 |
| right eye | 0.6572 | 0.7932 | 0.8088 | 0.7782 | 38,862 |
| nose | 0.9076 | 0.9516 | 0.9540 | 0.9492 | 377,211 |
| upper lip | 0.6350 | 0.7767 | 0.8283 | 0.7312 | 64,149 |
| inner mouth | 0.7097 | 0.8302 | 0.8472 | 0.8139 | 81,385 |
| lower lip | 0.6697 | 0.8022 | 0.8234 | 0.7821 | 103,726 |
| hair | 0.8885 | 0.9410 | 0.9435 | 0.9385 | 4,770,049 |

- **Mean IoU (excl. background):** 0.7488
- Mean IoU (incl. background): 0.7693
- Mean Dice (excl. background): 0.8523
- Frequency-weighted IoU: 0.9490
- Pixel accuracy: 0.9734
