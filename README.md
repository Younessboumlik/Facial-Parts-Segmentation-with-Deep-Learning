# 🎭 Facial Parts Segmentation with Deep Learning

[![Live Demo](https://img.shields.io/badge/▶_Live_Demo-Streamlit-FF4B4B.svg)](https://facial-parts-segmentation.streamlit.app/)
[![Models](https://img.shields.io/badge/🤗_Models-Hugging_Face-FFD21E.svg)](https://huggingface.co/YounessBoumlik/face-segmentation-models)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">
  <h3>Comparative Analysis of Deep Learning Architectures for Facial Feature Segmentation</h3>
  <p><i>A comprehensive study comparing U-Net, PSPNet, and SegNet for pixel-wise facial parts segmentation</i></p>
  <p><b><a href="https://facial-parts-segmentation.streamlit.app/">▶ Try the live demo</a></b> &nbsp;·&nbsp; upload a portrait and segment it with any of the three models</p>
</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Team Members](#-team-members)
- [Objective](#-objective)
- [Dataset](#-dataset)
- [Models & Architectures](#-models--architectures)
- [Installation](#-installation)
- [Live Demo](#-live-demo)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#%EF%B8%8F-methodology)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Applications](#-applications)
- [Contributing](#-contributing)
- [References](#-references)
- [License](#-license)

---

## 🔍 Overview

This project focuses on **facial parts segmentation** using state-of-the-art deep learning techniques. We conducted a comprehensive comparative analysis of several prominent CNN architectures to evaluate their effectiveness in detecting and segmenting facial features including:

- 👁️ Eyes (left and right, labelled separately)
- ✨ Eyebrows (left and right, labelled separately)
- 👃 Nose
- 👄 Upper lip, lower lip and inner mouth
- 🎨 Skin
- 💇 Hair
- ⬛ Background

That is the full LaPa label set: **11 classes**, background included.

The project implements and compares three semantic segmentation architectures: **U-Net**, **PSPNet** and **SegNet**. U-Net and PSPNet use an ImageNet-pretrained **MobileNetV2** encoder; SegNet is trained from scratch with its own symmetric encoder. Notably, the from-scratch model came out ahead — see [Results](#-results).

---

## 👥 Team Members

- **Youness Boumlik**
- **Abdellah Boulidam**
- **Zakaria El Houari**
- **Imane El Warraqi**
- **Nassima Rhannouch**

---

## 🎯 Objective

The primary aim of this project is to:

1. **Explore and compare** different convolutional neural network (CNN) architectures for accurate and efficient facial feature segmentation
2. **Identify the most suitable models** in terms of precision, robustness, and performance across various conditions
3. **Evaluate model performance** on pixel-wise segmentation tasks with 11 distinct facial part classes
4. **Provide insights** into the strengths and weaknesses of each architecture for real-world applications

---

## 📚 Dataset

### LAPA Dataset (Labeled Anatomic Parts of the face)

We use the **LAPA dataset**, a comprehensive dataset for facial part segmentation:

- **Images**: High-resolution facial images
- **Annotations**: Pixel-wise annotations for 11 facial part classes
- **Classes**: Background, skin, left eyebrow, right eyebrow, left eye, right eye, nose, upper lip, inner mouth, lower lip, hair
- **Split**: Training and validation sets for robust model evaluation
- **Preprocessing**: Images resized to 256×256 pixels for efficient training

The dataset can be downloaded from [Kaggle - LAPA Face Parsing Dataset](https://www.kaggle.com/datasets/kiranraghavendrauci/lapa-face-parsing-dataset).

---

## 🧠 Models & Architectures

### 1. U-Net (with MobileNetV2 Encoder)

**U-Net** is a popular encoder-decoder architecture originally designed for biomedical image segmentation:

- **Encoder**: MobileNetV2 pretrained on ImageNet
- **Skip Connections**: Direct connections from encoder to decoder at multiple scales
- **Decoder**: Progressive upsampling with concatenation of encoder features
- **Strengths**: Skip connections from the encoder restore spatial detail lost to
  downsampling; the finest skip is at 128×128
- **Measured**: mean IoU 0.7423 (excl. background), 2.8 s/img on CPU, 12.8 M params
- **Verdict**: Statistically tied with SegNet; the better choice when model size matters

### 2. PSPNet (Pyramid Scene Parsing Network)

**PSPNet** incorporates a pyramid pooling module to capture multi-scale contextual information:

- **Encoder**: MobileNetV2 backbone
- **Pyramid Pooling Module**: Aggregates context at multiple scales (1×1, 2×2, 3×3, 6×6)
- **Progressive Upsampling**: Five-stage decoder for full resolution reconstruction
- **Strengths**: Pyramid pooling aggregates context at several scales
- **Measured**: mean IoU 0.6190 (excl. background), 8.5 s/img on CPU, 10.7 M params
- **Verdict**: Last on every class and ~5× the inference cost of SegNet. With an 8×8
  bottleneck and no skip connections, fine detail is gone before the decoder starts —
  it loses over 20 IoU points on eyebrows and the upper lip

### 3. SegNet (trained from scratch)

**SegNet** uses a symmetric encoder-decoder structure, the decoder mirroring the encoder stage for stage:

- **Architecture**: Five encoder-decoder blocks
- **Encoding**: Convolutional layers with max pooling
- **Decoding**: Upsampling with skip connections from corresponding encoder layers
- **Strengths**: The decoder concatenates encoder features at the full 256×256
  resolution, retaining the most spatial detail of the three
- **Measured**: mean IoU 0.7488 (excl. background), 1.7 s/img on CPU, 26.7 M params
- **Verdict**: Best accuracy *and* fastest inference, despite the largest parameter
  count and no pretrained encoder. Its cost is memory, not time

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended for training, but CPU training is also supported - just slower)
- 8GB+ RAM (16GB+ recommended for GPU training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Younessboumlik/Facial-Parts-Segmentation-with-Deep-Learning.git
cd Facial-Parts-Segmentation-with-Deep-Learning
```

### Step 2: Install Dependencies

```bash
pip install tensorflow>=2.8.0
pip install opencv-python
pip install numpy
pip install matplotlib
pip install kagglehub
pip install scikit-learn
```

### Step 3: Download the Dataset

The training notebook includes code to download the LAPA dataset automatically using `kagglehub`. Alternatively, you can manually download it from Kaggle.

---

## 🚀 Live Demo

### ▶ [facial-parts-segmentation.streamlit.app](https://facial-parts-segmentation.streamlit.app/)

The app is deployed and ready to use — no setup required. Upload a portrait, pick
a model (or compare all three side by side), and get the segmentation mask plus an
adjustable overlay.

To run it locally instead (`app.py`):

The three `.keras` files total ~1 GB, too large for GitHub, so they are hosted on
the Hugging Face Hub and downloaded at runtime:

**🤗 [YounessBoumlik/face-segmentation-models](https://huggingface.co/YounessBoumlik/face-segmentation-models)**

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

To use a different repo, edit `DEFAULT_REPO_ID` in `app.py`, use the sidebar
field, or create `.streamlit/secrets.toml`:

```toml
HF_REPO_ID = "YounessBoumlik/face-segmentation-models"
# HF_TOKEN = "hf_..."   # only needed if the repo is private
```

A local `models/` directory, if present, takes priority over the Hub download.

**Implementation notes**

- Inputs are resized to 256×256 and kept in **BGR** order to match the training
  pipeline, which used `cv2.imread` without an RGB conversion.
- Models load with `compile=False`, so the notebook's custom metrics are not needed.
- Only one model is held in memory at a time, keeping the app inside the free
  Streamlit Cloud memory limit. CPU inference takes a few seconds per image.

---

## 💻 Usage

### Training Models

The project includes a comprehensive Jupyter notebook (`training.ipynb`) that contains all the code for:

1. **Dataset Loading**: Automatic download and preprocessing
2. **Model Building**: Implementation of all three architectures
3. **Training**: Complete training pipeline with callbacks
4. **Evaluation**: Performance metrics and visualization

#### Running the Training Notebook

```bash
jupyter notebook training.ipynb
```

Or upload to [Kaggle Notebooks](https://www.kaggle.com/code) for GPU acceleration.

### Key Training Parameters

```python
image_h, image_w = 256, 256    # Image dimensions
num_classes = 11                # Number of facial part classes
batch_size = 8                  # Batch size for training
lr = 1e-4                       # Learning rate
num_epochs = 10                 # Training epochs
```

### Model Architecture Usage

```python
# Build U-Net model
unet_model = build_unet(input_shape=(256, 256, 3), num_classes=11)

# Build PSPNet model
pspnet_model = build_pspnet(input_shape=(256, 256, 3), num_classes=11)

# Build SegNet model
segnet_model = build_segnet(input_shape=(256, 256, 3), num_classes=11)
```

### Making Predictions

```python
# Load a trained model
model = tf.keras.models.load_model('path/to/model.keras', 
                                   custom_objects={
                                       'iou': iou,
                                       'dice_coefficient': dice_coefficient,
                                       'precision': precision,
                                       'recall': recall
                                   })

# Load and preprocess an image
image = load_and_preprocess_image('path/to/image.jpg')
image_batch = np.expand_dims(image, axis=0)

# Make prediction
prediction = model.predict(image_batch)
mask = np.argmax(prediction[0], axis=-1)
```

### Visualizing Results

The notebook includes visualization functions to compare model predictions:

```python
visualize_predictions(
    models=[unet_model, pspnet_model, segnet_model],
    model_names=["U-Net", "PSPNet", "SegNet"],
    image_paths=test_images
)
```

---

## 📁 Project Structure

```
Facial-Parts-Segmentation-with-Deep-Learning/
│
├── training.ipynb          # Training notebook (all three models, with outputs)
├── app.py                  # Streamlit demo
├── metrics.py              # Correct per-class segmentation metrics
├── evaluate.py             # Scores saved weights on the LaPa validation set
├── requirements.txt        # Python dependencies
├── report.pdf              # Detailed project report
├── README.md               # This file
│
├── models/                 # Git-ignored. Optional local copy of the weights;
│                           # otherwise fetched from the Hugging Face Hub
└── LaPa/                   # Git-ignored. Dataset, downloaded not committed
```

---

## ⚙️ Methodology

Our comprehensive approach includes:

### 1. **Data Preprocessing**
   - Image resizing to 256×256 pixels
   - Normalization to [0, 1] range
   - One-hot encoding of segmentation masks
   - Data augmentation (optional)

### 2. **Model Selection and Configuration**
   - Three state-of-the-art architectures selected
   - MobileNetV2 pretrained encoder for transfer learning
   - Custom decoder implementations for each architecture

### 3. **Training Strategy**
   - Mixed precision training for efficiency
   - Categorical cross-entropy loss
   - Adam optimizer with learning rate scheduling
   - Early stopping and model checkpointing
   - Learning rate reduction on plateau

### 4. **Evaluation**
   - Quantitative metrics: IoU, Dice coefficient, Precision, Recall
   - Qualitative analysis: Visual comparison of predictions
   - Performance assessment under normal and occluded conditions

---

## 📊 Evaluation Metrics

> ⚠️ **Important correction.** The metric functions inside the training notebook
> (`iou`, `dice_coefficient`, `precision`, `recall`) are **not** correct
> implementations. All four compare `argmax` equality across the whole image
> without splitting by class, so `iou` is really global pixel accuracy,
> `dice_coefficient` is exactly twice that value, and `precision`/`recall`
> divide by a non-background pixel count and can exceed 1.0 — the training logs
> show values around 1.9 and 3.0, which is impossible for a bounded metric.
>
> They were only ever *reported* alongside the loss, never optimised against
> (training used `categorical_crossentropy`), so **the trained weights are
> unaffected**. The corrected implementations live in `metrics.py`, and
> `evaluate.py` recomputes real scores from the saved weights — no retraining
> required.

Scores are computed **per class** from a confusion matrix accumulated over the
whole validation set. For each class `c`:

- **TP** — pixels of class `c` predicted as `c`
- **FP** — pixels of another class predicted as `c`
- **FN** — pixels of class `c` predicted as something else

| Metric | Formula | Interpretation |
|---|---|---|
| **IoU** (Jaccard) | `TP / (TP + FP + FN)` | Overlap ÷ combined area. The strictest measure |
| **Dice** (F1) | `2·TP / (2·TP + FP + FN)` | Like IoU but more forgiving; always ≥ IoU |
| **Precision** | `TP / (TP + FP)` | When the model predicts `c`, how often is it right |
| **Recall** | `TP / (TP + FN)` | Of all true `c` pixels, how many were found |

All are bounded in [0, 1]. The headline figure is **mean IoU excluding
background**: background is roughly two thirds of every LaPa image, so including
it inflates the average and hides poor performance on small parts such as the
eyes and lips. Pixel accuracy and frequency-weighted IoU are reported as
secondary figures.

### Re-evaluating the models

```bash
python evaluate.py                        # all three models, 200 val images
python evaluate.py --limit 0              # the full validation set
python evaluate.py --models unet segnet   # a subset
python evaluate.py --data /path/to/LaPa   # skip the dataset download
```

Fetching the dataset needs `pip install kagglehub` (not required by the app, so
it is not in `requirements.txt`); `--data` skips it if LaPa is already local.
Results are written to `evaluation.md` and `evaluation.json`.

For future training runs, do not reuse the notebook's metric functions — Keras
ships a correct streaming implementation:

```python
metrics=[
    tf.keras.metrics.OneHotMeanIoU(num_classes=11, name="mean_iou"),
    tf.keras.metrics.CategoricalAccuracy(name="pixel_acc"),
]
```

---

## 🧪 Results

### Measurement protocol

All figures below were produced by `evaluate.py` on **500 of the 2,000 LaPa
validation images**, at 256×256, using the corrected per-class metrics in
`metrics.py`. They are **not** the numbers printed by the training notebook —
see the warning in [Evaluation Metrics](#-evaluation-metrics) for why those were
wrong.

**Training budget.** Each model was trained for **10 epochs at 256×256** on Kaggle's
free GPU tier, within its session time limit. No data augmentation, no class
weighting and no hyperparameter search were applied, and the learning-rate schedule
was left at its default. Validation loss was still improving at the final epoch for
all three models and no early-stopping trigger fired, so these results reflect a
deliberately modest compute budget rather than converged models.

### Summary

| Model | Mean IoU (no bg) | Mean Dice (no bg) | Pixel accuracy | Params | CPU inference |
|---|---|---|---|---|---|
| **SegNet** | **0.7488** | **0.8523** | 0.9734 | 26.7 M | **1.7 s/img** |
| U-Net | 0.7423 | 0.8479 | 0.9728 | 12.8 M | 2.8 s/img |
| PSPNet | 0.6190 | 0.7528 | 0.9603 | 10.7 M | 8.5 s/img |

SegNet and U-Net are effectively tied — 0.0065 mean IoU apart, which is well
inside run-to-run noise. PSPNet trails both by a wide margin.

Timings were measured on a laptop CPU that was not otherwise idle, so treat the
absolute values as indicative only. The **ordering** is robust — PSPNet is
roughly 5× slower than SegNet, a gap far larger than any scheduling noise.
Accuracy figures are unaffected, being deterministic given the weights and
inputs.

### Per-class IoU

| Class | U-Net | PSPNet | SegNet |
|---|---|---|---|
| background | 0.9754 | 0.9665 | 0.9742 |
| skin | 0.9185 | 0.8721 | **0.9257** |
| left eyebrow | 0.6782 | 0.4846 | **0.6918** |
| right eyebrow | 0.6755 | 0.4523 | **0.6899** |
| left eye | 0.6997 | 0.5361 | **0.7126** |
| right eye | **0.6971** | 0.5297 | 0.6572 |
| nose | 0.8861 | 0.8344 | **0.9076** |
| upper lip | 0.5907 | 0.4263 | **0.6350** |
| inner mouth | **0.7149** | 0.6168 | 0.7097 |
| lower lip | **0.6722** | 0.5876 | 0.6697 |
| hair | **0.8897** | 0.8500 | 0.8885 |

### Observations

- **Region size drives the score.** Every model handles background, skin, hair
  and nose strongly (0.83–0.98). Thin structures score lower — the upper lip is
  the hardest class for all three (0.43–0.64) — because a few pixels of boundary
  error barely move a large region's IoU but change a thin one's substantially.
  This is the expected behaviour of IoU on narrow classes at 256×256 input
  resolution, and is where additional resolution would pay off most.

- **Left/right pairs score almost identically** (SegNet eyebrows: 0.6918 vs
  0.6899), indicating no systematic left–right confusion — a useful sanity check
  on the label pipeline.

- **PSPNet loses exactly where spatial detail matters.** It is 20+ IoU points
  behind on eyebrows and the upper lip, but only 3–4 points behind on skin and
  hair. Its encoder bottleneck is 8×8 (256 ÷ 32) and it has no skip connections,
  so fine detail is discarded before the decoder ever runs. Pyramid pooling adds
  global context at the cost of the localisation this task needs.

- **The ranking tracks decoder resolution, not model size or pretraining.**
  SegNet's decoder concatenates encoder features at the full 256×256 resolution;
  U-Net's finest skip is at 128×128; PSPNet has none. That is the same order as
  the results.

- **Pixel accuracy is not a useful discriminator here.** All three land at
  0.96–0.97 while their mean IoU spans 0.62–0.75. This is precisely the
  flattering-but-uninformative metric the notebook was reporting as "IoU".

---

## 🔍 Key Findings

1. **Skip connections matter more than the backbone.** SegNet — trained from
   scratch, with no pretrained encoder — matched and slightly beat U-Net, which
   uses an ImageNet-pretrained MobileNetV2. What separated the models was how
   much spatial resolution the decoder could recover, not what the encoder had
   seen before.

2. **Aggressive downsampling is expensive for small parts.** PSPNet's 8×8
   bottleneck with no skip path costs it more than 20 IoU points on eyebrows and
   the upper lip, while costing only 3–4 points on large regions. Global context
   did not compensate.

3. **SegNet gives the best accuracy *and* the fastest inference** (1.7 s/img vs
   PSPNet's 8.5 s/img on CPU), despite having the most parameters at 26.7 M.
   Parameter count is a poor proxy for cost: SegNet's plain convolutions
   parallelise better than PSPNet's pyramid-pooling and transpose-conv stack.

4. **Thin, low-contrast classes are the real bottleneck.** The upper lip is the
   worst class for every model. Improving this task means targeting boundary
   quality — class-weighted or boundary-aware losses, higher input resolution —
   not adding capacity.

5. **The choice of metric changes the conclusion.** Ranked by pixel accuracy the
   three models look nearly identical (0.960–0.973). Ranked by mean IoU excluding
   background, PSPNet is clearly last. Reporting the wrong metric hid a real
   13-point quality gap.

6. **Only 10 epochs were trained.** None of the models had plateaued, and no
   early-stopping trigger fired. These figures are a floor, not a ceiling.

---

## 🎨 Applications

Facial parts segmentation has numerous practical applications:

### 1. **Face Recognition & Verification**
   - Enhanced feature extraction
   - Robust to partial occlusions

### 2. **Augmented Reality (AR)**
   - Face filters and effects
   - Virtual makeup application
   - Real-time face modification

### 3. **Medical Imaging**
   - Facial reconstruction planning
   - Anomaly detection
   - Cosmetic surgery simulation

### 4. **Animation & Entertainment**
   - Motion capture for facial animation
   - Character design and modeling
   - Video game development

### 5. **Biometrics & Security**
   - Enhanced authentication systems
   - Surveillance and monitoring
   - Identity verification

### 6. **Accessibility**
   - Emotion recognition
   - Facial expression analysis
   - Human-computer interaction

---

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution

- Additional model architectures (DeepLabV3+, HRNet, etc.)
- Data augmentation strategies
- Post-processing techniques
- Real-time inference optimization
- Mobile deployment (TensorFlow Lite)
- Web deployment (TensorFlow.js)
- Additional evaluation metrics
- Documentation improvements

---

## 📚 References

### Datasets
- **LAPA Dataset**: [Kaggle - LAPA Face Parsing Dataset](https://www.kaggle.com/datasets/kiranraghavendrauci/lapa-face-parsing-dataset)

### Architecture Papers
1. **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
2. **PSPNet**: Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid Scene Parsing Network. CVPR 2017.
3. **SegNet**: Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. TPAMI 2017.
4. **MobileNetV2**: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.

### Frameworks & Tools
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- [Kaggle Notebooks](https://www.kaggle.com/code)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to the creators of the LAPA dataset for providing high-quality annotations
- TensorFlow and Keras teams for excellent deep learning frameworks
- The research community for developing these powerful architectures
- Kaggle for providing computational resources

---

## 📞 Contact

For questions, suggestions, or collaborations, please contact the team members or open an issue in this repository.

---

<div align="center">
  <p>⭐ If you find this project useful, please consider giving it a star! ⭐</p>
  <p>Made with ❤️ by the Facial Segmentation Team</p>
</div>
