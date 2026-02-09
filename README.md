# 🎭 Facial Parts Segmentation with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">
  <h3>Comparative Analysis of Deep Learning Architectures for Facial Feature Segmentation</h3>
  <p><i>A comprehensive study comparing U-Net, PSPNet, and SegNet for pixel-wise facial parts segmentation</i></p>
</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Team Members](#-team-members)
- [Objective](#-objective)
- [Dataset](#-dataset)
- [Models & Architectures](#-models--architectures)
- [Installation](#-installation)
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

- 👁️ Eyes
- 👃 Nose
- 👄 Mouth
- ✨ Eyebrows
- 🦴 Jawline
- 👂 Ears
- 🎨 Skin regions
- 💇 Hair

The project implements and compares three powerful semantic segmentation architectures: **U-Net**, **PSPNet**, and **SegNet**, all leveraging **MobileNetV2** as the backbone encoder for efficient feature extraction.

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
- **Strengths**: Excellent at preserving fine-grained details and spatial information
- **Use Case**: Ideal for applications requiring high precision in boundary detection

### 2. PSPNet (Pyramid Scene Parsing Network)

**PSPNet** incorporates a pyramid pooling module to capture multi-scale contextual information:

- **Encoder**: MobileNetV2 backbone
- **Pyramid Pooling Module**: Aggregates context at multiple scales (1×1, 2×2, 3×3, 6×6)
- **Progressive Upsampling**: Five-stage decoder for full resolution reconstruction
- **Strengths**: Superior at capturing global context and preserving overall structure
- **Use Case**: Best for scenarios requiring understanding of facial composition

### 3. SegNet

**SegNet** uses a symmetric encoder-decoder structure with pooling indices:

- **Architecture**: Five encoder-decoder blocks
- **Encoding**: Convolutional layers with max pooling
- **Decoding**: Upsampling with skip connections from corresponding encoder layers
- **Strengths**: Memory efficient and good for real-time applications
- **Use Case**: Suitable for deployment on resource-constrained devices

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

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

## 💻 Usage

### Training Models

The project includes a comprehensive Jupyter notebook (`training-source-code.ipynb`) that contains all the code for:

1. **Dataset Loading**: Automatic download and preprocessing
2. **Model Building**: Implementation of all three architectures
3. **Training**: Complete training pipeline with callbacks
4. **Evaluation**: Performance metrics and visualization

#### Running the Training Notebook

```bash
jupyter notebook training-source-code.ipynb
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
├── training-source-code.ipynb    # Main training notebook with all implementations
├── report.pdf                     # Detailed project report
├── README.md                      # This file
├── .gitattributes                 # Git attributes configuration
│
└── files/                         # Generated during training
    ├── unet_model.keras          # Trained U-Net model
    ├── pspnet_model.keras        # Trained PSPNet model
    ├── segnet_model.keras        # Trained SegNet model
    ├── unet_data.csv             # U-Net training logs
    ├── pspnet_data.csv           # PSPNet training logs
    └── segnet_data.csv           # SegNet training logs
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

We employ multiple metrics to comprehensively evaluate model performance:

### 1. **IoU (Intersection over Union)**
   - Measures overlap between predicted and ground truth masks
   - Range: 0 (no overlap) to 1 (perfect overlap)
   - Primary metric for segmentation quality

### 2. **Dice Coefficient**
   - Similar to IoU but more sensitive to small regions
   - Harmonic mean of precision and recall
   - Range: 0 to 1

### 3. **Precision**
   - Measures accuracy of positive predictions
   - Important for minimizing false positives

### 4. **Recall**
   - Measures completeness of positive predictions
   - Important for minimizing false negatives

### 5. **Categorical Cross-Entropy Loss**
   - Training objective function
   - Measures pixel-wise classification accuracy

---

## 🧪 Results

### Performance Summary

The models were assessed on their ability to segment facial components under various conditions:

#### **U-Net**
- ✅ **Strengths**: Excellent at handling fine-grained details and preserving spatial information
- ✅ Best performance on boundary detection
- ✅ Superior for small facial features (eyes, eyebrows)
- ⚠️ Moderate performance on global structure

#### **PSPNet**
- ✅ **Strengths**: Superior at preserving global facial structure
- ✅ Better contextual understanding through pyramid pooling
- ✅ Robust to scale variations
- ⚠️ Slightly slower inference time

#### **SegNet**
- ✅ **Strengths**: Memory efficient architecture
- ✅ Faster inference for real-time applications
- ✅ Good balance between accuracy and efficiency
- ⚠️ Moderate performance on complex occlusions

### Key Observations

- All models achieved competitive performance on the LAPA dataset
- U-Net excels at detail preservation
- PSPNet performs best for overall facial structure understanding
- SegNet offers the best speed-accuracy tradeoff

---

## 🔍 Key Findings

1. **Architecture-Specific Strengths**: Each architecture has unique strengths and weaknesses depending on the type of facial features and occlusions being segmented.

2. **Transfer Learning Benefits**: Using pretrained MobileNetV2 as the encoder significantly improves convergence speed and final performance.

3. **Multi-Scale Context**: PSPNet's pyramid pooling module provides advantages in understanding facial composition as a whole.

4. **Skip Connections**: U-Net's skip connections are crucial for preserving fine-grained spatial details.

5. **Real-World Applicability**: Ensemble or hybrid approaches may improve robustness in real-world applications with various lighting conditions and occlusions.

6. **Computational Efficiency**: SegNet offers a good balance for deployment in resource-constrained environments.

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
