# Facial Parts Segmentation with Deep Learning

This project focuses on facial parts segmentation using deep learning techniques. We conducted a comparative analysis of several state-of-the-art architectures to evaluate their effectiveness in detecting and segmenting facial features such as eyes, nose, mouth, eyebrows, and jawline.

## 👥 Team Members

- Youness Boumlik  
- Abdellah Boulidam  
- Zakaria El Houari  
- Imane El Warraqi  
- Nassima Rhannouch  

## 🎯 Objective

The aim of this project is to explore and compare different convolutional neural network (CNN) architectures for accurate and efficient facial feature segmentation. This includes identifying the most suitable models in terms of precision, robustness, and performance across various conditions.

## 📚 Dataset

We used the **LAPA dataset** (Labeled Anatomic Parts of the face), which contains high-resolution images annotated for pixel-wise facial part segmentation. The dataset was preprocessed and split for training and evaluation.

## 🧠 Models Compared

We evaluated the following deep learning architectures:
- **U-Net**
- **SegNet**
- **PSPNet**

Each model was trained using the same preprocessing pipeline and hyperparameter tuning strategy to ensure fair comparison.

## ⚙️ Methodology

1. **Data preprocessing**  
2. **Model selection and configuration**  
3. **Training and evaluation**  
4. **Qualitative and quantitative analysis of results**

## 📊 Evaluation Metrics

- **IoU (Intersection over Union)**
- **Pixel Accuracy**
- **RMSE (Root Mean Square Error)**

## 🧪 Results

The models were assessed on their ability to segment facial components under normal and occluded conditions. U-Net showed promising results in handling fine-grained details, while PSPNet performed better in preserving global structure.

## 🔍 Key Findings

- Each architecture has unique strengths and weaknesses depending on the type of facial features and occlusions.
- Ensemble or hybrid approaches may improve robustness in real-world applications.

## 📝 Conclusion

Facial segmentation is a crucial task in computer vision with applications in face recognition, AR, and medical imaging. Deep learning architectures like U-Net and PSPNet offer strong baselines for such tasks, but there's room for optimization and further research.
