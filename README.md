# WithGlasses_OR_NoGlasses

A Convolutional Neural Network (CNN) project to classify images of people **with glasses** or **without glasses**. This project includes dataset preparation, model building, training, and a single-image prediction function.

---

## 📖 Project Overview

The goal of this project is to **automatically detect whether a person is wearing glasses** in an image. This is a typical **binary image classification problem** using deep learning techniques.

---

## 🧩 Problem Statement

- **Task:** Classify an image as either **“Glasses”** or **“No Glasses”**.  
- **Challenges faced:**
  1. Dataset imbalance: Some classes had fewer images than others.  
  2. Misclassification on single-image predictions initially (model giving opposite outputs).  
  3. Small dataset size led to overfitting during early experiments.  
- **Things to watch out for:**
  - Ensure dataset is well-labeled and balanced.  
  - Check preprocessing and augmentation to avoid mislabeled predictions.  
  - Be cautious with training/validation split to prevent data leakage.

---

## 🛠 Tools & Libraries Used

- **Python 3**  
- **TensorFlow & Keras** (CNN model building)  
- **NumPy & Pandas** (Data manipulation)  
- **Matplotlib** (Plotting, optional for visualizations)  
- **KaggleHub** (Simplified dataset download from Kaggle)  
- **Google Colab** (Development environment)  

---

## 📥 Dataset

- Dataset downloaded from Kaggle using **KaggleHub**.  
- Folder structure:  
```text
dataset/
  train/
    glasses/
    no_glasses/
  test/
    glasses/
    no_glasses/
```


# 📚 Project Steps

## 1. Import Libraries

Imported everything needed for:

- Data augmentation (`ImageDataGenerator`)  
- Model building (`Sequential`)  
- CNN layers (`Conv2D`, `MaxPooling2D`)  
- Single-image prediction (`load_img`, `img_to_array`)  

## 2. Define Dataset Paths

- Specified paths for training and testing images.  
- Optionally created `output_folder` for balanced dataset.  

## 3. Data Preparation & Augmentation

- Normalized images to **0–1 range**.  
- Applied random transformations (rotation, shift, shear, zoom, flip) to prevent overfitting.  
- **Benefits:** Robust model even with a small dataset.  

## 4. Build the CNN Model

- Added convolutional layers to extract features.  
- Added pooling layers to reduce spatial dimensions.  
- Flattened features for Dense layers.  
- Used **sigmoid** activation for binary classification.  
- **Optional:** Transfer Learning could have improved accuracy to 95%+ with small datasets.  

## 5. Train the Model

- Trained the CNN using `fit()` with training and validation generators.  
- **Optional callbacks:**  
  - **EarlyStopping:** Prevents overfitting and saves time.  
  - **ModelCheckpoint:** Saves the best model only.  

## 6. Save the Model

- Saved trained model as `glasses_classifier.keras`.  

## 7. Prediction Function

- Preprocesses a single image, predicts label, and prints **“Glasses”** or **“No Glasses”**.  
- Ensures normalization and batch dimension handling for accurate prediction.  

---

## ⚠️ Notes & Things to Watch

- Make sure the dataset paths are correctly defined in Colab or your local environment.  
- Augmentation parameters can be tuned for better generalization.  
- For small datasets, overfitting is common; use dropout, early stopping, and optionally transfer learning.  
- Single-image predictions may be inverted if the training dataset labeling or preprocessing was inconsistent.  

---

## 💡 Possible Improvements

- **Use Transfer Learning:** Pretrained models like VGG16, ResNet50 could achieve higher accuracy with fewer epochs.  
- **Increase Dataset Size:** Collect more images for each class to reduce overfitting.  
- **Hyperparameter Tuning:** Adjust learning rate, batch size, augmentation parameters for better performance.  
- **Add GUI:** Integrate with Streamlit or Flask for interactive single-image prediction.  
- **Multi-class Expansion:** Extend to classify different types of glasses (sunglasses, reading glasses, etc.).  

---

## 🖥 How to Run

1. Clone the repository:  
```bash
git clone https://github.com/JubayerRislam/WithGlasses_OR_NoGlasses.git
```
